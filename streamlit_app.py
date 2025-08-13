import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import av
import tempfile
import requests
import numpy as np
from pydub import AudioSegment
import io
import threading
import queue
from streamlit_calendar import calendar
import datetime
import os
import pytz
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from google.auth.transport.requests import Request
import json
import logging
from typing import Optional, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
st.set_page_config(page_title="AI Task Scheduler (Chat + Voice)")

# Constants - FIXED SCOPES
SCOPES = [
    'https://www.googleapis.com/auth/calendar',  # Full calendar access
    'https://www.googleapis.com/auth/calendar.events'  # Events access
]

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
MAX_AUDIO_DURATION = 300  # 5 minutes max
MIN_AUDIO_DURATION = 1    # 1 second min
AUDIO_SAMPLE_RATE = 48000

class Config:
    """Configuration management"""
    def __init__(self):
        self.backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
        self.credentials_path = os.getenv("GOOGLE_CREDENTIALS_PATH", "credentials.json")
        self.token_path = os.getenv("GOOGLE_TOKEN_PATH", "token.json")
        self.default_timezone = os.getenv("DEFAULT_TIMEZONE", "Asia/Kolkata")
        self.max_retries = int(os.getenv("MAX_RETRIES", "3"))
        self.request_timeout = int(os.getenv("REQUEST_TIMEOUT", "30"))

config = Config()

class AudioProcessor:
    """Enhanced audio processor with quality checks and thread safety"""
    def __init__(self):
        self.audio_frames = []
        self.lock = threading.Lock()
        self.recording_start_time = None
        self.max_duration = MAX_AUDIO_DURATION
        self.min_duration = MIN_AUDIO_DURATION
        
    def start_recording(self):
        """Start recording with timestamp"""
        with self.lock:
            self.recording_start_time = datetime.datetime.now()
            self.audio_frames = []
    
    def stop_recording(self):
        """Stop recording and validate duration"""
        with self.lock:
            if self.recording_start_time:
                duration = (datetime.datetime.now() - self.recording_start_time).total_seconds()
                if duration < self.min_duration:
                    raise ValueError(f"Recording too short: {duration:.2f}s (minimum: {self.min_duration}s)")
                if duration > self.max_duration:
                    raise ValueError(f"Recording too long: {duration:.2f}s (maximum: {self.max_duration}s)")
            self.recording_start_time = None
    
    def add_frame(self, frame: av.AudioFrame) -> av.AudioFrame:
        """Add audio frame with thread safety"""
        with self.lock:
            if self.recording_start_time:
                # Check if recording duration exceeded
                duration = (datetime.datetime.now() - self.recording_start_time).total_seconds()
                if duration > self.max_duration:
                    logger.warning(f"Recording duration exceeded {self.max_duration}s, stopping...")
                    return frame
                
                audio_array = frame.to_ndarray()
                # Basic quality check - ensure audio has content
                if np.abs(audio_array).mean() > 0.001:  # Not just silence
                    self.audio_frames.append(audio_array)
        return frame
    
    def get_audio_data(self) -> Optional[bytes]:
        """Get processed audio data with quality validation"""
        with self.lock:
            if not self.audio_frames:
                return None
            
            try:
                combined_audio = np.concatenate(self.audio_frames)
                
                # Audio quality checks
                if len(combined_audio) == 0:
                    raise ValueError("No audio data captured")
                
                # Check for silence (very low amplitude)
                if np.abs(combined_audio).mean() < 0.001:
                    raise ValueError("Audio appears to be silent")
                
                # Check for clipping (audio too loud)
                if np.abs(combined_audio).max() > 0.95:
                    logger.warning("Audio may be clipped (too loud)")
                
                # Create audio segment
                audio_segment = AudioSegment(
                    combined_audio.tobytes(),
                    frame_rate=AUDIO_SAMPLE_RATE,
                    sample_width=combined_audio.dtype.itemsize,
                    channels=1 if len(combined_audio.shape) == 1 else combined_audio.shape[1]
                )
                
                # Normalize audio
                audio_segment = audio_segment.normalize()
                
                # Export to WAV
                wav_buffer = io.BytesIO()
                audio_segment.export(wav_buffer, format="wav")
                wav_buffer.seek(0)
                
                return wav_buffer.getvalue()
                
            except Exception as e:
                logger.error(f"Audio processing error: {e}")
                raise
    
    def clear(self):
        """Clear audio frames"""
        with self.lock:
            self.audio_frames = []
            self.recording_start_time = None

class GoogleCalendarManager:
    """Enhanced Google Calendar integration with error handling and scope fix"""
    def __init__(self):
        self.service = None
        self.timezone = config.default_timezone
    
    def reset_credentials(self):
        """Reset credentials by deleting token file"""
        try:
            if os.path.exists(config.token_path):
                os.remove(config.token_path)
                st.info("🔄 Credentials reset. Please re-authorize with Google Calendar.")
                return True
        except Exception as e:
            logger.error(f"Error resetting credentials: {e}")
            return False
        return False
    
    def get_calendar_service(self):
        """Get Google Calendar service with improved error handling and scope fix"""
        try:
            creds = None
            
            # Load existing credentials
            if os.path.exists(config.token_path):
                try:
                    creds = Credentials.from_authorized_user_file(config.token_path, SCOPES)
                except Exception as e:
                    logger.warning(f"Error loading existing credentials: {e}")
                    # Reset credentials and try again
                    self.reset_credentials()
                    creds = None
            
            # If no valid credentials, get new ones
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    try:
                        creds.refresh(Request())
                        logger.info("✅ Credentials refreshed successfully")
                    except Exception as e:
                        logger.error(f"Error refreshing credentials: {e}")
                        # Reset and re-authorize
                        self.reset_credentials()
                        creds = None
                
                if not creds:
                    if not os.path.exists(config.credentials_path):
                        raise FileNotFoundError(f"Google credentials file not found: {config.credentials_path}")
                    
                    try:
                        flow = InstalledAppFlow.from_client_secrets_file(config.credentials_path, SCOPES)
                        # Use run_local_server with specific parameters to avoid issues
                        creds = flow.run_local_server(
                            port=0,
                            prompt='consent',  # Force consent screen to ensure all scopes are granted
                            authorization_prompt_message="Please visit this URL to authorize the application: {url}",
                            success_message="Authorization successful! You can close this tab and return to the app."
                        )
                        logger.info("✅ New credentials obtained successfully")
                    except Exception as e:
                        logger.error(f"Error during OAuth flow: {e}")
                        raise
                
                # Save credentials
                with open(config.token_path, "w") as token:
                    token.write(creds.to_json())
                    logger.info("✅ Credentials saved successfully")
            
            # Build service
            self.service = build("calendar", "v3", credentials=creds)
            
            # Test the service with a simple call
            try:
                calendar_list = self.service.calendarList().list(maxResults=1).execute()
                logger.info("✅ Google Calendar service initialized and tested")
            except Exception as e:
                logger.error(f"Error testing calendar service: {e}")
                raise
            
            return self.service
            
        except Exception as e:
            logger.error(f"Google Calendar authentication error: {e}")
            raise
    
    def add_event(self, summary: str, start_time: str, end_time: str, description: str = "") -> Optional[str]:
        """Add event to Google Calendar with enhanced error handling"""
        try:
            if not self.service:
                self.get_calendar_service()
            
            # Parse and format datetime strings properly
            try:
                # Handle ISO format datetime strings
                if isinstance(start_time, str):
                    if 'T' in start_time and not start_time.endswith('Z') and '+' not in start_time:
                        # Add timezone if not present
                        start_time = f"{start_time}+05:30"  # IST offset
                
                if isinstance(end_time, str):
                    if 'T' in end_time and not end_time.endswith('Z') and '+' not in end_time:
                        # Add timezone if not present
                        end_time = f"{end_time}+05:30"  # IST offset
                        
            except Exception as e:
                logger.warning(f"Datetime formatting issue: {e}")
            
            event = {
                'summary': summary,
                'description': description,
                'start': {
                    'dateTime': start_time,
                    'timeZone': self.timezone
                },
                'end': {
                    'dateTime': end_time,
                    'timeZone': self.timezone
                },
                'reminders': {
                    'useDefault': True,
                },
                # Add some additional properties for better event management
                'visibility': 'private',
                'status': 'confirmed'
            }
            
            logger.info(f"Creating calendar event: {summary}")
            logger.info(f"Start: {start_time}, End: {end_time}")
            
            created_event = self.service.events().insert(
                calendarId='primary', 
                body=event,
                sendNotifications=True
            ).execute()
            
            event_link = created_event.get('htmlLink')
            logger.info(f"✅ Event created successfully: {event_link}")
            return event_link
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Error adding event to Google Calendar: {error_msg}")
            
            # Handle specific error cases
            if "invalid_scope" in error_msg.lower():
                st.error("❌ Calendar scope error. Please reset authorization.")
                if st.button("🔄 Reset Google Calendar Authorization", key="reset_auth"):
                    self.reset_credentials()
                    st.rerun()
            elif "forbidden" in error_msg.lower():
                st.error("❌ Access denied. Please check your Google Calendar permissions.")
            elif "not found" in error_msg.lower():
                st.error("❌ Calendar not found. Please check your Google Calendar setup.")
            
            return None

class BackendClient:
    """Enhanced backend client with retry logic and error handling"""
    def __init__(self):
        self.base_url = config.backend_url
        self.session = requests.Session()
        self.session.timeout = config.request_timeout
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """Make HTTP request with retry logic and detailed error logging"""
        url = f"{self.base_url}{endpoint}"
        
        # Log the request for debugging
        if method == "POST" and "json" in kwargs:
            logger.info(f"Making {method} request to {url}")
            logger.info(f"Request payload: {kwargs['json']}")
        
        for attempt in range(config.max_retries):
            try:
                response = self.session.request(method, url, **kwargs)
                
                # Log response details for debugging
                logger.info(f"Response status: {response.status_code}")
                if not response.ok:
                    logger.error(f"Response content: {response.text}")
                
                response.raise_for_status()
                return response
                
            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout (attempt {attempt + 1}/{config.max_retries})")
                if attempt == config.max_retries - 1:
                    raise
                    
            except requests.exceptions.ConnectionError:
                logger.warning(f"Connection error (attempt {attempt + 1}/{config.max_retries})")
                if attempt == config.max_retries - 1:
                    raise
                    
            except requests.exceptions.HTTPError as e:
                # For 422 errors, we want to see the detailed error message
                if e.response.status_code == 422:
                    try:
                        error_detail = e.response.json()
                        logger.error(f"Validation error (422): {error_detail}")
                        return e.response  # Return the response so we can handle the error gracefully
                    except:
                        logger.error(f"422 error with no JSON response: {e.response.text}")
                logger.error(f"HTTP error: {e}")
                raise
                
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                raise
    
    def schedule_task(self, task_input: str, user_timezone: str = None, priority: str = "Medium") -> Dict[str, Any]:
        """Schedule task with proper payload format"""
        try:
            # Ensure the payload matches exactly what FastAPI expects
            payload = {
                "input": task_input,
                "timezone": user_timezone or config.default_timezone,
                "priority": priority  # Add priority field
            }
            
            # Validate payload before sending
            if not task_input or not task_input.strip():
                return {"error": "Task input cannot be empty"}
            
            if len(task_input) > 500:
                return {"error": "Task input too long (max 500 characters)"}
            
            # Make sure timezone is in the supported list
            supported_timezones = [
                "Asia/Kolkata", "America/New_York", "America/Los_Angeles", 
                "Europe/London", "Europe/Paris", "Asia/Tokyo", "Australia/Sydney"
            ]
            if payload["timezone"] not in supported_timezones:
                payload["timezone"] = "Asia/Kolkata"  # Fallback to default
            
            logger.info(f"Sending schedule request with payload: {payload}")
            
            response = self._make_request("POST", "/schedule", json=payload)
            
            # Handle 422 responses gracefully
            if response.status_code == 422:
                try:
                    error_data = response.json()
                    error_msg = "Validation error: "
                    if "detail" in error_data:
                        if isinstance(error_data["detail"], list):
                            # Pydantic validation errors
                            for error in error_data["detail"]:
                                error_msg += f"{error.get('loc', [])} - {error.get('msg', 'Unknown error')}; "
                        else:
                            error_msg += str(error_data["detail"])
                    return {"error": error_msg}
                except:
                    return {"error": f"Server validation error: {response.text}"}
            
            return response.json()
            
        except Exception as e:
            logger.error(f"Error scheduling task: {e}")
            return {"error": str(e)}
    
    def transcribe_audio(self, audio_data: bytes, language: str = "en-US") -> Dict[str, Any]:
        """Transcribe audio with error handling"""
        try:
            files = {"file": ("audio.wav", audio_data, "audio/wav")}
            data = {"language": language}  # Send language as form data
            response = self._make_request("POST", "/speech-to-text", files=files, data=data)
            return response.json()
            
        except Exception as e:
            logger.error(f"Error transcribing audio: {e}")
            return {"error": str(e)}

# Initialize components
if "audio_processor" not in st.session_state:
    st.session_state.audio_processor = AudioProcessor()

if "calendar_manager" not in st.session_state:
    st.session_state.calendar_manager = GoogleCalendarManager()

if "backend_client" not in st.session_state:
    st.session_state.backend_client = BackendClient()

# UI Components
st.title("🎙️ AI Task Scheduler (Chat + Voice)")

# Timezone selection
st.sidebar.header("⚙️ Settings")
available_timezones = [
    "Asia/Kolkata", "America/New_York", "America/Los_Angeles", 
    "Europe/London", "Europe/Paris", "Asia/Tokyo", "Australia/Sydney"
]
selected_timezone = st.sidebar.selectbox(
    "Select your timezone:",
    available_timezones,
    index=0
)

# Google Calendar settings
st.sidebar.subheader("📅 Google Calendar")
if st.sidebar.button("🔄 Reset Google Calendar Auth", key="sidebar_reset"):
    if st.session_state.calendar_manager.reset_credentials():
        st.sidebar.success("✅ Authorization reset successfully!")
    else:
        st.sidebar.error("❌ Failed to reset authorization")

# Connection status
try:
    health_check = st.session_state.backend_client._make_request("GET", "/health")
    st.sidebar.success("✅ Backend Connected")
except:
    st.sidebar.error("❌ Backend Disconnected")

# Text input section
st.subheader("📌 Type your task here")
text_input = st.text_input("What do you want to schedule?", key="text_task")

if st.button("Submit Text Task", key="submit_text"):
    if text_input:
        with st.spinner("Processing task..."):
            try:
                # Schedule task
                response = st.session_state.backend_client.schedule_task(text_input, selected_timezone)
                
                if "error" in response:
                    st.error(f"Error: {response['error']}")
                else:
                    st.success("✅ Task scheduled successfully!")
                    
                    # Display AI response
                    if "response" in response:
                        try:
                            ai_response = json.loads(response["response"])
                            st.json(ai_response)
                            
                            # Add to Google Calendar
                            if "suggested_time" in ai_response:
                                start_time = ai_response["suggested_time"]
                                duration = ai_response.get("duration_minutes", 60)
                                
                                # Calculate end time properly
                                try:
                                    start_dt = datetime.datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                                    end_dt = start_dt + datetime.timedelta(minutes=duration)
                                    end_time = end_dt.isoformat()
                                except Exception as e:
                                    logger.warning(f"Date parsing issue: {e}")
                                    # Fallback
                                    end_time = start_time
                                
                                link = st.session_state.calendar_manager.add_event(
                                    summary=ai_response.get("task", text_input),
                                    start_time=start_time,
                                    end_time=end_time,
                                    description=ai_response.get("reason", "")
                                )
                                
                                if link:
                                    st.markdown(f"📅 [Event added to Google Calendar]({link})")
                                else:
                                    st.warning("⚠️ Could not add event to Google Calendar")
                                    
                        except json.JSONDecodeError:
                            st.info(response["response"])
                            
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please enter a task first.")

# Voice input section
st.markdown("## 🎤 Or record a voice command")

# WebRTC configuration
rtc_configuration = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

# Voice recording interface
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🎙️ Start Recording", key="start_recording"):
        st.session_state.audio_processor.start_recording()
        st.success("Recording started...")

with col2:
    if st.button("⏹️ Stop Recording", key="stop_recording"):
        try:
            st.session_state.audio_processor.stop_recording()
            st.success("Recording stopped.")
        except ValueError as e:
            st.error(f"Recording error: {e}")

with col3:
    if st.button("🗑️ Clear Recording", key="clear_recording"):
        st.session_state.audio_processor.clear()
        st.success("Recording cleared!")

# WebRTC streamer
webrtc_ctx = webrtc_streamer(
    key="speech-to-text",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=rtc_configuration,
    media_stream_constraints={
        "video": False,
        "audio": {
            "echoCancellation": True,
            "noiseSuppression": True,
            "autoGainControl": True,
        },
    },
    async_processing=True,
)

# Process audio frames
if webrtc_ctx.audio_receiver:
    try:
        audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
        for audio_frame in audio_frames:
            st.session_state.audio_processor.add_frame(audio_frame)
    except queue.Empty:
        pass

# Recording status
if st.session_state.audio_processor.recording_start_time:
    duration = (datetime.datetime.now() - st.session_state.audio_processor.recording_start_time).total_seconds()
    st.info(f"🎙️ Recording... {duration:.1f}s / {MAX_AUDIO_DURATION}s")
elif st.session_state.audio_processor.audio_frames:
    st.info("🔴 Recording ready for submission")
else:
    st.info("🎤 Click 'Start Recording' to begin")

# Submit voice task
if st.button("Submit Voice Task", key="submit_voice"):
    try:
        audio_data = st.session_state.audio_processor.get_audio_data()
        if not audio_data:
            st.warning("No audio recorded. Please record something first.")
        else:
            with st.spinner("Processing voice..."):
                # Transcribe audio
                transcription_result = st.session_state.backend_client.transcribe_audio(audio_data)
                
                if "error" in transcription_result:
                    st.error(f"Transcription error: {transcription_result['error']}")
                else:
                    transcription = transcription_result.get("text", "")
                    if not transcription:
                        st.error("No speech detected in audio")
                    else:
                        st.success(f"📝 Transcribed: {transcription}")
                        
                        # Schedule the transcribed task
                        response = st.session_state.backend_client.schedule_task(transcription, selected_timezone)
                        
                        if "error" in response:
                            st.error(f"Scheduling error: {response['error']}")
                        else:
                            st.success("✅ Voice task scheduled successfully!")
                            
                            # Display and process AI response
                            if "response" in response:
                                try:
                                    ai_response = json.loads(response["response"])
                                    st.json(ai_response)
                                    
                                    # Add to Google Calendar
                                    if "suggested_time" in ai_response:
                                        start_time = ai_response["suggested_time"]
                                        duration = ai_response.get("duration_minutes", 60)
                                        
                                        # Calculate end time properly
                                        try:
                                            start_dt = datetime.datetime.fromisoformat(start_time.replace('Z', '+00:00'))
                                            end_dt = start_dt + datetime.timedelta(minutes=duration)
                                            end_time = end_dt.isoformat()
                                        except Exception as e:
                                            logger.warning(f"Date parsing issue: {e}")
                                            # Fallback
                                            end_time = start_time
                                        
                                        link = st.session_state.calendar_manager.add_event(
                                            summary=ai_response.get("task", transcription),
                                            start_time=start_time,
                                            end_time=end_time,
                                            description=ai_response.get("reason", "")
                                        )
                                        
                                        if link:
                                            st.markdown(f"📅 [Event added to Google Calendar]({link})")
                                        else:
                                            st.warning("⚠️ Could not add event to Google Calendar")
                                            
                                except json.JSONDecodeError:
                                    st.info(response["response"])
                        
                        # Clear audio after successful processing
                        st.session_state.audio_processor.clear()
                        
    except Exception as e:
        st.error(f"Error processing voice: {e}")

# Calendar display
st.markdown("## 📅 Your Scheduled Tasks")
events = [
    {
        "title": "Team Meeting", 
        "start": "2025-07-20T10:00:00", 
        "end": "2025-07-20T11:00:00",
        "color": "#FF6B6B"
    },
    {
        "title": "Submit Project", 
        "start": "2025-07-21T15:00:00",
        "end": "2025-07-21T16:00:00",
        "color": "#4ECDC4"
    },
]

calendar_options = {
    "initialView": "dayGridMonth",
    "editable": False,
    "selectable": True,
    "headerToolbar": {
        "left": "prev,next today",
        "center": "title",
        "right": "dayGridMonth,timeGridWeek,timeGridDay"
    },
    "height": 650,
    "businessHours": {
        "daysOfWeek": [1, 2, 3, 4, 5],
        "startTime": "09:00",
        "endTime": "18:00"
    }
}

selected_event = calendar(events=events, options=calendar_options)

# Display debug info in sidebar
if st.sidebar.checkbox("Show Debug Info"):
    st.sidebar.write("Session State Keys:", list(st.session_state.keys()))
    st.sidebar.write("Config:", {
        "Backend URL": config.backend_url,
        "Timezone": selected_timezone,
        "Max Retries": config.max_retries,
        "Request Timeout": config.request_timeout
    })

# Help section
with st.sidebar.expander("ℹ️ Help & Troubleshooting"):
    st.write("""
    **Google Calendar Issues:**
    1. If you get "invalid_scope" errors, click "Reset Google Calendar Auth"
    2. Make sure your Google Cloud project has Calendar API enabled
    3. Check that your credentials.json file is valid
    
    **Task Scheduling:**
    - Be specific with your time preferences
    - Include dates and times for better accuracy
    - Use natural language like "tomorrow at 3 PM"
    
    **Voice Commands:**
    - Speak clearly and at normal volume
    - Ensure good microphone connection
    - Keep background noise to minimum
    """)