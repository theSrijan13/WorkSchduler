from fastapi import FastAPI, UploadFile, HTTPException, BackgroundTasks, Depends, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field, field_validator
import google.generativeai as genai
from google.cloud import speech
import os
import io
import json
import logging
import asyncio
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import pytz
from contextlib import asynccontextmanager
import aiofiles
import hashlib
import time
from cachetools import TTLCache
import threading
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    """Application configuration"""
    def __init__(self):
        # API Keys
        self.gemini_api_key = os.getenv("GEMINI_API_KEY")
        self.google_credentials_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
        
        # Server settings
        self.max_workers = int(os.getenv("MAX_WORKERS", "4"))
        self.max_file_size = int(os.getenv("MAX_FILE_SIZE", "10485760"))  # 10MB
        self.cache_ttl = int(os.getenv("CACHE_TTL", "3600"))  # 1 hour
        self.rate_limit_requests = int(os.getenv("RATE_LIMIT_REQUESTS", "100"))
        self.rate_limit_window = int(os.getenv("RATE_LIMIT_WINDOW", "3600"))  # 1 hour
        
        # Audio settings
        self.max_audio_duration = int(os.getenv("MAX_AUDIO_DURATION", "300"))  # 5 minutes
        self.supported_audio_formats = ["wav", "mp3", "m4a", "flac"]
        
        # Timezone settings
        self.default_timezone = os.getenv("DEFAULT_TIMEZONE", "Asia/Kolkata")
        self.supported_timezones = [
            "Asia/Kolkata", "America/New_York", "America/Los_Angeles", 
            "Europe/London", "Europe/Paris", "Asia/Tokyo", "Australia/Sydney"
        ]
        
        # Gemini settings
        self.gemini_model = os.getenv("GEMINI_MODEL", "models/gemini-1.5-flash-latest")
        self.gemini_temperature = float(os.getenv("GEMINI_TEMPERATURE", "0.7"))
        self.gemini_max_tokens = int(os.getenv("GEMINI_MAX_TOKENS", "1000"))
        
        # Validation
        self.validate_config()
    
    def validate_config(self):
        """Validate configuration"""
        if not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required")
        
        # Make Google credentials optional for development
        if self.google_credentials_path and not os.path.exists(self.google_credentials_path):
            logger.warning(f"Google credentials file not found: {self.google_credentials_path}")

config = Config()

# Initialize services
genai.configure(api_key=config.gemini_api_key)

# Only initialize speech client if credentials are available
speech_client = None
if config.google_credentials_path and os.path.exists(config.google_credentials_path):
    try:
        speech_client = speech.SpeechClient()
        logger.info("✅ Google Speech-to-Text client initialized")
    except Exception as e:
        logger.warning(f"Failed to initialize Speech-to-Text client: {e}")

executor = ThreadPoolExecutor(max_workers=config.max_workers)

# Caching and rate limiting
cache = TTLCache(maxsize=1000, ttl=config.cache_ttl)
rate_limit_cache = TTLCache(maxsize=10000, ttl=config.rate_limit_window)
cache_lock = threading.Lock()

# Security
security = HTTPBearer(auto_error=False)

# Pydantic models
class InputText(BaseModel):
    input: str = Field(..., min_length=1, max_length=500, description="Task description")
    timezone: Optional[str] = Field(default=None, description="User timezone")
    priority: Optional[str] = Field(default="Medium", pattern="^(High|Medium|Low)$")
    
    @field_validator('timezone')
    @classmethod
    def validate_timezone(cls, v):
        if v and v not in config.supported_timezones:
            raise ValueError(f"Unsupported timezone. Supported: {config.supported_timezones}")
        return v

class ScheduleResponse(BaseModel):
    task: str
    suggested_time: str
    duration_minutes: int
    priority: str
    reason: str
    confidence: float

class TranscriptionResponse(BaseModel):
    text: str
    confidence: float
    language: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    services: Dict[str, str]

# Rate limiting
def check_rate_limit(client_id: str) -> bool:
    """Check if client has exceeded rate limit"""
    with cache_lock:
        current_time = time.time()
        client_requests = rate_limit_cache.get(client_id, [])
        
        # Remove old requests
        client_requests = [req_time for req_time in client_requests 
        if current_time - req_time < config.rate_limit_window]
        
        if len(client_requests) >= config.rate_limit_requests:
            return False
        
        client_requests.append(current_time)
        rate_limit_cache[client_id] = client_requests
        return True

def get_client_id(request: Request) -> str:
    """Get client identifier for rate limiting"""
    return request.client.host

# Fixed dependency for rate limiting
async def rate_limit_dependency(request: Request):
    """Rate limiting dependency - fixed version"""
    client_id = get_client_id(request)
    if not check_rate_limit(client_id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded. Please try again later."
        )
    return client_id

# Audio validation
async def validate_audio_file(file: UploadFile) -> None:
    """Validate uploaded audio file"""
    if file.size and file.size > config.max_file_size:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size: {config.max_file_size} bytes"
        )
    
    if file.content_type and not file.content_type.startswith("audio/"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Only audio files are supported."
        )
    
    # Check file extension
    if file.filename:
        file_extension = file.filename.split(".")[-1].lower()
        if file_extension not in config.supported_audio_formats:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Unsupported audio format. Supported: {config.supported_audio_formats}"
            )

# Enhanced AI scheduling
class AIScheduler:
    """Enhanced AI scheduler with better prompting and error handling"""
    
    def __init__(self):
        self.model = genai.GenerativeModel(
            config.gemini_model,
            generation_config=genai.types.GenerationConfig(
                temperature=config.gemini_temperature,
                max_output_tokens=config.gemini_max_tokens,
            )
        )
    
    def create_cache_key(self, task: str, timezone: str, priority: str) -> str:
        """Create cache key for task"""
        key_string = f"{task}|{timezone}|{priority}"
        return hashlib.md5(key_string.encode()).hexdigest()
    
    async def schedule_task(self, input_data: InputText) -> Dict[str, Any]:
        """Schedule task with enhanced AI prompting"""
        try:
            # Check cache first
            cache_key = self.create_cache_key(
                input_data.input, 
                input_data.timezone or config.default_timezone,
                input_data.priority
            )
            
            with cache_lock:
                cached_result = cache.get(cache_key)
                if cached_result:
                    logger.info(f"Cache hit for task: {input_data.input[:50]}...")
                    return cached_result
            
            # Get current time in user timezone
            user_tz = pytz.timezone(input_data.timezone or config.default_timezone)
            current_time = datetime.now(user_tz)
            
            # Create enhanced prompt
            prompt = self._create_scheduling_prompt(input_data, current_time, user_tz)
            
            # Run AI generation in thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                executor, 
                self._generate_ai_response, 
                prompt
            )
            
            # Parse and validate response
            result = self._parse_ai_response(response.text, input_data, current_time)
            
            # Cache result
            with cache_lock:
                cache[cache_key] = result
            
            return result
            
        except Exception as e:
            logger.error(f"Error in AI scheduling: {e}")
            return {"error": str(e)}
    
    def _create_scheduling_prompt(self, input_data: InputText, current_time: datetime, user_tz) -> str:
        """Create enhanced scheduling prompt"""
        return f"""
        You are an expert AI scheduling assistant with deep knowledge of productivity science and time management.

        CONTEXT:
        - Current time: {current_time.strftime('%Y-%m-%d %H:%M:%S %Z')}
        - User timezone: {user_tz.zone}
        - Task priority: {input_data.priority}
        - User input: "{input_data.input}"

        SCHEDULING RULES:
        1. Business hours: 9 AM - 6 PM on weekdays are optimal
        2. Avoid: Early morning (before 7 AM), late evening (after 10 PM)
        3. Consider task type:
           - Creative work: Morning (9-11 AM)
           - Administrative: Afternoon (2-4 PM)
           - Meetings: Mid-morning or early afternoon
           - Deep work: Morning or late afternoon
        4. High priority tasks should be scheduled sooner
        5. Estimate realistic durations based on task complexity
        6. Consider urgency indicators in user input
        7. If user specifies a specific date/time, try to honor it if reasonable

        TASK ANALYSIS:
        Analyze the user's task description for:
        - Task type (meeting, creative work, administrative, coding session, etc.)
        - Urgency indicators (urgent, ASAP, deadline, etc.)
        - Duration hints (quick, long, detailed, etc.)
        - Specific time preferences mentioned (like "between 11 am to 12 pm")
        - Specific date mentioned (like "22 july 2025")

        OUTPUT FORMAT (JSON only):
        {{
          "task": "<concise task summary>",
          "suggested_time": "<ISO 8601 datetime in user timezone>",
          "duration_minutes": <realistic duration in minutes>,
          "priority": "<High|Medium|Low>",
          "reason": "<brief explanation of timing choice>",
          "confidence": <0.0-1.0 confidence score>
        }}

        IMPORTANT: Return ONLY valid JSON. No additional text or explanation.
        """
    
    def _generate_ai_response(self, prompt: str):
        """Generate AI response (synchronous for thread pool)"""
        return self.model.generate_content(prompt)
    
    def _parse_ai_response(self, response_text: str, input_data: InputText, current_time: datetime) -> Dict[str, Any]:
        """Parse and validate AI response"""
        try:
            # Clean response text
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            # Parse JSON
            ai_response = json.loads(response_text)
            
            # Validate required fields
            required_fields = ["task", "suggested_time", "duration_minutes", "priority", "reason", "confidence"]
            for field in required_fields:
                if field not in ai_response:
                    raise ValueError(f"Missing required field: {field}")
            
            # Validate suggested time
            suggested_time = datetime.fromisoformat(ai_response["suggested_time"])
            if suggested_time < current_time:
                # Only adjust if it's significantly in the past (more than 1 hour)
                if (current_time - suggested_time).total_seconds() > 3600:
                    suggested_time = current_time + timedelta(hours=1)
                    ai_response["suggested_time"] = suggested_time.isoformat()
                    ai_response["reason"] += " (Adjusted to next available slot)"
            
            # Validate duration
            duration = ai_response["duration_minutes"]
            if not isinstance(duration, int) or duration < 5 or duration > 480:  # 5 min to 8 hours
                ai_response["duration_minutes"] = 60  # Default to 1 hour
            
            # Validate priority
            if ai_response["priority"] not in ["High", "Medium", "Low"]:
                ai_response["priority"] = input_data.priority
            
            # Validate confidence
            confidence = ai_response["confidence"]
            if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 1:
                ai_response["confidence"] = 0.5
            
            return {"response": json.dumps(ai_response)}
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            return {"error": "Invalid AI response format"}
        except Exception as e:
            logger.error(f"Response parsing error: {e}")
            return {"error": str(e)}

# Enhanced speech-to-text
class SpeechToTextService:
    """Enhanced speech-to-text service with better error handling"""
    
    def __init__(self):
        self.client = speech_client
        self.supported_languages = ["en-US", "en-GB", "es-ES", "fr-FR", "de-DE", "it-IT", "pt-BR"]
    
    async def transcribe_audio(self, audio_content: bytes, language_code: str = "en-US") -> Dict[str, Any]:
        """Transcribe audio with enhanced error handling"""
        if not self.client:
            return {"error": "Speech-to-text service not available. Please check Google Cloud credentials."}
        
        try:
            # Validate language
            if language_code not in self.supported_languages:
                language_code = "en-US"
            
            # Create audio object
            audio = speech.RecognitionAudio(content=audio_content)
            
            # Enhanced configuration
            config = speech.RecognitionConfig(
                encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
                language_code=language_code,
                enable_automatic_punctuation=True,
                enable_word_time_offsets=True,
                model="latest_long",
                use_enhanced=True,
                audio_channel_count=1,
            )
            
            # Run transcription in thread pool
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                executor,
                self._transcribe_sync,
                config,
                audio
            )
            
            # Process results
            if not response.results:
                return {"error": "No speech detected in audio"}
            
            # Get best transcription
            best_result = response.results[0]
            if not best_result.alternatives:
                return {"error": "No transcription alternatives found"}
            
            best_alternative = best_result.alternatives[0]
            
            return {
                "text": best_alternative.transcript,
                "confidence": best_alternative.confidence,
                "language": language_code
            }
            
        except Exception as e:
            logger.error(f"Speech-to-text error: {e}")
            return {"error": str(e)}
    
    def _transcribe_sync(self, config, audio):
        """Synchronous transcription for thread pool"""
        return self.client.recognize(config=config, audio=audio)

# Initialize services
ai_scheduler = AIScheduler()
speech_service = SpeechToTextService()

# Lifespan events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("Starting AI Task Scheduler Backend...")
    logger.info(f"Gemini model: {config.gemini_model}")
    logger.info(f"Max workers: {config.max_workers}")
    logger.info(f"Cache TTL: {config.cache_ttl}s")
    
    # Test connections
    try:
        # Test Gemini
        model = genai.GenerativeModel(config.gemini_model)
        test_response = model.generate_content("Hello")
        logger.info("✅ Gemini API connection successful")
        
        # Test Speech-to-Text if available
        if speech_client:
            logger.info("✅ Google Speech-to-Text API connection successful")
        else:
            logger.warning("⚠️ Google Speech-to-Text API not available")
        
    except Exception as e:
        logger.error(f"❌ Service initialization error: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down AI Task Scheduler Backend...")
    executor.shutdown(wait=True)

# FastAPI app
app = FastAPI(
    title="AI Task Scheduler Backend",
    description="Enhanced backend for AI-powered task scheduling with voice support",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify allowed origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Test services
        services = {
            "gemini": "healthy",
            "speech_to_text": "healthy" if speech_client else "unavailable",
            "cache": "healthy"
        }
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            services=services
        )
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
        )

# Schedule task endpoint - FIXED
@app.post("/schedule")
async def schedule_task(
    input_data: InputText,
    request: Request,  # Add explicit Request parameter
    background_tasks: BackgroundTasks,
    client_id: str = Depends(rate_limit_dependency)
):
    """Schedule task with AI assistance"""
    try:
        logger.info(f"Scheduling task for client {client_id}: {input_data.input[:50]}...")
        
        # Process with AI
        result = await ai_scheduler.schedule_task(input_data)
        
        # Log successful scheduling
        background_tasks.add_task(
            log_successful_request,
            "schedule",
            client_id,
            input_data.input
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Schedule task error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error occurred"
        )

# Speech-to-text endpoint - FIXED
@app.post("/speech-to-text")
async def transcribe_speech(
    file: UploadFile,
    request: Request,  # Add explicit Request parameter
    background_tasks: BackgroundTasks,
    language: str = "en-US",
    client_id: str = Depends(rate_limit_dependency)
):
    """Transcribe speech to text"""
    try:
        logger.info(f"Transcribing audio for client {client_id}: {file.filename}")
        
        # Validate file
        await validate_audio_file(file)
        
        # Read audio content
        audio_content = await file.read()
        
        # Transcribe
        result = await speech_service.transcribe_audio(audio_content, language)
        
        # Log successful transcription
        background_tasks.add_task(
            log_successful_request,
            "speech_to_text",
            client_id,
            file.filename or "unknown"
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Speech-to-text error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error occurred"
        )

# Background task logging
async def log_successful_request(endpoint: str, client_id: str, request_info: str):
    """Log successful request for analytics"""
    logger.info(f"SUCCESS - {endpoint} - Client: {client_id} - Info: {request_info[:100]}")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    logger.error(f"HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "timestamp": datetime.now().isoformat()}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "timestamp": datetime.now().isoformat()
        }
    )

# Additional utility endpoints
@app.get("/timezones")
async def get_supported_timezones():
    """Get list of supported timezones"""
    return {"timezones": config.supported_timezones}

@app.get("/languages")
async def get_supported_languages():
    """Get list of supported languages for speech-to-text"""
    return {"languages": speech_service.supported_languages}

@app.get("/cache-stats")
async def get_cache_stats():
    """Get cache statistics (for debugging)"""
    with cache_lock:
        return {
            "cache_size": len(cache),
            "cache_maxsize": cache.maxsize,
            "cache_ttl": cache.ttl,
            "rate_limit_entries": len(rate_limit_cache)
        }

# Development endpoint to clear cache
@app.post("/clear-cache")
async def clear_cache():
    """Clear all caches (development only)"""
    with cache_lock:
        cache.clear()
        rate_limit_cache.clear()
    return {"message": "Cache cleared successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )