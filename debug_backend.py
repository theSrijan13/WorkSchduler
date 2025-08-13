#!/usr/bin/env python3
"""
Debug script to test the FastAPI backend directly
Run this script to check if your backend is working correctly
"""

import requests
import json
import sys

def test_backend():
    """Test the backend endpoints"""
    base_url = "http://localhost:8000"
    
    print("🔍 Testing AI Task Scheduler Backend")
    print("=" * 50)
    
    # Test 1: Health check
    print("\n1️⃣ Testing health endpoint...")
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        print(f"   Status: {response.status_code}")
        if response.ok:
            health_data = response.json()
            print(f"   Response: {json.dumps(health_data, indent=2)}")
            print("   ✅ Health check passed")
        else:
            print(f"   ❌ Health check failed: {response.text}")
    except requests.exceptions.ConnectionError:
        print("   ❌ Cannot connect to backend. Is it running on localhost:8000?")
        return False
    except Exception as e:
        print(f"   ❌ Health check error: {e}")
        return False
    
    # Test 2: Schedule endpoint
    print("\n2️⃣ Testing schedule endpoint...")
    test_payloads = [
        {
            "input": "Schedule a coding session today between 11 am to 12 pm IST",
            "timezone": "Asia/Kolkata",
            "priority": "Medium"
        },
        {
            "input": "Team meeting tomorrow at 2 PM",
            "timezone": "Asia/Kolkata"
        },
        {
            "input": "Quick task",
            "priority": "High"
        }
    ]
    
    for i, payload in enumerate(test_payloads, 1):
        print(f"\n   Test {i}: {json.dumps(payload, indent=4)}")
        try:
            response = requests.post(
                f"{base_url}/schedule", 
                json=payload,
                timeout=30,
                headers={"Content-Type": "application/json"}
            )
            print(f"   Status: {response.status_code}")
            
            if response.ok:
                result = response.json()
                print(f"   ✅ Success: {json.dumps(result, indent=4)}")
            else:
                print(f"   ❌ Failed: {response.text}")
                try:
                    error_json = response.json()
                    print(f"   Error details: {json.dumps(error_json, indent=4)}")
                except:
                    pass
                    
        except Exception as e:
            print(f"   ❌ Request error: {e}")
    
    # Test 3: Check available endpoints
    print("\n3️⃣ Testing additional endpoints...")
    endpoints_to_test = ["/timezones", "/languages", "/cache-stats"]
    
    for endpoint in endpoints_to_test:
        try:
            response = requests.get(f"{base_url}{endpoint}", timeout=10)
            print(f"   {endpoint}: {response.status_code}")
            if response.ok:
                data = response.json()
                print(f"   Response: {json.dumps(data, indent=2)}")
        except Exception as e:
            print(f"   {endpoint}: Error - {e}")
    
    return True

def test_pydantic_validation():
    """Test the InputText model validation"""
    print("\n4️⃣ Testing Pydantic validation locally...")
    
    # Import the models from your backend (adjust import path as needed)
    try:
        from pydantic import BaseModel, Field, field_validator
        
        # Recreate the InputText model for testing
        class InputText(BaseModel):
            input: str = Field(..., min_length=1, max_length=500, description="Task description")
            timezone: Optional[str] = Field(default=None, description="User timezone")
            priority: Optional[str] = Field(default="Medium", pattern="^(High|Medium|Low)$")
            
            @field_validator('timezone')
            @classmethod
            def validate_timezone(cls, v):
                supported_timezones = [
                    "Asia/Kolkata", "America/New_York", "America/Los_Angeles", 
                    "Europe/London", "Europe/Paris", "Asia/Tokyo", "Australia/Sydney"
                ]
                if v and v not in supported_timezones:
                    raise ValueError(f"Unsupported timezone. Supported: {supported_timezones}")
                return v
        
        # Test different inputs
        test_cases = [
            {
                "input": "Schedule a coding session today between 11 am to 12 pm IST",
                "timezone": "Asia/Kolkata",
                "priority": "Medium"
            },
            {
                "input": "Quick task"
            },
            {
                "input": "",  # This should fail
                "timezone": "Invalid/Timezone"  # This should also fail
            }
        ]
        
        for i, case in enumerate(test_cases, 1):
            print(f"   Test case {i}: {case}")
            try:
                validated = InputText(**case)
                print(f"   ✅ Valid: {validated.model_dump()}")
            except Exception as e:
                print(f"   ❌ Invalid: {e}")
                
    except ImportError:
        print("   ⚠️ Cannot import Pydantic models for local testing")

if __name__ == "__main__":
    print("Starting backend debug session...")
    
    # Test backend connectivity and endpoints
    backend_ok = test_backend()
    
    if backend_ok:
        # Test Pydantic validation
        test_pydantic_validation()
    
    print("\n" + "=" * 50)
    print("🏁 Debug session complete")
    
    if not backend_ok:
        print("\n💡 Next steps:")
        print("1. Make sure your FastAPI backend is running: python main.py")
        print("2. Check if port 8000 is available")
        print("3. Verify all environment variables are set")
        print("4. Check the backend logs for any startup errors")