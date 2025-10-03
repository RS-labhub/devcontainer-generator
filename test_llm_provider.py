#!/usr/bin/env python3
"""
Test script to verify LLM provider setup
"""

import os
import logging
from dotenv import load_dotenv
from helpers.llm_provider import LLMProviderManager
from helpers.openai_helpers import setup_llm_client, setup_instructor, check_env_vars

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Load environment variables
load_dotenv()

def test_provider_setup():
    """Test the LLM provider setup"""
    print("=" * 60)
    print("Testing LLM Provider Setup")
    print("=" * 60)
    
    # Check environment variables
    print("\n1. Checking environment variables...")
    if not check_env_vars():
        print("Environment variables check failed!")
        return False
    print("Environment variables OK")
    
    # Get provider info
    provider = LLMProviderManager.get_provider()
    model = os.getenv("MODEL") or LLMProviderManager.get_default_model()
    print(f"\n2. Provider Configuration:")
    print(f"   Provider: {provider}")
    print(f"   Model: {model}")
    
    # Validate model
    is_valid, error_msg = LLMProviderManager.validate_model_for_provider(model, provider)
    if not is_valid:
        print(f"   Warning: {error_msg}")
    else:
        print(f"   Model compatible with provider")
    
    # Initialize client
    print(f"\n3. Initializing {provider} client...")
    try:
        llm_client = setup_llm_client()
        print(f"   Client initialized: {type(llm_client).__name__}")
    except Exception as e:
        print(f"   Failed to initialize client: {str(e)}")
        return False
    
    # Setup instructor
    print(f"\n4. Setting up Instructor wrapper...")
    try:
        instructor_client = setup_instructor(llm_client)
        print(f"   Instructor setup complete: {type(instructor_client).__name__}")
        print(f"   Available methods: {[m for m in dir(instructor_client) if not m.startswith('_')][:5]}...")
    except Exception as e:
        print(f"   Failed to setup instructor: {str(e)}")
        return False
    
    # Test simple completion (if client has the method)
    print(f"\n5. Testing API connection...")
    try:
        if hasattr(instructor_client, 'chat'):
            print(f"   Client has 'chat' attribute")
            if hasattr(instructor_client.chat, 'completions'):
                print(f"   Client has 'chat.completions' attribute")
                print(f"   Ready to make API calls")
            else:
                print(f"   Client missing 'chat.completions' attribute")
        else:
            print(f"   Client missing 'chat' attribute")
            print(f"   Available: {[m for m in dir(instructor_client) if not m.startswith('_')]}")
    except Exception as e:
        print(f"   Error checking client: {str(e)}")
    
    print("\n" + "=" * 60)
    print("Setup test completed!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    try:
        success = test_provider_setup()
        exit(0 if success else 1)
    except Exception as e:
        logging.error(f"Test failed with error: {str(e)}", exc_info=True)
        exit(1)
