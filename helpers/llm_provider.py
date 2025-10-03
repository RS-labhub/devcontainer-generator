"""
LLM Provider Manager
Handles initialization of different LLM providers (Azure OpenAI, OpenAI, Anthropic, Google, Groq)
"""

import os
import logging
from typing import Any, Optional


class LLMProviderManager:
    """Manages initialization and configuration of different LLM providers"""
    
    SUPPORTED_PROVIDERS = ["AzureOpenAI", "OpenAI", "Anthropic", "Google", "Groq"]
    
    @staticmethod
    def get_provider() -> str:
        """Get the configured LLM provider from environment"""
        provider = os.getenv("LLM_PROVIDER", "AzureOpenAI")
        if provider not in LLMProviderManager.SUPPORTED_PROVIDERS:
            logging.warning(f"Unsupported provider '{provider}'. Defaulting to AzureOpenAI")
            return "AzureOpenAI"
        return provider
    
    @staticmethod
    def setup_azure_openai() -> Any:
        """Initialize Azure OpenAI client"""
        try:
            from openai import AzureOpenAI
            
            logging.info("Setting up Azure OpenAI client...")
            client = AzureOpenAI(
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            )
            logging.info("Azure OpenAI client initialized successfully")
            return client
        except Exception as e:
            logging.error(f"Failed to initialize Azure OpenAI client: {str(e)}")
            raise
    
    @staticmethod
    def setup_openai() -> Any:
        """Initialize OpenAI client"""
        try:
            from openai import OpenAI
            
            logging.info("Setting up OpenAI client...")
            client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
            )
            logging.info("OpenAI client initialized successfully")
            return client
        except Exception as e:
            logging.error(f"Failed to initialize OpenAI client: {str(e)}")
            raise
    
    @staticmethod
    def setup_anthropic() -> Any:
        """Initialize Anthropic client"""
        try:
            from anthropic import Anthropic
            
            logging.info("Setting up Anthropic client...")
            client = Anthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY"),
            )
            logging.info("Anthropic client initialized successfully")
            return client
        except Exception as e:
            logging.error(f"Failed to initialize Anthropic client: {str(e)}")
            raise
    
    @staticmethod
    def setup_google() -> Any:
        """Initialize Google Generative AI client"""
        try:
            import google.generativeai as genai
            
            logging.info("Setting up Google Generative AI client...")
            genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
            # Get the model name
            model = os.getenv("MODEL") or "gemini-1.5-pro"
            
            # Create a GenerativeModel instance (required by instructor)
            client = genai.GenerativeModel(model)
            logging.info(f"Google Generative AI client initialized successfully with model: {model}")
            return client
        except Exception as e:
            logging.error(f"Failed to initialize Google Generative AI client: {str(e)}")
            raise
    
    @staticmethod
    def setup_groq() -> Any:
        """Initialize Groq client"""
        try:
            from groq import Groq
            
            logging.info("Setting up Groq client...")
            client = Groq(
                api_key=os.getenv("GROQ_API_KEY"),
            )
            logging.info("Groq client initialized successfully")
            return client
        except Exception as e:
            logging.error(f"Failed to initialize Groq client: {str(e)}")
            raise
    
    @staticmethod
    def setup_client() -> Any:
        """
        Setup the appropriate LLM client based on the configured provider
        Returns the initialized client
        """
        provider = LLMProviderManager.get_provider()
        logging.info(f"Initializing LLM provider: {provider}")
        
        if provider == "AzureOpenAI":
            return LLMProviderManager.setup_azure_openai()
        elif provider == "OpenAI":
            return LLMProviderManager.setup_openai()
        elif provider == "Anthropic":
            return LLMProviderManager.setup_anthropic()
        elif provider == "Google":
            return LLMProviderManager.setup_google()
        elif provider == "Groq":
            return LLMProviderManager.setup_groq()
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    @staticmethod
    def get_required_env_vars() -> list:
        """
        Get the list of required environment variables based on the selected provider
        """
        provider = LLMProviderManager.get_provider()
        
        # Common required variables
        common_vars = ["MODEL", "GITHUB_TOKEN", "SUPABASE_URL", "SUPABASE_KEY", "LLM_PROVIDER"]
        
        provider_vars = {
            "AzureOpenAI": ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_API_VERSION"],
            "OpenAI": ["OPENAI_API_KEY"],
            "Anthropic": ["ANTHROPIC_API_KEY"],
            "Google": ["GOOGLE_API_KEY"],
            "Groq": ["GROQ_API_KEY"],
        }
        
        return common_vars + provider_vars.get(provider, [])
    
    @staticmethod
    def supports_embeddings() -> bool:
        """
        Check if the current provider supports embeddings
        Only OpenAI and Azure OpenAI support embeddings currently
        """
        provider = LLMProviderManager.get_provider()
        return provider in ["AzureOpenAI", "OpenAI"]
    
    @staticmethod
    def get_default_model() -> str:
        """
        Get the default model for the current provider
        If MODEL is not set or incompatible with provider, returns the default
        """
        provider = LLMProviderManager.get_provider()
        model = os.getenv("MODEL")
        
        # Default models per provider
        defaults = {
            "AzureOpenAI": "gpt-4o-mini",
            "OpenAI": "gpt-4o-mini",
            "Anthropic": "claude-3-5-sonnet-20241022",
            "Google": "gemini-1.5-pro",
            "Groq": "llama-3.3-70b-versatile",
        }
        
        default_model = defaults.get(provider, "gpt-4o-mini")
        
        # If no model is set, return the default
        if not model:
            logging.warning(f"No MODEL specified in .env, using default for {provider}: {default_model}")
            return default_model
        
        # Check if the model is compatible with the provider
        is_valid, error_msg = LLMProviderManager.validate_model_for_provider(model, provider)
        if not is_valid:
            logging.warning(f"MODEL '{model}' may not be compatible with provider '{provider}'")
            logging.warning(f"Auto-selecting default model: {default_model}")
            logging.warning(f"To fix: Update MODEL in .env to a compatible model")
            return default_model
        
        return model
    
    @staticmethod
    def validate_model_for_provider(model: str, provider: str) -> tuple[bool, str]:
        """
        Validate if a model is compatible with the given provider
        Returns (is_valid, error_message)
        """
        # Valid models per provider (non-exhaustive, but common ones)
        valid_models = {
            "AzureOpenAI": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-35-turbo", "gpt-3.5-turbo"],
            "OpenAI": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4", "gpt-3.5-turbo"],
            "Anthropic": ["claude-3-5-sonnet-20241022", "claude-3-opus-20240229", "claude-3-sonnet-20240229", "claude-3-haiku-20240307"],
            "Google": ["gemini-2.5-pro", "gemini-2.0-flash", "gemini-1.5-pro", "gemini-1.5-flash", "gemini-pro", "gemini-1.0-pro"],
            "Groq": ["llama-3.3-70b-versatile", "llama-3.1-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768", "gemma2-9b-it"],
        }
        
        provider_models = valid_models.get(provider, [])
        
        # Check if model matches any known model for this provider
        if not provider_models:
            return True, ""  # Unknown provider, skip validation
        
        # Flexible matching - check if model name contains or is contained in valid models
        for valid_model in provider_models:
            if valid_model in model or model in valid_model:
                return True, ""
        
        return False, (
            f"Model '{model}' may not be compatible with provider '{provider}'. "
            f"Common models for {provider}: {', '.join(provider_models[:3])}"
        )
