import os
import logging
import instructor
from helpers.llm_provider import LLMProviderManager

def setup_llm_client():
    """
    Setup the LLM client based on the configured provider
    Wrapper function for backward compatibility
    """
    return LLMProviderManager.setup_client()

def setup_azure_openai():
    """
    Setup Azure OpenAI client
    Deprecated: Use setup_llm_client() instead
    """
    logging.warning("setup_azure_openai() is deprecated. Use setup_llm_client() instead.")
    return LLMProviderManager.setup_azure_openai()

def setup_instructor(llm_client):
    """
    Setup Instructor client with the given LLM client
    Works with OpenAI-compatible APIs (Azure OpenAI, OpenAI, Groq)
    
    For Anthropic and Google, instructor can work with mode=instructor.Mode.ANTHROPIC_JSON
    or instructor.Mode.GEMINI_JSON respectively
    """
    logging.info("Setting up Instructor client...")
    provider = LLMProviderManager.get_provider()
    
    try:
        if provider in ["AzureOpenAI", "OpenAI"]:
            # OpenAI and Azure OpenAI work with default instructor patching
            logging.info("Using default instructor patching for OpenAI-compatible API")
            return instructor.patch(llm_client)
        elif provider == "Groq":
            # Groq is OpenAI-compatible, use default patching
            # Note: Groq might have limitations with complex structured outputs
            logging.info("Using default instructor patching for Groq (OpenAI-compatible)")
            try:
                # Try with tool calling mode first (default)
                return instructor.patch(llm_client, mode=instructor.Mode.TOOLS)
            except Exception as e:
                logging.warning(f"TOOLS mode failed for Groq, trying JSON mode: {str(e)}")
                # Fallback to JSON mode if TOOLS doesn't work
                return instructor.patch(llm_client, mode=instructor.Mode.JSON)
        elif provider == "Anthropic":
            # Anthropic requires special mode
            logging.info("Using Anthropic-specific instructor setup")
            return instructor.from_anthropic(llm_client)
        elif provider == "Google":
            # Google Generative AI requires special mode
            logging.info("Using Google Gemini-specific instructor setup")
            return instructor.from_gemini(
                client=llm_client,
                mode=instructor.Mode.GEMINI_JSON
            )
        else:
            logging.warning(f"Unknown provider {provider}, attempting default instructor patching")
            return instructor.patch(llm_client)
    except Exception as e:
        logging.error(f"Failed to setup instructor client: {str(e)}", exc_info=True)
        raise

def check_env_vars():
    """
    Check if all required environment variables are set based on the selected LLM provider
    """
    required_vars = LLMProviderManager.get_required_env_vars()
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        print(
            f"Missing environment variables: {', '.join(missing_vars)}. "
            "Please configure the env vars file properly."
        )
        return False
    return True