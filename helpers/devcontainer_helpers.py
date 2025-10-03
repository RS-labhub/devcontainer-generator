# helpers/devcontainer_helpers.py

import json
import logging
import os
import jsonschema
import tiktoken
from helpers.jinja_helper import process_template
from helpers.llm_provider import LLMProviderManager
from schemas import DevContainerModel
from supabase_client import supabase
from models import DevContainer


import logging
import tiktoken

def truncate_context(context, max_tokens=120000):
    logging.info(f"Starting truncate_context with max_tokens={max_tokens}")
    logging.debug(f"Initial context length: {len(context)} characters")

    encoding = tiktoken.encoding_for_model("gpt-4o-mini")
    tokens = encoding.encode(context)

    logging.info(f"Initial token count: {len(tokens)}")

    if len(tokens) <= max_tokens:
        logging.info("Context is already within token limit. No truncation needed.")
        return context

    logging.info(f"Context size is {len(tokens)} tokens. Truncation needed.")

    # Prioritize keeping the repository structure and languages
    structure_end = context.find("<<END_SECTION: Repository Structure >>")
    languages_end = context.find("<<END_SECTION: Repository Languages >>")

    logging.debug(f"Structure end position: {structure_end}")
    logging.debug(f"Languages end position: {languages_end}")

    important_content = context[:languages_end] + "<<END_SECTION: Repository Languages >>\n\n"
    remaining_content = context[languages_end + len("<<END_SECTION: Repository Languages >>\n\n"):]

    important_tokens = encoding.encode(important_content)
    logging.debug(f"Important content token count: {len(important_tokens)}")

    if len(important_tokens) > max_tokens:
        logging.warning("Important content alone exceeds max_tokens. Truncating important content.")
        important_content = encoding.decode(important_tokens[:max_tokens])
        return important_content

    remaining_tokens = max_tokens - len(important_tokens)
    logging.info(f"Tokens available for remaining content: {remaining_tokens}")

    truncated_remaining = encoding.decode(encoding.encode(remaining_content)[:remaining_tokens])

    final_context = important_content + truncated_remaining
    final_tokens = encoding.encode(final_context)

    logging.info(f"Final token count: {len(final_tokens)}")
    logging.debug(f"Final context length: {len(final_context)} characters")

    return final_context

def generate_devcontainer_json(instructor_client, repo_url, repo_context, devcontainer_url=None, max_retries=3, regenerate=False):
    existing_devcontainer = None
    if "<<EXISTING_DEVCONTAINER>>" in repo_context:
        logging.info("Existing devcontainer.json found in the repository.")
        existing_devcontainer = (
            repo_context.split("<<EXISTING_DEVCONTAINER>>")[1]
            .split("<<END_EXISTING_DEVCONTAINER>>")[0]
            .strip()
        )
        if not regenerate and devcontainer_url:
            logging.info(f"Using existing devcontainer.json from URL: {devcontainer_url}")
            return existing_devcontainer, devcontainer_url

    logging.info("Generating devcontainer.json...")

    # Truncate the context to fit within token limits
    truncated_context = truncate_context(repo_context, max_tokens=126000)

    template_data = {
        "repo_url": repo_url,
        "repo_context": truncated_context,
        "existing_devcontainer": existing_devcontainer
    }

    prompt = process_template("prompts/devcontainer.jinja", template_data)

    for attempt in range(max_retries + 1):
        try:
            logging.debug(f"Attempt {attempt + 1}/{max_retries + 1} to generate devcontainer.json")
            model = os.getenv("MODEL") or LLMProviderManager.get_default_model()
            provider = LLMProviderManager.get_provider()
            logging.debug(f"Using provider: {provider}, model: {model}")
            
            # Add better error handling and logging for debugging
            try:
                # Add temperature parameter for more consistent outputs
                # Lower temperature = more deterministic
                # Also add max_retries for instructor to handle JSON errors
                response = instructor_client.chat.completions.create(
                    model=model,
                    response_model=DevContainerModel,
                    temperature=0.1,  # Very low temperature for more consistent structured output
                    max_retries=2,  # Let instructor retry on parsing failures
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant that generates valid, complete devcontainer.json files. Ensure all JSON is properly closed with matching braces."},
                        {"role": "user", "content": prompt},
                    ],
                )
            except AttributeError as ae:
                logging.error(f"AttributeError when calling instructor_client: {str(ae)}")
                logging.error(f"Instructor client type: {type(instructor_client)}")
                logging.error(f"Available attributes: {dir(instructor_client)}")
                raise
            except Exception as api_error:
                error_msg = str(api_error)
                logging.error(f"API call failed: {error_msg}")
                
                # Check for specific error types
                if "tool_use_failed" in error_msg.lower() or "failed to call a function" in error_msg.lower():
                    logging.warning(f"Function calling failed (attempt {attempt + 1}/{max_retries + 1}). This is common with Groq. Retrying...")
                    if attempt < max_retries:
                        continue  # Retry with next attempt
                    else:
                        raise ValueError(
                            "The LLM provider failed to generate valid structured output after multiple attempts. "
                            "This is a known limitation with some providers (especially Groq with complex schemas). "
                            "Recommendation: Switch to OpenAI or Anthropic for better reliability. "
                            f"Run: ./switch_to_openai.sh"
                        )
                elif "does not support" in error_msg.lower() or "temperature" in error_msg.lower():
                    logging.warning("Temperature parameter not supported, retrying without it...")
                    # Retry without temperature if not supported
                    response = instructor_client.chat.completions.create(
                        model=model,
                        response_model=DevContainerModel,
                        max_retries=2,
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant that generates valid, complete devcontainer.json files. Ensure all JSON is properly closed with matching braces."},
                            {"role": "user", "content": prompt},
                        ],
                    )
                else:
                    raise
            
            devcontainer_json = json.dumps(response.dict(exclude_none=True), indent=2)
            
            # Log the generated JSON for debugging
            logging.debug(f"Generated devcontainer.json (first 500 chars): {devcontainer_json[:500]}")

            if validate_devcontainer_json(devcontainer_json):
                logging.info("Successfully generated and validated devcontainer.json")
                if existing_devcontainer and not regenerate:
                    return existing_devcontainer, devcontainer_url
                else:
                    return devcontainer_json, None  # Return None as URL for generated content
            else:
                logging.warning(f"Generated JSON failed validation on attempt {attempt + 1}")
                logging.warning(f"Will retry with a new attempt..." if attempt < max_retries else "No more retries left.")
                if attempt == max_retries:
                    # On final failure, log the full JSON
                    logging.error(f"Final failed JSON: {devcontainer_json}")
                    raise ValueError(
                        "Failed to generate valid devcontainer.json after maximum retries. "
                        "Check logs for validation errors. This may be due to the LLM provider's "
                        "capabilities with structured outputs."
                    )
        except Exception as e:
            logging.error(f"Error on attempt {attempt + 1}: {str(e)}", exc_info=True)
            if attempt == max_retries:
                raise

    raise ValueError("Failed to generate valid devcontainer.json after maximum retries")

def validate_devcontainer_json(devcontainer_json):
    logging.info("Validating devcontainer.json...")
    schema_path = os.path.join(os.path.dirname(__file__), "..", "schemas", "devContainer.base.schema.json")
    with open(schema_path, "r") as schema_file:
        schema = json.load(schema_file)
    try:
        logging.debug("Running validation...")
        parsed_json = json.loads(devcontainer_json)
        jsonschema.validate(instance=parsed_json, schema=schema)
        logging.info("Validation successful.")
        return True
    except json.JSONDecodeError as e:
        logging.error(f"JSON parsing failed: {e}")
        logging.error(f"Invalid JSON content: {devcontainer_json[:500]}")
        return False
    except jsonschema.exceptions.ValidationError as e:
        logging.error(f"Schema validation failed: {e.message}")
        logging.error(f"Failed at path: {list(e.path)}")
        logging.error(f"Schema path: {list(e.schema_path)}")
        logging.error(f"Validator: {e.validator}")
        
        # Log the generated JSON for debugging
        try:
            parsed = json.loads(devcontainer_json)
            logging.error(f"Generated JSON keys: {list(parsed.keys())}")
            if 'image' in parsed:
                logging.error(f"Image value: {parsed['image']}")
        except:
            pass
        
        return False

def save_devcontainer(new_devcontainer):
    try:
        result = supabase.table("devcontainers").insert(new_devcontainer.dict()).execute()
        return result.data[0] if result.data else None
    except Exception as e:
        logging.error(f"Error saving devcontainer to Supabase: {str(e)}")
        raise