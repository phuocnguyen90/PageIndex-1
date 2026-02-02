import openai
import logging
import os
import asyncio
import random
import time
from dotenv import load_dotenv

from .cost_tracker import get_global_tracker
from .token_utils import count_tokens

load_dotenv()
logger = logging.getLogger(__name__)

# Performance / Concurrency Settings
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "5"))

# Global semaphores dictionary bound to loops to avoid cross-loop errors in Streamlit
_loop_semaphores = {}

def get_semaphore():
    try:
        loop = asyncio.get_running_loop()
        loop_id = id(loop)
        if loop_id not in _loop_semaphores:
            _loop_semaphores[loop_id] = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        return _loop_semaphores[loop_id]
    except RuntimeError:
        # No running loop
        return None

# API Keys
CHATGPT_API_KEY = os.getenv("CHATGPT_API_KEY") or os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL")


def _get_model_from_env(model_var=None):
    """Get model name from environment variable"""
    if model_var:
        return os.getenv(model_var)
    return None


def resolve_model_name(model, api_provider=None, model_name=None, env_model_var=None):
    """
    Resolve the actual model name to use based on provider configuration.

    Args:
        model: Default model from config
        api_provider: API provider (openai, openrouter, etc.)
        model_name: Provider-specific model name (optional override)
        env_model_var: Environment variable name containing model

    Returns:
        str: The model name to use for API calls
    """
    # First, check for environment variable model
    if env_model_var:
        env_model = _get_model_from_env(env_model_var)
        if env_model:
            return env_model

    # Then check for explicit model name override
    if model_name:
        return model_name

    # If using OpenRouter and no model name specified, apply prefixes
    if api_provider == "openrouter":
        # For OpenRouter, we need to prepend the provider prefix for common models
        # Common OpenRouter models:
        # - openai/gpt-4o
        # - anthropic/claude-3-5-sonnet-20241022
        # - meta-llama/llama-3.1-70b-instruct
        if model.startswith("gpt-"):
            return f"openai/{model}"
        elif model.startswith("claude-"):
            return f"anthropic/{model}"
        elif model.startswith("llama-"):
            return f"meta-llama/{model}"
        else:
            # If no known prefix, assume it's already in the correct format for OpenRouter
            return model

    # Default OpenAI or other providers
    return model


def _create_client(api_key=None, base_url=None):
    """Create a client for OpenAI or OpenRouter based on configuration"""
    # Determine which API key to use and base URL
    final_api_key = None
    final_base_url = None

    if base_url:
        # Custom endpoint (could be OpenRouter or other compatible API)
        if base_url == "https://openrouter.ai/api/v1" or "openrouter" in base_url:
            # OpenRouter base URL
            final_api_key = api_key or OPENROUTER_API_KEY
            final_base_url = "https://openrouter.ai/api/v1"
        else:
            # Custom endpoint
            final_api_key = api_key or CHATGPT_API_KEY
            final_base_url = base_url
    elif api_key and api_key == OPENROUTER_API_KEY:
        # Explicit OpenRouter API key
        final_api_key = OPENROUTER_API_KEY
        final_base_url = "https://openrouter.ai/api/v1"
    elif OPENROUTER_API_KEY:
        # OpenRouter API key is available
        final_api_key = OPENROUTER_API_KEY
        final_base_url = "https://openrouter.ai/api/v1"
    else:
        # Default OpenAI
        final_api_key = api_key or CHATGPT_API_KEY
        final_base_url = "https://api.openai.com/v1"

    # Create client with determined settings
    client = openai.OpenAI(api_key=final_api_key, base_url=final_base_url)
    return client


def ChatGPT_API_with_finish_reason(model, prompt, api_key=None, base_url=None, chat_history=None, track_cost=True, response_format=None):
    """
    Make API call with finish reason tracking.

    Args:
        model: Model name
        prompt: User prompt
        api_key: Optional API key
        base_url: Optional base URL
        chat_history: Optional chat history
        track_cost: Whether to track cost (default: True)

    Returns:
        Tuple of (response_content, finish_reason)
    """
    max_retries = 15
    client = _create_client(api_key=api_key, base_url=base_url)

    for i in range(max_retries):
        try:
            if chat_history:
                messages = chat_history.copy()
                messages.append({"role": "user", "content": prompt})
            else:
                messages = [{"role": "user", "content": prompt}]

            # Count input tokens for cost tracking
            input_text = " ".join([msg["content"] for msg in messages])
            input_tokens = count_tokens(input_text, model=model) if track_cost else 0

            params = {
                'model': model,
                'messages': messages,
                'temperature': 0,
            }
            if response_format:
                params['response_format'] = response_format

            response = client.chat.completions.create(**params)

            # Track cost
            if track_cost:
                output_tokens = response.usage.completion_tokens if hasattr(response, 'usage') and response.usage else 0
                tracker = get_global_tracker()
                tracker.track_call(
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    function_name="ChatGPT_API_with_finish_reason"
                )

            if response.choices[0].finish_reason == "length":
                return response.choices[0].message.content, "max_output_reached"
            else:
                return response.choices[0].message.content, "finished"

        except Exception as e:
            wait_time = (2 ** i) + random.random()
            error_msg = str(e)
            
            if "429" in error_msg or "rate_limit" in error_msg.lower():
                # On rate limit, wait longer
                wait_time = min(60, wait_time * 2 + 5)
                logger.warning(f"Rate limit hit. Retrying in {wait_time:.2f}s... (Attempt {i+1}/{max_retries})")
            else:
                logger.warning(f"Error: {e}. Retrying in {wait_time:.2f}s... (Attempt {i+1}/{max_retries})")
            
            if i < max_retries - 1:
                time.sleep(wait_time)
            else:
                logger.error('Max retries reached for prompt: ' + prompt[:100] + "...")
                return "Error", "error"


def ChatGPT_API(model, prompt, api_key=None, base_url=None, chat_history=None, track_cost=True, response_format=None):
    """
    Make synchronous API call.

    Args:
        model: Model name
        prompt: User prompt
        api_key: Optional API key
        base_url: Optional base URL
        chat_history: Optional chat history
        track_cost: Whether to track cost (default: True)

    Returns:
        Response content string
    """
    max_retries = 15
    client = _create_client(api_key=api_key, base_url=base_url)

    for i in range(max_retries):
        try:
            if chat_history:
                messages = chat_history.copy()
                messages.append({"role": "user", "content": prompt})
            else:
                messages = [{"role": "user", "content": prompt}]

            # Count input tokens for cost tracking
            input_text = " ".join([msg["content"] for msg in messages])
            input_tokens = count_tokens(input_text, model=model) if track_cost else 0

            params = {
                'model': model,
                'messages': messages,
                'temperature': 0,
            }
            if response_format:
                params['response_format'] = response_format

            response = client.chat.completions.create(**params)

            # Track cost
            if track_cost:
                output_tokens = response.usage.completion_tokens if hasattr(response, 'usage') and response.usage else 0
                tracker = get_global_tracker()
                tracker.track_call(
                    model=model,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    function_name="ChatGPT_API"
                )

            return response.choices[0].message.content

        except Exception as e:
            wait_time = (2 ** i) + random.random()
            error_msg = str(e)
            
            if "429" in error_msg or "rate_limit" in error_msg.lower():
                wait_time = min(60, wait_time * 2 + 5)
                logger.warning(f"Rate limit hit. Retrying in {wait_time:.2f}s... (Attempt {i+1}/{max_retries})")
            else:
                logger.warning(f"Error: {e}. Retrying in {wait_time:.2f}s... (Attempt {i+1}/{max_retries})")
                
            if i < max_retries - 1:
                time.sleep(wait_time)
            else:
                logger.error('Max retries reached for prompt: ' + prompt[:100] + "...")
                return "Error"


async def ChatGPT_API_async(model, prompt, api_key=None, base_url=None, track_cost=True, response_format=None):
    """
    Make asynchronous API call.

    Args:
        model: Model name
        prompt: User prompt
        api_key: Optional API key
        base_url: Optional base URL
        track_cost: Whether to track cost (default: True)
        response_format: Optional response format

    Returns:
        Response content string
    """
    max_retries = 15
    messages = [{"role": "user", "content": prompt}]
    
    # Use semaphore for concurrency control
    sem = get_semaphore()

    for i in range(max_retries):
        try:
            # If we have a semaphore, use it
            if sem:
                async with sem:
                    return await _do_chatgpt_api_async(model, messages, api_key, base_url, track_cost, response_format)
            else:
                return await _do_chatgpt_api_async(model, messages, api_key, base_url, track_cost, response_format)

        except Exception as e:
            wait_time = (2 ** i) + random.random()
            error_msg = str(e)
            
            if "429" in error_msg or "rate_limit" in error_msg.lower():
                wait_time = min(60, wait_time * 2 + 5)
                logger.warning(f"Rate limit hit (async). Retrying in {wait_time:.2f}s... (Attempt {i+1}/{max_retries})")
            else:
                logger.warning(f"Error (async): {e}. Retrying in {wait_time:.2f}s... (Attempt {i+1}/{max_retries})")
                
            if i < max_retries - 1:
                await asyncio.sleep(wait_time)
            else:
                logger.error('Max retries reached (async) for prompt: ' + prompt[:100] + "...")
                return "Error"

async def _do_chatgpt_api_async(model, messages, api_key, base_url, track_cost, response_format=None):
    """Internal helper for async API call"""
    # Create sync client first to get config, then create async client
    sync_client = _create_client(api_key=api_key, base_url=base_url)
    async_client = openai.AsyncOpenAI(
        api_key=sync_client.api_key,
        base_url=sync_client.base_url
    )

    # Count input tokens for cost tracking
    prompt = messages[0]["content"]
    input_tokens = count_tokens(prompt, model=model) if track_cost else 0

    params = {
        'model': model,
        'messages': messages,
        'temperature': 0,
    }
    if response_format:
        params['response_format'] = response_format

    response = await async_client.chat.completions.create(**params)

    # Track cost
    if track_cost:
        output_tokens = response.usage.completion_tokens if hasattr(response, 'usage') and response.usage else 0
        tracker = get_global_tracker()
        tracker.track_call(
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            function_name="ChatGPT_API_async"
        )

    return response.choices[0].message.content


def make_api_call(config, model=None, prompt=None, chat_history=None, api_function=None, track_cost=True, response_format=None):
    """
    Unified API call function that handles model resolution and provider configuration.

    Args:
        config: Configuration object with api_provider, api_base_url, etc.
        model: Model name to use (overrides config.model if provided)
        prompt: The prompt to send to the API
        chat_history: Optional chat history
        api_function: Which API function to call ('async', 'with_finish_reason', or sync)
        track_cost: Whether to track cost (default: True)
        response_format: Optional response format

    Returns:
        Response from the API call
    """
    # Resolve the model name
    resolved_model = resolve_model_name(
        model or config.model,
        config.api_provider,
        getattr(config, 'model_name', None),
        getattr(config, 'env_model_var', None)
    )

    # Prepare API parameters
    api_params = {
        'model': resolved_model,
        'prompt': prompt,
        'api_key': None,  # Uses environment variables
        'base_url': getattr(config, 'api_base_url', None),
        'track_cost': track_cost,
        'response_format': response_format
    }

    if chat_history is not None:
        api_params['chat_history'] = chat_history

    # Make the appropriate API call
    if api_function == 'async':
        # Note: This returns a coroutine and needs to be awaited
        return ChatGPT_API_async(**api_params)
    elif api_function == 'with_finish_reason':
        return ChatGPT_API_with_finish_reason(**api_params)
    else:  # default sync
        return ChatGPT_API(**api_params)


# Re-export for backward compatibility
__all__ = [
    'ChatGPT_API',
    'ChatGPT_API_async',
    'ChatGPT_API_with_finish_reason',
    'make_api_call',
    'resolve_model_name',
    '_create_client',
]
