"""
Token utilities for PageIndex.

Provides token counting and validation functions.
"""
import tiktoken
import logging

logger = logging.getLogger(__name__)


def count_tokens(text, model=None):
    """
    Count tokens in text using tiktoken.

    Args:
        text: Text to count tokens for
        model: Model name for encoding (defaults to gpt-4o if not specified)

    Returns:
        Number of tokens
    """
    if not text:
        return 0

    try:
        # Use specified model or default to gpt-4o
        model_for_encoding = model or "gpt-4o"

        # For OpenRouter models, extract the base model name
        if "/" in model_for_encoding:
            # Try to get the provider/model parts
            parts = model_for_encoding.split("/")
            if len(parts) == 2:
                provider, base_model = parts
                # Map common providers to OpenAI-compatible models
                if provider == "openai":
                    model_for_encoding = base_model
                elif provider in ["anthropic", "google", "meta-llama"]:
                    # For non-OpenAI models, use cl100k_base as a reasonable approximation
                    model_for_encoding = "cl100k_base"
                else:
                    # Use a general encoding as fallback
                    model_for_encoding = "cl100k_base"

        # Try to get the encoding
        enc = tiktoken.encoding_for_model(model_for_encoding)
        tokens = enc.encode(text)
        return len(tokens)
    except Exception as e:
        # Silently fall back to character-based estimation for unknown models
        # This avoids spamming warnings for legitimate OpenRouter models
        # Rough approximation: 1 token ≈ 4 characters for most LLMs
        return len(text) // 4


def check_token_limit(structure, limit=110000):
    """
    Check if structure exceeds token limit.

    Args:
        structure: Document structure
        limit: Maximum allowed tokens (default: 110000)

    Returns:
        True if within limit, False otherwise
    """
    from .document_utils import structure_to_list

    num_tokens = 0
    nodes = structure_to_list(structure)

    for node in nodes:
        if 'text' in node:
            num_tokens += count_tokens(node['text'], model='gpt-4o')

    if num_tokens > limit:
        logger.warning(f"Token count {num_tokens} exceeds limit {limit}")

    return num_tokens <= limit


def estimate_tokens_from_pages(num_pages, avg_words_per_page=500):
    """
    Estimate tokens from number of pages.

    Args:
        num_pages: Number of pages
        avg_words_per_page: Average words per page (default: 500)

    Returns:
        Estimated token count
    """
    # Rough estimate: 1 token ≈ 0.75 words
    estimated_words = num_pages * avg_words_per_page
    estimated_tokens = int(estimated_words / 0.75)
    return estimated_tokens


def estimate_tokens_from_text_length(text_length):
    """
    Estimate tokens from text length.

    Args:
        text_length: Length of text in characters

    Returns:
        Estimated token count
    """
    # Rough estimate: 1 token ≈ 4 characters
    return text_length // 4


__all__ = [
    'count_tokens',
    'check_token_limit',
    'estimate_tokens_from_pages',
    'estimate_tokens_from_text_length',
]
