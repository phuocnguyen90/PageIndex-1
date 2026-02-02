#!/usr/bin/env python3
"""
Test script to verify OpenRouter integration
"""
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from pageindex.utils import resolve_model_name, _create_client

load_dotenv()

def test_model_resolution():
    """Test model name resolution for different providers"""
    print("Testing model name resolution...")

    # Test OpenAI model
    model = resolve_model_name("gpt-4o", api_provider="openai")
    print(f"OpenAI model: {model}")
    assert model == "gpt-4o", f"Expected gpt-4o, got {model}"

    # Test OpenRouter model with OpenAI prefix
    model = resolve_model_name("gpt-4o", api_provider="openrouter")
    print(f"OpenRouter OpenAI model: {model}")
    assert model == "openai/gpt-4o", f"Expected openai/gpt-4o, got {model}"

    # Test OpenRouter model with Anthropic prefix
    model = resolve_model_name("claude-3-5-sonnet-20241022", api_provider="openrouter")
    print(f"OpenRouter Anthropic model: {model}")
    assert model == "anthropic/claude-3-5-sonnet-20241022", f"Expected anthropic/claude-3-5-sonnet-20241022, got {model}"

    # Test explicit model name override
    model = resolve_model_name("gpt-4o", api_provider="openrouter", model_name="custom/model")
    print(f"Explicit model name: {model}")
    assert model == "custom/model", f"Expected custom/model, got {model}"

    print("✅ Model resolution tests passed!")

def test_client_creation():
    """Test client creation with different configurations"""
    print("\nTesting client creation...")

    # Test default client creation
    client = _create_client()
    print(f"Default client base URL: {client.base_url}")
    # Since OPENROUTER_API_KEY is set, it will use OpenRouter by default
    if str(client.base_url) == "https://openrouter.ai/api/v1/":
        print("✅ Using OpenRouter by default (expected when OPENROUTER_API_KEY is set)")
    elif str(client.base_url) == "https://api.openai.com/v1/":
        print("✅ Using OpenAI by default")
    else:
        assert False, f"Expected either OpenAI or OpenRouter base URL, got {client.base_url}"

    # Test OpenRouter client creation if API key is available
    openrouter_key = os.getenv("OPENROUTER_API_KEY")
    if openrouter_key:
        client = _create_client(base_url="https://openrouter.ai/api/v1")
        print(f"OpenRouter base URL: {client.base_url}")
        assert str(client.base_url) == "https://openrouter.ai/api/v1/", f"Expected OpenRouter base URL, got {client.base_url}"
        print("✅ OpenRouter client creation test passed!")
    else:
        print("⚠️  OPENROUTER_API_KEY not set, skipping OpenRouter client test")

    # Test custom endpoint client creation
    custom_url = "https://api.example.com/v1/"
    client = _create_client(base_url=custom_url)
    print(f"Custom endpoint URL: {client.base_url}")
    assert str(client.base_url) == custom_url, f"Expected {custom_url}, got {client.base_url}"
    print("✅ Custom endpoint client creation test passed!")

def main():
    print("🧪 Testing PageIndex OpenRouter Integration\n")

    try:
        test_model_resolution()
        test_client_creation()
        print("\n✅ All tests passed! OpenRouter integration is working correctly.")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return 1

    return 0

if __name__ == "__main__":
    exit(main())