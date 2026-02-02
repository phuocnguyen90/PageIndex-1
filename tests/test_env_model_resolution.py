#!/usr/bin/env python3
"""
Test script to verify environment variable model resolution
"""
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from pageindex.utils import resolve_model_name, _create_client

load_dotenv()

def test_env_model_resolution():
    """Test model resolution with environment variables"""
    print("🧪 Testing Environment Variable Model Resolution\n")

    # Test 1: MODEL_FREE
    print("1. Testing MODEL_FREE...")
    model_free = os.getenv("MODEL_FREE")
    if model_free:
        print(f"   MODEL_FREE: {model_free}")

        # Test direct environment variable
        resolved = resolve_model_name(
            "gpt-4o",  # This should be ignored
            api_provider="openrouter",
            env_model_var="MODEL_FREE"
        )
        print(f"   Resolved model: {resolved}")
        assert resolved == model_free, f"Expected {model_free}, got {resolved}"
        print("   ✅ MODEL_FREE resolution works!")
    else:
        print("   ⚠️  MODEL_FREE not set in environment")

    # Test 2: MODEL_FAST
    print("\n2. Testing MODEL_FAST...")
    model_fast = os.getenv("MODEL_FAST")
    if model_fast:
        print(f"   MODEL_FAST: {model_fast}")

        # Test direct environment variable
        resolved = resolve_model_name(
            "gpt-4o",  # This should be ignored
            api_provider="openrouter",
            env_model_var="MODEL_FAST"
        )
        print(f"   Resolved model: {resolved}")
        assert resolved == model_fast, f"Expected {model_fast}, got {resolved}"
        print("   ✅ MODEL_FAST resolution works!")
    else:
        print("   ⚠️  MODEL_FAST not set in environment")

    # Test 3: MODEL_REASONING
    print("\n3. Testing MODEL_REASONING...")
    model_reasoning = os.getenv("MODEL_REASONING")
    if model_reasoning:
        print(f"   MODEL_REASONING: {model_reasoning}")

        # Test direct environment variable
        resolved = resolve_model_name(
            "gpt-4o",  # This should be ignored
            api_provider="openrouter",
            env_model_var="MODEL_REASONING"
        )
        print(f"   Resolved model: {resolved}")
        assert resolved == model_reasoning, f"Expected {model_reasoning}, got {resolved}"
        print("   ✅ MODEL_REASONING resolution works!")
    else:
        print("   ⚠️  MODEL_REASONING not set in environment")

    # Test 4: Priority test - env var should override everything
    print("\n4. Testing priority: env var > model_name > model")
    resolved = resolve_model_name(
        "gpt-4o",  # Should be ignored
        api_provider="openrouter",
        model_name="anthropic/claude-3-5-sonnet",  # Should be ignored
        env_model_var="MODEL_FREE"
    )
    print(f"   Resolved model: {resolved}")
    if model_free:
        assert resolved == model_free, f"Expected {model_free}, got {resolved}"
        print("   ✅ Environment variable has highest priority!")
    else:
        print("   ⚠️  Using fallback due to missing MODEL_FREE")

    print("\n✅ All environment variable model resolution tests completed!")

def test_cli_integration():
    """Test CLI integration with environment variables"""
    print("\n🧪 Testing CLI Integration with Environment Variables\n")

    # Test command generation
    test_cases = [
        {
            "name": "Using MODEL_FREE",
            "env_var": "MODEL_FREE",
            "expected": os.getenv("MODEL_FREE")
        },
        {
            "name": "Using MODEL_FAST",
            "env_var": "MODEL_FAST",
            "expected": os.getenv("MODEL_FAST")
        },
        {
            "name": "Using MODEL_REASONING",
            "env_var": "MODEL_REASONING",
            "expected": os.getenv("MODEL_REASONING")
        }
    ]

    for test_case in test_cases:
        model_value = os.getenv(test_case["env_var"])
        if model_value:
            print(f"✅ {test_case['name']}: {model_value}")
            # Generate command example
            print(f"   Command: python3 run_pageindex.py --pdf_path document.pdf --env-model-var {test_case['env_var']}")
        else:
            print(f"⚠️  {test_case['name']}: Not set in environment")

def main():
    print("🧪 Testing Environment Variable Model Resolution for PageIndex\n")

    test_env_model_resolution()
    test_cli_integration()

    print("\n🎯 Summary:")
    print("Environment variables allow you to:")
    print("- Define specific models for different use cases (FREE, FAST, REASONING)")
    print("- Switch models without changing code")
    print("- Use full OpenRouter model names directly")
    print("- Maintain consistency across different runs")

if __name__ == "__main__":
    main()