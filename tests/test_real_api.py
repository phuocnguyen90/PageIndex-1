#!/usr/bin/env python3
"""
Test script to verify actual API calls with models from .env
"""
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncio
from dotenv import load_dotenv
from pageindex.utils import ChatGPT_API_async, resolve_model_name

load_dotenv()

async def test_model(model_name, model_label, prompt="What is 2+2? Please answer in one word."):
    """Test a single model with an API call"""
    try:
        print(f"Testing {model_label}...")
        print(f"   Model: {model_name}")

        response = await ChatGPT_API_async(
            model=model_name,
            prompt=prompt
        )
        print(f"   Response: {response}")
        print(f"   ✅ {model_label} test passed!\n")
        return True
    except Exception as e:
        print(f"   ❌ {model_label} test failed: {e}\n")
        return False

async def test_api_call():
    """Test API calls with models from .env"""
    print("🧪 Testing Real API Calls with .env Models\n")
    print("=" * 60)

    # Get models from environment
    model_free = os.getenv("MODEL_FREE")
    model_fast = os.getenv("MODEL_FAST")
    model_reasoning = os.getenv("MODEL_REASONING")

    results = {}

    # Test MODEL_FREE
    if model_free:
        print("\n1. Testing MODEL_FREE")
        print("-" * 60)
        results["MODEL_FREE"] = await test_model(model_free, "MODEL_FREE")
    else:
        print("\n1. ⚠️  MODEL_FREE not set in environment\n")
        results["MODEL_FREE"] = None

    # Test MODEL_FAST
    if model_fast:
        print("\n2. Testing MODEL_FAST")
        print("-" * 60)
        results["MODEL_FAST"] = await test_model(model_fast, "MODEL_FAST")
    else:
        print("\n2. ⚠️  MODEL_FAST not set in environment\n")
        results["MODEL_FAST"] = None

    # Test MODEL_REASONING
    if model_reasoning:
        print("\n3. Testing MODEL_REASONING")
        print("-" * 60)
        results["MODEL_REASONING"] = await test_model(model_reasoning, "MODEL_REASONING")
    else:
        print("\n3. ⚠️  MODEL_REASONING not set in environment\n")
        results["MODEL_REASONING"] = None

    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)

    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)

    for model_name, result in results.items():
        if result is True:
            status = "✅ PASSED"
        elif result is False:
            status = "❌ FAILED"
        else:
            status = "⚠️  SKIPPED (not set)"
        print(f"   {model_name}: {status}")

    print(f"\n   Total: {passed} passed, {failed} failed, {skipped} skipped")
    print("=" * 60)

def main():
    print("🧪 Testing Real API Calls\n")
    asyncio.run(test_api_call())

if __name__ == "__main__":
    main()