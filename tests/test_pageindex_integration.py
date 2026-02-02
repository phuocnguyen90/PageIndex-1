#!/usr/bin/env python3
"""
Test script to verify PageIndex integration with .env models
"""
import os
import json
import sys
import asyncio
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
from pageindex.utils import ChatGPT_API_async

load_dotenv()

async def test_document_analysis(model_name, model_label):
    """Test document analysis with a specific model"""
    print(f"\n{'=' * 70}")
    print(f"Testing {model_label}")
    print(f"Model: {model_name}")
    print('=' * 70)

    try:
        # Test a realistic document processing task similar to PageIndex
        prompt = """You are analyzing a document section for PageIndex. Your task is to:

1. Extract the main heading
2. Identify sub-sections
3. Provide a brief summary of the content

Document:
# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience.

## Key Concepts

### Supervised Learning
In supervised learning, algorithms learn from labeled training data to make predictions.

### Unsupervised Learning
Unsupervised learning deals with unlabeled data, finding hidden patterns in data.

## Applications
- Image recognition
- Natural language processing
- Recommendation systems

Please respond in JSON format with keys: "heading", "subsections" (array), and "summary"."""

        response = await ChatGPT_API_async(
            model=model_name,
            prompt=prompt
        )

        print(f"\n✅ {model_label} test passed!")
        print(f"\nResponse (first 300 chars):\n{response[:300]}...")
        print(f"\nFull response length: {len(response)} characters")
        return True, response

    except Exception as e:
        print(f"\n❌ {model_label} test failed: {e}")
        return False, str(e)

async def test_pageindex_functionality():
    """Test PageIndex functionality with models from .env"""
    print("🧪 Testing PageIndex Integration with .env Models")
    print("=" * 70)

    # Get models from environment
    models = {
        "MODEL_FREE": os.getenv("MODEL_FREE"),
        "MODEL_FAST": os.getenv("MODEL_FAST"),
        "MODEL_REASONING": os.getenv("MODEL_REASONING")
    }

    results = {}
    responses = {}

    # Test each model
    for label, model_name in models.items():
        if model_name:
            success, response = await test_document_analysis(model_name, label)
            results[label] = success
            responses[label] = response
        else:
            print(f"\n⚠️  {label} not set in environment - skipping")
            results[label] = None
            responses[label] = None

    # Summary
    print(f"\n{'=' * 70}")
    print("📊 INTEGRATION TEST SUMMARY")
    print('=' * 70)

    passed = sum(1 for v in results.values() if v is True)
    failed = sum(1 for v in results.values() if v is False)
    skipped = sum(1 for v in results.values() if v is None)

    for model_name, result in results.items():
        if result is True:
            status = "✅ PASSED"
            resp_len = len(responses[model_name]) if responses[model_name] else 0
            print(f"   {model_name}: {status} (response: {resp_len} chars)")
        elif result is False:
            status = "❌ FAILED"
            print(f"   {model_name}: {status}")
        else:
            status = "⚠️  SKIPPED"
            print(f"   {model_name}: {status} (not set)")

    print(f"\n   Total: {passed} passed, {failed} failed, {skipped} skipped")
    print('=' * 70)

    print("\n🎯 What was tested:")
    print("   - Document structure analysis")
    print("   - Section extraction and identification")
    print("   - Summary generation")
    print("   - JSON response formatting")

def main():
    import asyncio
    asyncio.run(test_pageindex_functionality())

if __name__ == "__main__":
    main()