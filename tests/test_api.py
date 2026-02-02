"""
Pytest tests for PageIndex core functionality.

Run with: pytest tests/
"""
import pytest
import asyncio
from pageindex.utils import ChatGPT_API_async, resolve_model_name
from pageindex.cost_tracker import get_global_tracker, reset_global_tracker


class TestAPI:
    """Test API client functionality."""

    @pytest.mark.api
    @pytest.mark.asyncio
    async def test_chatgpt_api_async(self):
        """Test basic async API call."""
        response = await ChatGPT_API_async(
            model="gpt-4o-mini",
            prompt="What is 2+2? Answer with one word."
        )
        assert response
        assert len(response) > 0

    @pytest.mark.api
    def test_resolve_model_name(self):
        """Test model name resolution."""
        # Test OpenRouter prefix addition
        model = resolve_model_name(
            model="gpt-4o",
            api_provider="openrouter"
        )
        assert model == "openai/gpt-4o"

        # Test with explicit model_name
        model = resolve_model_name(
            model="gpt-4o",
            model_name="anthropic/claude-3-5-sonnet-20241022"
        )
        assert model == "anthropic/claude-3-5-sonnet-20241022"

    @pytest.mark.api
    @pytest.mark.asyncio
    async def test_cost_tracking(self):
        """Test cost tracking functionality."""
        reset_global_tracker()
        tracker = get_global_tracker()

        # Make a test API call
        await ChatGPT_API_async(
            model="gpt-4o-mini",
            prompt="Test"
        )

        summary = tracker.get_summary()
        assert summary['total_calls'] == 1
        assert summary['total_cost'] > 0


class TestEnvironmentModels:
    """Test environment variable model resolution."""

    @pytest.mark.api
    def test_model_free_resolution(self):
        """Test MODEL_FREE environment variable resolution."""
        import os
        model_free = os.getenv("MODEL_FREE")
        if not model_free:
            pytest.skip("MODEL_FREE not set in environment")

        resolved = resolve_model_name(
            model="gpt-4o",
            env_model_var="MODEL_FREE"
        )
        assert resolved == model_free

    @pytest.mark.api
    def test_model_fast_resolution(self):
        """Test MODEL_FAST environment variable resolution."""
        import os
        model_fast = os.getenv("MODEL_FAST")
        if not model_fast:
            pytest.skip("MODEL_FAST not set in environment")

        resolved = resolve_model_name(
            model="gpt-4o",
            env_model_var="MODEL_FAST"
        )
        assert resolved == model_fast

    @pytest.mark.api
    def test_model_reasoning_resolution(self):
        """Test MODEL_REASONING environment variable resolution."""
        import os
        model_reasoning = os.getenv("MODEL_REASONING")
        if not model_reasoning:
            pytest.skip("MODEL_REASONING not set in environment")

        resolved = resolve_model_name(
            model="gpt-4o",
            env_model_var="MODEL_REASONING"
        )
        assert resolved == model_reasoning
