"""
Cost tracking module for PageIndex API calls.

Tracks token usage and calculates costs for different models and providers.
"""
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

logger = logging.getLogger(__name__)


# Pricing data (per 1M tokens)
# Prices as of 2024
MODEL_PRICING = {
    # OpenAI Models
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4-turbo": {"input": 10.00, "output": 30.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},

    # OpenRouter Models (using base model names)
    "openai/gpt-4o": {"input": 2.50, "output": 10.00},
    "openai/gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "anthropic/claude-3-5-sonnet-20241022": {"input": 3.00, "output": 15.00},
    "anthropic/claude-haiku-4.5": {"input": 0.10, "output": 0.50},
    "google/gemini-2.5-flash-lite-preview-09-2025": {"input": 0.075, "output": 0.30},
    "google/gemini-3-flash-preview": {"input": 0.075, "output": 0.30},
    "meta-llama/llama-3.1-70b-instruct": {"input": 0.59, "output": 0.79},

    # Free models
    "nvidia/nemotron-3-nano-30b-a3b:free": {"input": 0.00, "output": 0.00},
}


@dataclass
class APICallRecord:
    """Record of a single API call"""
    timestamp: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    input_cost: float
    output_cost: float
    total_cost: float
    function_name: str


class CostTracker:
    """
    Track token usage and costs for API calls.

    Usage:
        tracker = CostTracker()

        # Track an API call
        tracker.track_call(
            model="gpt-4o",
            input_tokens=1000,
            output_tokens=500,
            function_name="generate_summary"
        )

        # Get total cost
        total_cost = tracker.get_total_cost()

        # Get summary
        summary = tracker.get_summary()

        # Save to file
        tracker.save_to_file("costs.json")

        # Load from file
        tracker.load_from_file("costs.json")
    """

    def __init__(self):
        self.calls: List[APICallRecord] = []
        self.total_cost = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_tokens = 0
        self.costs_by_model: Dict[str, float] = defaultdict(float)
        self.tokens_by_model: Dict[str, Tuple[int, int, int]] = {}

    def get_pricing(self, model: str) -> Tuple[float, float]:
        """
        Get pricing for a model.

        Args:
            model: Model name

        Returns:
            Tuple of (input_price_per_million, output_price_per_million)
        """
        # Try exact match
        if model in MODEL_PRICING:
            pricing = MODEL_PRICING[model]
            return pricing["input"], pricing["output"]

        # Try prefix match (e.g., "gpt-4o-2024-11-20" matches "gpt-4o")
        # We check from longest to shortest keys to avoid greedy matches
        sorted_keys = sorted(MODEL_PRICING.keys(), key=len, reverse=True)
        for key in sorted_keys:
            if "/" not in key and model.startswith(key):
                pricing = MODEL_PRICING[key]
                return pricing["input"], pricing["output"]

        # Try to match base model name for OpenRouter models
        if "/" in model:
            parts = model.split("/")
            if len(parts) > 1:
                base_model = parts[1]
                # Strip ":free" or other suffixes if present for matching
                base_model_clean = base_model.split(":")[0]
                
                for key, pricing in MODEL_PRICING.items():
                    if base_model_clean in key or key.endswith(base_model_clean):
                        return pricing["input"], pricing["output"]

        # Default pricing (estimate)
        logger.warning(f"No pricing found for model {model}, using default estimate")
        return 1.0, 2.0

    def calculate_cost(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int
    ) -> Tuple[float, float, float]:
        """
        Calculate cost for an API call.

        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens

        Returns:
            Tuple of (input_cost, output_cost, total_cost)
        """
        input_price, output_price = self.get_pricing(model)

        input_cost = (input_tokens / 1_000_000) * input_price
        output_cost = (output_tokens / 1_000_000) * output_price
        total_cost = input_cost + output_cost

        return input_cost, output_cost, total_cost

    def track_call(
        self,
        model: str,
        input_tokens: int,
        output_tokens: int,
        function_name: str = "unknown"
    ) -> APICallRecord:
        """
        Track an API call.

        Args:
            model: Model name
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            function_name: Name of the function making the call

        Returns:
            APICallRecord with the call details
        """
        input_cost, output_cost, total_cost = self.calculate_cost(
            model, input_tokens, output_tokens
        )

        total_tokens = input_tokens + output_tokens

        record = APICallRecord(
            timestamp=datetime.now().isoformat(),
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            input_cost=round(input_cost, 6),
            output_cost=round(output_cost, 6),
            total_cost=round(total_cost, 6),
            function_name=function_name
        )

        self.calls.append(record)

        # Update totals
        self.total_cost += total_cost
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_tokens += total_tokens

        # Update by-model stats
        self.costs_by_model[model] += total_cost
        self.tokens_by_model[model] = (
            self.tokens_by_model.get(model, (0, 0, 0))[0] + input_tokens,
            self.tokens_by_model.get(model, (0, 0, 0))[1] + output_tokens,
            self.tokens_by_model.get(model, (0, 0, 0))[2] + total_tokens
        )

        logger.info(
            f"API Call: {function_name} | Model: {model} | "
            f"Tokens: {input_tokens:,} in, {output_tokens:,} out | "
            f"Cost: ${total_cost:.6f}"
        )

        return record

    def get_total_cost(self) -> float:
        """Get total cost across all calls."""
        return round(self.total_cost, 6)

    def get_total_tokens(self) -> Tuple[int, int, int]:
        """Get total tokens across all calls."""
        return self.total_input_tokens, self.total_output_tokens, self.total_tokens

    def get_summary(self) -> Dict:
        """
        Get summary of all tracked calls.

        Returns:
            Dict with summary statistics
        """
        return {
            "total_calls": len(self.calls),
            "total_cost": round(self.total_cost, 6),
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "costs_by_model": {
                model: round(cost, 6)
                for model, cost in self.costs_by_model.items()
            },
            "calls_by_model": {
                model: sum(1 for call in self.calls if call.model == model)
                for model in self.costs_by_model.keys()
            },
            "average_cost_per_call": round(
                self.total_cost / len(self.calls), 6
            ) if self.calls else 0.0
        }

    def get_calls_by_function(self, function_name: str) -> List[APICallRecord]:
        """Get all calls for a specific function."""
        return [call for call in self.calls if call.function_name == function_name]

    def estimate_cost(
        self,
        model: str,
        estimated_input_tokens: int,
        estimated_output_tokens: int = None,
        multiplier: float = 1.5
    ) -> Dict:
        """
        Estimate cost for a document analysis task.

        Args:
            model: Model name
            estimated_input_tokens: Estimated input tokens
            estimated_output_tokens: Estimated output tokens (defaults to 20% of input)
            multiplier: Safety multiplier (default 1.5x for estimates)

        Returns:
            Dict with cost estimate
        """
        if estimated_output_tokens is None:
            estimated_output_tokens = int(estimated_input_tokens * 0.2)

        input_cost, output_cost, total_cost = self.calculate_cost(
            model,
            int(estimated_input_tokens * multiplier),
            int(estimated_output_tokens * multiplier)
        )

        return {
            "model": model,
            "estimated_input_tokens": int(estimated_input_tokens * multiplier),
            "estimated_output_tokens": int(estimated_output_tokens * multiplier),
            "estimated_total_tokens": int(
                (estimated_input_tokens + estimated_output_tokens) * multiplier
            ),
            "estimated_input_cost": round(input_cost, 6),
            "estimated_output_cost": round(output_cost, 6),
            "estimated_total_cost": round(total_cost, 6),
            "multiplier_used": multiplier
        }

    def save_to_file(self, filepath: str):
        """Save tracking data to JSON file."""
        data = {
            "summary": self.get_summary(),
            "calls": [asdict(call) for call in self.calls]
        }

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved cost tracking data to {filepath}")

    def load_from_file(self, filepath: str):
        """Load tracking data from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        # Reset current state
        self.__init__()

        # Load calls
        for call_data in data.get("calls", []):
            record = APICallRecord(**call_data)
            self.calls.append(record)

            # Recalculate totals
            self.total_cost += record.total_cost
            self.total_input_tokens += record.input_tokens
            self.total_output_tokens += record.output_tokens
            self.total_tokens += record.total_tokens
            self.costs_by_model[record.model] += record.total_cost

        logger.info(f"Loaded {len(self.calls)} cost records from {filepath}")

    def print_summary(self):
        """Print a formatted summary to console."""
        summary = self.get_summary()

        print("\n" + "=" * 60)
        print("💰 COST TRACKING SUMMARY")
        print("=" * 60)
        print(f"Total Calls: {summary['total_calls']}")
        print(f"Total Cost: ${summary['total_cost']:.6f}")
        print(f"Total Tokens: {summary['total_tokens']:,}")
        print(f"  - Input: {summary['total_input_tokens']:,}")
        print(f"  - Output: {summary['total_output_tokens']:,}")
        print(f"\nAverage Cost per Call: ${summary['average_cost_per_call']:.6f}")

        if summary['costs_by_model']:
            print("\nCosts by Model:")
            for model, cost in sorted(summary['costs_by_model'].items(), key=lambda x: -x[1]):
                calls = summary['calls_by_model'][model]
                print(f"  {model}: ${cost:.6f} ({calls} calls)")

        print("=" * 60 + "\n")

    def reset(self):
        """Reset all tracking data."""
        self.__init__()


# Global cost tracker instance
_global_tracker: Optional[CostTracker] = None


def get_global_tracker() -> CostTracker:
    """Get or create the global cost tracker instance."""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = CostTracker()
    return _global_tracker


def reset_global_tracker():
    """Reset the global cost tracker."""
    global _global_tracker
    _global_tracker = None
