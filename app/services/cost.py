from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelPrice:
    """Price per 1M tokens in USD."""

    input_per_1m: float
    output_per_1m: float


# Example baseline prices (easy to update in one place).
MODEL_PRICES: dict[str, ModelPrice] = {
    "gpt-4o": ModelPrice(input_per_1m=2.50, output_per_1m=10.00),
    "claude-3-5-sonnet": ModelPrice(input_per_1m=3.00, output_per_1m=15.00),
    "gemini-pro": ModelPrice(input_per_1m=1.25, output_per_1m=5.00),
}


def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """
    Calculate estimated USD cost for a request.

    Returns 0.0 when the model has no configured price.
    """
    if prompt_tokens < 0 or completion_tokens < 0:
        raise ValueError("Token counts must be non-negative")

    price = MODEL_PRICES.get(model)
    if price is None:
        return 0.0

    prompt_cost = (prompt_tokens / 1_000_000) * price.input_per_1m
    completion_cost = (completion_tokens / 1_000_000) * price.output_per_1m
    return round(prompt_cost + completion_cost, 10)


class CostService:
    """Thin wrapper for cost calculations with overridable price table."""

    def __init__(self, prices: dict[str, ModelPrice] | None = None) -> None:
        self._prices = prices or MODEL_PRICES

    def calculate_cost(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
    ) -> float:
        if prompt_tokens < 0 or completion_tokens < 0:
            raise ValueError("Token counts must be non-negative")
        price = self._prices.get(model)
        if price is None:
            return 0.0
        prompt_cost = (prompt_tokens / 1_000_000) * price.input_per_1m
        completion_cost = (completion_tokens / 1_000_000) * price.output_per_1m
        return round(prompt_cost + completion_cost, 10)
