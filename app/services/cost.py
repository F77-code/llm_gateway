from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelPrice:
    """Price per 1M tokens in USD."""

    input_per_1m: float
    output_per_1m: float


# Prices per 1M tokens in USD. Kept in sync with Settings.provider_by_model.
# Update when providers change their pricing.
MODEL_PRICES: dict[str, ModelPrice] = {
    # DeepSeek
    "deepseek-chat": ModelPrice(input_per_1m=0.14, output_per_1m=0.28),
    "deepseek-reasoner": ModelPrice(input_per_1m=0.55, output_per_1m=2.19),
    # Google Gemini
    "gemini-2.5-pro": ModelPrice(input_per_1m=1.25, output_per_1m=10.00),
    "gemini-2.5-flash": ModelPrice(input_per_1m=0.15, output_per_1m=0.60),
    "gemini-2.5-flash-lite": ModelPrice(input_per_1m=0.10, output_per_1m=0.40),
    "gemini-2.0-flash": ModelPrice(input_per_1m=0.10, output_per_1m=0.40),
    "gemini-2.0-flash-lite": ModelPrice(input_per_1m=0.075, output_per_1m=0.30),
    # xAI Grok
    "grok-4-1-fast-non-reasoning": ModelPrice(input_per_1m=2.00, output_per_1m=10.00),
    "grok-4-1-fast-reasoning": ModelPrice(input_per_1m=3.00, output_per_1m=15.00),
    # Perplexity Sonar
    "sonar": ModelPrice(input_per_1m=1.00, output_per_1m=1.00),
    "sonar-pro": ModelPrice(input_per_1m=3.00, output_per_1m=15.00),
    "sonar-reasoning-pro": ModelPrice(input_per_1m=2.00, output_per_1m=8.00),
    "sonar-deep-research": ModelPrice(input_per_1m=2.00, output_per_1m=8.00),
    # Anthropic Claude
    "claude-opus-4-6": ModelPrice(input_per_1m=15.00, output_per_1m=75.00),
    "claude-sonnet-4-5-20250929": ModelPrice(input_per_1m=3.00, output_per_1m=15.00),
    "claude-haiku-4-5-20251001": ModelPrice(input_per_1m=0.80, output_per_1m=4.00),
    # OpenAI GPT-5 family
    "gpt-5": ModelPrice(input_per_1m=2.50, output_per_1m=10.00),
    "gpt-5-mini": ModelPrice(input_per_1m=0.60, output_per_1m=2.40),
    "gpt-5-nano": ModelPrice(input_per_1m=0.15, output_per_1m=0.60),
    "gpt-5.1": ModelPrice(input_per_1m=3.00, output_per_1m=10.00),
    "gpt-5.2": ModelPrice(input_per_1m=5.00, output_per_1m=15.00),
    "gpt-5.3-chat-latest": ModelPrice(input_per_1m=7.50, output_per_1m=22.50),
    "gpt-5.4": ModelPrice(input_per_1m=10.00, output_per_1m=30.00),
    "gpt-4.1-mini": ModelPrice(input_per_1m=0.40, output_per_1m=1.60),
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
