"""Unit tests for the cost calculation service.

No fixtures needed — pure function logic.
"""

from __future__ import annotations

import pytest

from app.config import Settings
from app.services.cost import MODEL_PRICES, CostService, ModelPrice, calculate_cost


def test_model_prices_covers_all_configured_models():
    """Every model in Settings.provider_by_model must have a price entry."""
    missing = [m for m in Settings.provider_by_model if m not in MODEL_PRICES]
    assert not missing, f"Models without price entry: {missing}"


def test_calculate_cost_returns_zero_for_unknown_model():
    cost = calculate_cost("unknown-model-xyz", prompt_tokens=1000, completion_tokens=500)
    assert cost == 0.0


def test_calculate_cost_zero_tokens():
    model = next(iter(MODEL_PRICES))
    assert calculate_cost(model, prompt_tokens=0, completion_tokens=0) == 0.0


def test_calculate_cost_one_million_tokens():
    model = next(iter(MODEL_PRICES))
    price = MODEL_PRICES[model]
    cost = calculate_cost(model, prompt_tokens=1_000_000, completion_tokens=1_000_000)
    expected = price.input_per_1m + price.output_per_1m
    assert abs(cost - expected) < 1e-9


def test_calculate_cost_prompt_only():
    model = next(iter(MODEL_PRICES))
    price = MODEL_PRICES[model]
    cost = calculate_cost(model, prompt_tokens=1_000_000, completion_tokens=0)
    assert abs(cost - price.input_per_1m) < 1e-9


def test_calculate_cost_completion_only():
    model = next(iter(MODEL_PRICES))
    price = MODEL_PRICES[model]
    cost = calculate_cost(model, prompt_tokens=0, completion_tokens=1_000_000)
    assert abs(cost - price.output_per_1m) < 1e-9


def test_calculate_cost_negative_prompt_tokens_raises():
    model = next(iter(MODEL_PRICES))
    with pytest.raises(ValueError):
        calculate_cost(model, prompt_tokens=-1, completion_tokens=0)


def test_calculate_cost_negative_completion_tokens_raises():
    model = next(iter(MODEL_PRICES))
    with pytest.raises(ValueError):
        calculate_cost(model, prompt_tokens=0, completion_tokens=-1)


def test_cost_service_uses_custom_prices():
    custom = {"my-model": ModelPrice(input_per_1m=10.0, output_per_1m=20.0)}
    svc = CostService(prices=custom)
    cost = svc.calculate_cost("my-model", prompt_tokens=1_000_000, completion_tokens=1_000_000)
    assert abs(cost - 30.0) < 1e-9


def test_cost_service_unknown_model_returns_zero():
    svc = CostService()
    assert svc.calculate_cost("ghost-model", prompt_tokens=100, completion_tokens=100) == 0.0
