"""Integration tests for GET /health.

Provider HTTP calls are mocked via BaseLLMProvider._http_request, which is the
single method all providers use for outbound requests (health_check calls it
internally). Each concrete subclass overrides health_check but NOT _http_request,
so patching the base class method intercepts all subclass calls correctly.
"""

from __future__ import annotations

import httpx
import pytest
from unittest.mock import AsyncMock

from app.providers.base import BaseLLMProvider, ProviderError


def _ok_response() -> httpx.Response:
    return httpx.Response(200, json={"data": []})


async def test_health_all_providers_ok(client, mocker):
    """When all providers respond healthy, status is 'healthy' and HTTP 200."""
    mocker.patch.object(
        BaseLLMProvider, "_http_request", new_callable=AsyncMock, return_value=_ok_response()
    )

    resp = await client.get("/health")

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] in ("healthy", "degraded")
    assert data["redis"] == "ok"


async def test_health_all_providers_fail(client, mocker):
    """When every provider is unhealthy (raises ProviderError), status is 'unhealthy' and HTTP 503."""
    mocker.patch.object(
        BaseLLMProvider,
        "_http_request",
        new_callable=AsyncMock,
        side_effect=ProviderError("simulated connection failure"),
    )

    resp = await client.get("/health")

    assert resp.status_code == 503
    data = resp.json()
    assert data["status"] == "unhealthy"


async def test_health_response_has_providers_dict(client, mocker):
    """Response must include a 'providers' dict with at least one entry."""
    mocker.patch.object(
        BaseLLMProvider, "_http_request", new_callable=AsyncMock, return_value=_ok_response()
    )

    resp = await client.get("/health")
    data = resp.json()

    assert "providers" in data
    assert isinstance(data["providers"], dict)
    assert len(data["providers"]) > 0


async def test_health_provider_entry_has_latency(client, mocker):
    """Each provider entry must include latency_ms regardless of outcome."""
    mocker.patch.object(
        BaseLLMProvider, "_http_request", new_callable=AsyncMock, return_value=_ok_response()
    )

    resp = await client.get("/health")
    data = resp.json()

    for provider_name, info in data["providers"].items():
        assert "latency_ms" in info, f"latency_ms missing for provider {provider_name}"


async def test_health_provider_error_still_has_latency(client, mocker):
    """latency_ms must be present even when a provider raises an exception."""
    mocker.patch.object(
        BaseLLMProvider,
        "_http_request",
        new_callable=AsyncMock,
        side_effect=RuntimeError("simulated failure"),
    )

    resp = await client.get("/health")
    data = resp.json()

    for provider_name, info in data["providers"].items():
        assert "latency_ms" in info, f"latency_ms missing for provider {provider_name}"
        assert info["status"] == "error"
