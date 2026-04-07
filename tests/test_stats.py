"""Integration tests for GET /stats/{api_key}."""

from __future__ import annotations

from app.services.redis import RedisService
from tests.conftest import AUTH_HEADERS, GATEWAY_KEY


async def test_stats_empty_returns_zeros(client):
    """Stats for a key with no activity returns all-zero buckets."""
    resp = await client.get(f"/stats/{GATEWAY_KEY}", headers=AUTH_HEADERS)

    assert resp.status_code == 200
    data = resp.json()
    assert data["today"]["request_count"] == 0
    assert data["today"]["prompt_tokens"] == 0
    assert data["today"]["completion_tokens"] == 0
    assert data["today"]["total_cost_usd"] == 0.0
    assert data["current_month"]["request_count"] == 0


async def test_stats_key_mismatch_returns_401(client):
    """Querying stats for a different key must return 401."""
    resp = await client.get("/stats/somebody-elses-key", headers=AUTH_HEADERS)

    assert resp.status_code == 401
    data = resp.json()
    assert data["error"]["code"] == "api_key_mismatch"
    assert data["error"]["type"] == "authentication_error"


async def test_stats_reflects_persisted_usage(client, fake_redis):
    """After usage is persisted, stats endpoint returns the correct counters."""
    svc = RedisService(fake_redis)
    await svc.increment_usage_stats(
        api_key=GATEWAY_KEY,
        prompt_tokens=150,
        completion_tokens=250,
        total_cost=0.02,
    )

    resp = await client.get(f"/stats/{GATEWAY_KEY}", headers=AUTH_HEADERS)

    assert resp.status_code == 200
    data = resp.json()
    assert data["today"]["request_count"] == 1
    assert data["today"]["prompt_tokens"] == 150
    assert data["today"]["completion_tokens"] == 250
    assert data["today"]["total_cost_usd"] == 0.02


async def test_stats_accumulates_multiple_requests(client, fake_redis):
    """Multiple persisted usages must sum correctly."""
    svc = RedisService(fake_redis)
    for _ in range(4):
        await svc.increment_usage_stats(
            api_key=GATEWAY_KEY,
            prompt_tokens=100,
            completion_tokens=100,
            total_cost=0.005,
        )

    resp = await client.get(f"/stats/{GATEWAY_KEY}", headers=AUTH_HEADERS)
    data = resp.json()

    assert data["today"]["request_count"] == 4
    assert data["today"]["prompt_tokens"] == 400
    assert data["current_month"]["completion_tokens"] == 400


async def test_stats_api_key_is_masked_in_response(client, fake_redis):
    """The api_key field in the response must be masked, not the full value."""
    resp = await client.get(f"/stats/{GATEWAY_KEY}", headers=AUTH_HEADERS)

    data = resp.json()
    assert GATEWAY_KEY not in data["api_key"], "Full API key must not appear in response"
    assert data["api_key"].endswith(GATEWAY_KEY[-4:])
