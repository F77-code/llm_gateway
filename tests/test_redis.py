"""Unit tests for RedisService with fakeredis (no real Redis needed)."""

from __future__ import annotations

import time
from datetime import UTC, datetime

import fakeredis.aioredis
import pytest
import pytest_asyncio

from app.services.redis import RateLimitResult, RedisService


@pytest_asyncio.fixture
async def svc():
    r = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield RedisService(r)
    await r.aclose()


# ---------------------------------------------------------------------------
# Sliding-window rate limit
# ---------------------------------------------------------------------------


async def test_rate_limit_first_request_is_allowed(svc):
    result = await svc.check_and_increment_sliding_window(key="rl:k1", limit=5, window_seconds=60)
    assert result.allowed is True
    assert result.limit == 5
    assert result.remaining == 4


async def test_rate_limit_remaining_decrements(svc):
    for i in range(3):
        r = await svc.check_and_increment_sliding_window(key="rl:k2", limit=5, window_seconds=60)
        assert r.remaining == 4 - i


async def test_rate_limit_blocks_at_limit(svc):
    for _ in range(5):
        await svc.check_and_increment_sliding_window(key="rl:k3", limit=5, window_seconds=60)

    result = await svc.check_and_increment_sliding_window(key="rl:k3", limit=5, window_seconds=60)

    assert result.allowed is False
    assert result.remaining == 0


async def test_rate_limit_window_expiry_allows_new_requests(svc):
    now = time.time()

    # Fill the window in the past (outside current 60-second window)
    for _ in range(5):
        await svc.check_and_increment_sliding_window(
            key="rl:k4", limit=5, window_seconds=60, now=now - 120
        )

    # New request is within the window → old entries cleaned up → allowed
    result = await svc.check_and_increment_sliding_window(
        key="rl:k4", limit=5, window_seconds=60, now=now
    )
    assert result.allowed is True


async def test_rate_limit_reset_ts_is_in_future(svc):
    result = await svc.check_and_increment_sliding_window(key="rl:k5", limit=3, window_seconds=60)
    assert result.reset_ts > int(time.time())


async def test_rate_limit_denied_reset_ts_is_in_future(svc):
    now = time.time()
    for _ in range(3):
        await svc.check_and_increment_sliding_window(
            key="rl:k6", limit=3, window_seconds=60, now=now
        )
    result = await svc.check_and_increment_sliding_window(
        key="rl:k6", limit=3, window_seconds=60, now=now
    )
    assert result.allowed is False
    assert result.reset_ts > int(now)


# ---------------------------------------------------------------------------
# Usage stats
# ---------------------------------------------------------------------------


async def test_usage_stats_incremented(svc):
    await svc.increment_usage_stats(
        api_key="key-a",
        prompt_tokens=100,
        completion_tokens=200,
        total_cost=0.01,
    )

    now = datetime.now(UTC)
    daily_key = f"usage:key-a:daily:{now.strftime('%Y-%m-%d')}"
    monthly_key = f"usage:key-a:monthly:{now.strftime('%Y-%m')}"

    daily = await svc.client.hgetall(daily_key)
    assert int(daily["prompt_tokens"]) == 100
    assert int(daily["completion_tokens"]) == 200
    assert int(daily["total_tokens"]) == 300
    assert int(daily["request_count"]) == 1
    assert float(daily["total_cost"]) == pytest.approx(0.01)

    monthly = await svc.client.hgetall(monthly_key)
    assert int(monthly["request_count"]) == 1


async def test_usage_stats_accumulate_across_calls(svc):
    for _ in range(3):
        await svc.increment_usage_stats(
            api_key="key-b",
            prompt_tokens=50,
            completion_tokens=100,
            total_cost=0.005,
        )

    now = datetime.now(UTC)
    daily_key = f"usage:key-b:daily:{now.strftime('%Y-%m-%d')}"
    daily = await svc.client.hgetall(daily_key)

    assert int(daily["request_count"]) == 3
    assert int(daily["prompt_tokens"]) == 150
    assert int(daily["completion_tokens"]) == 300
    assert float(daily["total_cost"]) == pytest.approx(0.015)


async def test_usage_stats_different_keys_are_independent(svc):
    await svc.increment_usage_stats(api_key="key-x", prompt_tokens=10, completion_tokens=20, total_cost=0.001)
    await svc.increment_usage_stats(api_key="key-y", prompt_tokens=30, completion_tokens=40, total_cost=0.002)

    now = datetime.now(UTC)
    day = now.strftime("%Y-%m-%d")

    x = await svc.client.hgetall(f"usage:key-x:daily:{day}")
    y = await svc.client.hgetall(f"usage:key-y:daily:{day}")

    assert int(x["prompt_tokens"]) == 10
    assert int(y["prompt_tokens"]) == 30
