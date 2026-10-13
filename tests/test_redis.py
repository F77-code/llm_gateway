"""Unit tests for RedisService with fakeredis (no real Redis needed)."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import UUID

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
    result = await svc.check_and_increment_sliding_window(
        key="rl:k1", limit=5, window_seconds=60
    )
    assert result.allowed is True
    assert result.limit == 5
    assert result.remaining == 4


async def test_rate_limit_remaining_decrements(svc):
    for i in range(3):
        r = await svc.check_and_increment_sliding_window(
            key="rl:k2", limit=5, window_seconds=60
        )
        assert r.remaining == 4 - i


async def test_rate_limit_blocks_at_limit(svc):
    for _ in range(5):
        await svc.check_and_increment_sliding_window(
            key="rl:k3", limit=5, window_seconds=60
        )

    result = await svc.check_and_increment_sliding_window(
        key="rl:k3", limit=5, window_seconds=60
    )

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
    result = await svc.check_and_increment_sliding_window(
        key="rl:k5", limit=3, window_seconds=60
    )
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


@pytest.mark.parametrize(("now", "reset_ts"), [(100.0, 160), (100.25, 161)])
async def test_rate_limit_rounds_reset_up_for_accepted_and_denied_calls(
    svc, now, reset_ts
):
    first = await svc.check_and_increment_sliding_window(
        key="rl:rounded-reset", limit=1, now=now
    )
    denied = await svc.check_and_increment_sliding_window(
        key="rl:rounded-reset", limit=1, now=now + 1
    )

    assert first.reset_ts == reset_ts
    assert denied.allowed is False
    assert denied.reset_ts == reset_ts


async def test_rate_limit_accepted_reset_tracks_oldest_active_admission(svc):
    first = await svc.check_and_increment_sliding_window(
        key="rl:oldest-reset", limit=3, now=100.25
    )
    second = await svc.check_and_increment_sliding_window(
        key="rl:oldest-reset", limit=3, now=110.75
    )
    after_first_expires = await svc.check_and_increment_sliding_window(
        key="rl:oldest-reset", limit=3, now=160.25
    )

    assert first.remaining == 2
    assert second.remaining == 1
    assert second.reset_ts == first.reset_ts == 161
    assert after_first_expires.allowed is True
    assert after_first_expires.remaining == 1
    assert after_first_expires.reset_ts == 171


@pytest.mark.parametrize("start", [100.0, 100.25])
async def test_rate_limit_left_window_boundary_is_exclusive(svc, start):
    await svc.check_and_increment_sliding_window(key="rl:boundary", limit=1, now=start)
    before_boundary = await svc.check_and_increment_sliding_window(
        key="rl:boundary", limit=1, now=start + 60 - 0.001
    )
    at_boundary = await svc.check_and_increment_sliding_window(
        key="rl:boundary", limit=1, now=start + 60
    )

    assert before_boundary.allowed is False
    assert at_boundary.allowed is True
    assert at_boundary.remaining == 0
    assert await svc.client.zcard("rl:boundary") == 1


async def test_rate_limit_identical_timestamps_have_independent_uuid_members(svc):
    results = [
        await svc.check_and_increment_sliding_window(
            key="rl:identical-scores", limit=3, now=100.125
        )
        for _ in range(4)
    ]
    entries = await svc.client.zrange("rl:identical-scores", 0, -1, withscores=True)

    assert [result.allowed for result in results] == [True, True, True, False]
    assert [result.remaining for result in results] == [2, 1, 0, 0]
    assert len(entries) == 3
    assert len({UUID(hex=member) for member, _ in entries}) == 3
    assert all(score == 100.125 for _, score in entries)


async def test_rate_limit_admission_sets_ttl_and_rejection_does_not_refresh_it(svc):
    await svc.check_and_increment_sliding_window(
        key="rl:ttl", limit=1, window_seconds=60, now=100
    )
    assert 120 <= await svc.client.ttl("rl:ttl") <= 121
    await svc.client.expire("rl:ttl", 17)

    result = await svc.check_and_increment_sliding_window(
        key="rl:ttl", limit=1, window_seconds=60, now=101
    )

    assert result.allowed is False
    assert 0 < await svc.client.ttl("rl:ttl") <= 17
    assert await svc.client.zcard("rl:ttl") == 1


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("limit", 0),
        ("limit", -1),
        ("limit", 1.5),
        ("limit", True),
        ("limit", "5"),
        ("limit", None),
        ("limit", 2**53),
        ("window_seconds", 0),
        ("window_seconds", -1),
        ("window_seconds", 1.5),
        ("window_seconds", True),
        ("window_seconds", "60"),
        ("window_seconds", None),
        ("window_seconds", 2**31),
        ("window_seconds", 2**62),
        ("now", float("nan")),
        ("now", float("inf")),
        ("now", float("-inf")),
        ("now", "not-a-timestamp"),
        ("now", 10**400),
    ],
)
async def test_rate_limit_invalid_arguments_fail_before_redis_mutation(
    svc, monkeypatch, field, value
):
    await svc.client.zadd("rl:invalid", {"existing-admission": 1})
    eval_spy = AsyncMock(wraps=svc.client.eval)
    monkeypatch.setattr(svc.client, "eval", eval_spy)
    arguments = {"limit": 5, "window_seconds": 60, "now": 100.0, field: value}

    with pytest.raises(ValueError, match="positive integer|finite timestamp"):
        await svc.check_and_increment_sliding_window(key="rl:invalid", **arguments)

    eval_spy.assert_not_awaited()
    assert await svc.client.zrange("rl:invalid", 0, -1, withscores=True) == [
        ("existing-admission", 1.0)
    ]


async def test_rate_limit_accepts_largest_supported_integer_arguments(svc):
    result = await svc.check_and_increment_sliding_window(
        key="rl:maximum-arguments", limit=2**53 - 1, window_seconds=2**31 - 1, now=0
    )

    assert result == RateLimitResult(True, 2**53 - 1, 2**53 - 2, 2**31 - 1)
    assert await svc.client.zcard("rl:maximum-arguments") == 1


# ---------------------------------------------------------------------------
# Usage stats
# ---------------------------------------------------------------------------


async def test_usage_stats_incremented(svc):
    await svc.increment_usage_stats(
        client_id="client-a",
        prompt_tokens=100,
        completion_tokens=200,
        total_cost=0.01,
    )

    now = datetime.now(UTC)
    daily_key = f"usage:client-a:daily:{now.strftime('%Y-%m-%d')}"
    monthly_key = f"usage:client-a:monthly:{now.strftime('%Y-%m')}"

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
            client_id="client-b",
            prompt_tokens=50,
            completion_tokens=100,
            total_cost=0.005,
        )

    now = datetime.now(UTC)
    daily_key = f"usage:client-b:daily:{now.strftime('%Y-%m-%d')}"
    daily = await svc.client.hgetall(daily_key)

    assert int(daily["request_count"]) == 3
    assert int(daily["prompt_tokens"]) == 150
    assert int(daily["completion_tokens"]) == 300
    assert float(daily["total_cost"]) == pytest.approx(0.015)


async def test_usage_stats_different_keys_are_independent(svc):
    await svc.increment_usage_stats(
        client_id="client-x", prompt_tokens=10, completion_tokens=20, total_cost=0.001
    )
    await svc.increment_usage_stats(
        client_id="client-y", prompt_tokens=30, completion_tokens=40, total_cost=0.002
    )

    now = datetime.now(UTC)
    day = now.strftime("%Y-%m-%d")

    x = await svc.client.hgetall(f"usage:client-x:daily:{day}")
    y = await svc.client.hgetall(f"usage:client-y:daily:{day}")

    assert int(x["prompt_tokens"]) == 10
    assert int(y["prompt_tokens"]) == 30
