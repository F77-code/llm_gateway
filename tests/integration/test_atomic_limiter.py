"""Real Redis contracts and deterministic regression for the old two-phase race."""

from __future__ import annotations

import asyncio
import multiprocessing
import queue
import time
import traceback
from contextlib import ExitStack
from typing import TYPE_CHECKING, Any
from unittest.mock import patch

import pytest
import redis.asyncio as redis

from app.services.redis import RedisService

if TYPE_CHECKING:
    from multiprocessing.queues import Queue
    from multiprocessing.synchronize import Barrier

    from redis.asyncio.client import Pipeline

    from tests.integration.conftest import RedisTestScope


pytestmark = pytest.mark.redis_integration


class _ReadPhasePipeline:
    """Hold a legacy read result until all callers have read the same capacity.

    Only the original cleanup + ZCARD pipeline waits. An atomic EVAL/EVALSHA
    never enters this barrier. Reads and writes still execute on real Redis.
    """

    def __init__(self, pipeline: Pipeline, barrier: asyncio.Barrier) -> None:
        self._pipeline = pipeline
        self._barrier = barrier
        self._has_cleanup = False
        self._has_count = False

    async def __aenter__(self) -> _ReadPhasePipeline:
        await self._pipeline.__aenter__()
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self._pipeline.__aexit__(*args)

    def zremrangebyscore(self, *args: Any, **kwargs: Any) -> _ReadPhasePipeline:
        self._has_cleanup = True
        self._pipeline.zremrangebyscore(*args, **kwargs)
        return self

    def zcard(self, *args: Any, **kwargs: Any) -> _ReadPhasePipeline:
        self._has_count = True
        self._pipeline.zcard(*args, **kwargs)
        return self

    def __getattr__(self, name: str) -> Any:
        return getattr(self._pipeline, name)

    async def execute(self) -> list[Any]:
        results = await self._pipeline.execute()
        if self._has_cleanup and self._has_count:
            async with asyncio.timeout(10):
                await self._barrier.wait()
        return results


async def test_concurrent_clients_never_exceed_quota(
    redis_scope: RedisTestScope,
) -> None:
    """Same invariant passes Lua and fails the git-show original implementation."""
    key = redis_scope.key("independent-clients")
    quota, callers, now = 7, 32, 1_000.125
    barrier = asyncio.Barrier(callers)
    clients = [
        redis.from_url(
            redis_scope.url,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        for _ in range(callers)
    ]

    def pipeline_factory(client: redis.Redis) -> Any:
        original = client.pipeline

        def factory(*args: Any, **kwargs: Any) -> _ReadPhasePipeline:
            return _ReadPhasePipeline(original(*args, **kwargs), barrier)

        return factory

    try:
        # Each service has its own connection pool. Installing the old-read
        # wrapper on the clients makes the original race reproducible without
        # retaining a broken rate-limiter implementation in the test suite.
        await asyncio.gather(*(client.ping() for client in clients))
        with ExitStack() as stack:
            for client in clients:
                stack.enter_context(
                    patch.object(client, "pipeline", pipeline_factory(client))
                )
            async with asyncio.timeout(15):
                decisions = await asyncio.gather(
                    *(
                        RedisService(client).check_and_increment_sliding_window(
                            key=key, limit=quota, window_seconds=60, now=now
                        )
                        for client in clients
                    )
                )
        admitted = sum(result.allowed for result in decisions)
        cardinality = await redis_scope.client.zcard(key)
        assert (admitted, cardinality) == (quota, quota)
        assert sorted(
            result.remaining for result in decisions if result.allowed
        ) == list(range(quota))
        assert all(result.remaining == 0 for result in decisions if not result.allowed)
        members = await redis_scope.client.zrange(key, 0, -1, withscores=True)
        assert len({member for member, _ in members}) == quota
        assert all(score == now for _, score in members)
        assert all(result.reset_ts == 1_061 for result in decisions)
    finally:
        await asyncio.gather(*(client.aclose() for client in clients))


def _process_worker(
    url: str,
    key: str,
    quota: int,
    now: float,
    attempts: int,
    barrier: Barrier,
    output: Queue,
) -> None:
    """Spawned processes construct their own clients and event loops."""

    async def run() -> list[tuple[bool, int, int]]:
        client = redis.from_url(
            url,
            decode_responses=True,
            socket_connect_timeout=2,
            socket_timeout=2,
        )
        try:
            await client.ping()
            await asyncio.to_thread(barrier.wait, 15)
            service = RedisService(client)
            async with asyncio.timeout(15):
                results = await asyncio.gather(
                    *(
                        service.check_and_increment_sliding_window(
                            key=key, limit=quota, window_seconds=60, now=now
                        )
                        for _ in range(attempts)
                    )
                )
            return [(r.allowed, r.remaining, r.reset_ts) for r in results]
        finally:
            await client.aclose()

    try:
        output.put(("ok", asyncio.run(run())))
    except BaseException:
        output.put(("error", traceback.format_exc()))
        raise


def _collect_process_results(processes: list[Any], output: Queue) -> list[Any]:
    deadline = time.monotonic() + 30
    replies = []
    for _ in processes:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise AssertionError("Spawned limiter workers did not finish within 30s")
        try:
            replies.append(output.get(timeout=remaining))
        except queue.Empty as exc:
            raise AssertionError(
                "Spawned limiter worker did not report a result"
            ) from exc
    for process in processes:
        process.join(timeout=max(0, deadline - time.monotonic()))
        assert not process.is_alive(), "Spawned limiter worker did not exit"
        assert process.exitcode == 0, f"Limiter worker failed: {replies!r}"
    errors = [payload for status, payload in replies if status != "ok"]
    assert not errors, "\n".join(errors)
    return [decision for _, payload in replies for decision in payload]


async def test_spawned_processes_never_exceed_quota(
    redis_scope: RedisTestScope,
) -> None:
    key = redis_scope.key("spawned-processes")
    quota, workers, attempts, now = 11, 6, 8, 2_000.875
    context = multiprocessing.get_context("spawn")
    barrier = context.Barrier(workers)
    output = context.Queue()
    processes = [
        context.Process(
            target=_process_worker,
            args=(redis_scope.url, key, quota, now, attempts, barrier, output),
        )
        for _ in range(workers)
    ]
    started = []
    try:
        for process in processes:
            process.start()
            started.append(process)
        decisions = await asyncio.to_thread(_collect_process_results, started, output)
        admitted = sum(allowed for allowed, _, _ in decisions)
        cardinality = await redis_scope.client.zcard(key)
        assert (admitted, cardinality) == (quota, quota)
        assert len(decisions) == workers * attempts
        members = await redis_scope.client.zrange(key, 0, -1, withscores=True)
        assert len({member for member, _ in members}) == quota
        assert all(score == now for _, score in members)
        assert sorted(
            remaining for allowed, remaining, _ in decisions if allowed
        ) == list(range(quota))
        assert all(remaining == 0 for allowed, remaining, _ in decisions if not allowed)
        assert all(reset_ts == 2_061 for _, _, reset_ts in decisions)
    finally:
        for process in started:
            if process.is_alive():
                process.terminate()
        for process in started:
            await asyncio.to_thread(process.join, 5)
            if process.is_alive():
                process.kill()
                await asyncio.to_thread(process.join, 5)
            process.close()
        output.close()
        output.join_thread()


@pytest.mark.parametrize("start", [100.0, 100.25])
async def test_boundary_expires_oldest_and_reset_uses_next_slot(
    redis_scope: RedisTestScope, start: float
) -> None:
    service = RedisService(redis_scope.client)
    key = redis_scope.key("boundary")
    first = await service.check_and_increment_sliding_window(
        key=key, limit=2, window_seconds=10, now=start
    )
    second = await service.check_and_increment_sliding_window(
        key=key, limit=2, window_seconds=10, now=start + 4
    )
    expected_reset = 110 if start == 100.0 else 111
    assert (first.allowed, first.remaining, first.reset_ts) == (True, 1, expected_reset)
    assert (second.allowed, second.remaining, second.reset_ts) == (
        True,
        0,
        expected_reset,
    )

    denied = await service.check_and_increment_sliding_window(
        key=key, limit=2, window_seconds=10, now=start + 10 - 0.000_001
    )
    assert (denied.allowed, denied.remaining, denied.reset_ts) == (
        False,
        0,
        expected_reset,
    )
    assert await redis_scope.client.zcard(key) == 2

    boundary = await service.check_and_increment_sliding_window(
        key=key, limit=2, window_seconds=10, now=start + 10
    )
    next_reset = 114 if start == 100.0 else 115
    assert (boundary.allowed, boundary.remaining, boundary.reset_ts) == (
        True,
        0,
        next_reset,
    )
    members = await redis_scope.client.zrange(key, 0, -1, withscores=True)
    assert [score for _, score in members] == [start + 4, start + 10]

    empty = await service.check_and_increment_sliding_window(
        key=key, limit=2, window_seconds=10, now=start + 20
    )
    fresh_reset = 130 if start == 100.0 else 131
    assert (empty.allowed, empty.remaining, empty.reset_ts) == (True, 1, fresh_reset)
    assert await redis_scope.client.zcard(key) == 1


async def test_independent_keys_have_independent_quota(
    redis_scope: RedisTestScope,
) -> None:
    service = RedisService(redis_scope.client)
    first_key = redis_scope.key("policy-a")
    second_key = redis_scope.key("policy-b")
    first = await service.check_and_increment_sliding_window(
        key=first_key, limit=1, now=100
    )
    denied = await service.check_and_increment_sliding_window(
        key=first_key, limit=1, now=100
    )
    second = await service.check_and_increment_sliding_window(
        key=second_key, limit=1, now=100
    )
    assert (first.allowed, denied.allowed, second.allowed) == (True, False, True)
    assert await redis_scope.client.zcard(first_key) == 1
    assert await redis_scope.client.zcard(second_key) == 1


async def test_admission_sets_ttl_and_rejection_does_not_extend_it(
    redis_scope: RedisTestScope,
) -> None:
    service = RedisService(redis_scope.client)
    key = redis_scope.key("ttl-retention")
    await service.check_and_increment_sliding_window(
        key=key, limit=1, window_seconds=10, now=100
    )
    assert 20_000 <= await redis_scope.client.pttl(key) <= 21_000
    assert await redis_scope.client.pexpire(key, 3_000)
    before = await redis_scope.client.pttl(key)
    rejected = await service.check_and_increment_sliding_window(
        key=key, limit=1, window_seconds=10, now=100
    )
    after = await redis_scope.client.pttl(key)
    assert rejected.allowed is False
    assert 0 < after <= before
    assert await redis_scope.client.zcard(key) == 1


async def test_key_expires_after_admission_ttl(redis_scope: RedisTestScope) -> None:
    service = RedisService(redis_scope.client)
    key = redis_scope.key("ttl-expiry")
    await service.check_and_increment_sliding_window(
        key=key, limit=1, window_seconds=1, now=100
    )
    assert 2_000 <= await redis_scope.client.pttl(key) <= 3_000
    async with asyncio.timeout(5):
        # Redis expiry is external and provides no local Event to await.
        while await redis_scope.client.exists(key):  # noqa: ASYNC110
            await asyncio.sleep(0.05)
    assert await redis_scope.client.pttl(key) == -2
    result = await service.check_and_increment_sliding_window(
        key=key, limit=1, window_seconds=1, now=101
    )
    assert (result.allowed, result.remaining, result.reset_ts) == (True, 0, 102)
    assert await redis_scope.client.zcard(key) == 1


@pytest.mark.parametrize(
    "arguments",
    [
        {"limit": 0, "window_seconds": 10, "now": 100.0},
        {"limit": 1, "window_seconds": 0, "now": 100.0},
        {"limit": 1, "window_seconds": 10, "now": float("nan")},
        {"limit": 1, "window_seconds": 10, "now": float("inf")},
    ],
)
async def test_invalid_arguments_preserve_existing_redis_state(
    redis_scope: RedisTestScope, arguments: dict[str, Any]
) -> None:
    key = redis_scope.key("invalid-input")
    await redis_scope.client.zadd(key, {"expired-but-owned": 80, "active": 95})
    await redis_scope.client.pexpire(key, 3_000)
    before = await redis_scope.client.zrange(key, 0, -1, withscores=True)
    ttl_before = await redis_scope.client.pttl(key)
    with pytest.raises((ValueError, TypeError)):
        await RedisService(redis_scope.client).check_and_increment_sliding_window(
            key=key, **arguments
        )
    assert await redis_scope.client.zrange(key, 0, -1, withscores=True) == before
    assert 0 < await redis_scope.client.pttl(key) <= ttl_before


async def test_wrong_redis_key_type_surfaces_error_and_preserves_value(
    redis_scope: RedisTestScope,
) -> None:
    key = redis_scope.key("wrong-type")
    await redis_scope.client.set(key, "owned-sentinel", px=3_000)
    ttl_before = await redis_scope.client.pttl(key)
    with pytest.raises(redis.ResponseError, match="WRONGTYPE"):
        await RedisService(redis_scope.client).check_and_increment_sliding_window(
            key=key, limit=1, now=100
        )
    assert await redis_scope.client.get(key) == "owned-sentinel"
    assert 0 < await redis_scope.client.pttl(key) <= ttl_before
