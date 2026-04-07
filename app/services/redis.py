from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

import redis.asyncio as redis


@dataclass(slots=True)
class RateLimitResult:
    allowed: bool
    limit: int
    remaining: int
    reset_ts: int


class RedisService:
    """Thin wrapper over `redis.asyncio` for gateway use-cases."""

    def __init__(self, client: redis.Redis) -> None:
        self._client = client

    @classmethod
    def from_url(cls, redis_url: str) -> RedisService:
        client = redis.from_url(redis_url, decode_responses=True)
        return cls(client)

    @property
    def client(self) -> redis.Redis:
        return self._client

    async def aclose(self) -> None:
        await self._client.aclose()

    async def check_and_increment_sliding_window(
        self,
        *,
        key: str,
        limit: int,
        window_seconds: int = 60,
        now: float | None = None,
    ) -> RateLimitResult:
        ts = float(now if now is not None else time.time())
        member = f"{ts:.6f}:{uuid.uuid4().hex}"
        ttl = window_seconds * 2 + 1
        min_score = ts - window_seconds

        # Step 1: clean up expired entries and read current count atomically.
        async with self._client.pipeline(transaction=False) as pipe:
            pipe.zremrangebyscore(key, "-inf", min_score)
            pipe.zcard(key)
            results = await pipe.execute()

        current_count = int(results[1])

        if current_count >= limit:
            oldest = await self._client.zrange(key, 0, 0, withscores=True)
            if oldest:
                reset_ts = int(float(oldest[0][1]) + window_seconds)
            else:
                reset_ts = int(ts + window_seconds)
            return RateLimitResult(
                allowed=False,
                limit=limit,
                remaining=0,
                reset_ts=reset_ts,
            )

        # Step 2: add the new entry.
        async with self._client.pipeline(transaction=False) as pipe:
            pipe.zadd(key, {member: ts})
            pipe.expire(key, ttl)
            await pipe.execute()

        return RateLimitResult(
            allowed=True,
            limit=limit,
            remaining=max(0, limit - current_count - 1),
            reset_ts=int(ts + window_seconds),
        )

    async def increment_usage_stats(
        self,
        *,
        api_key: str,
        prompt_tokens: int,
        completion_tokens: int,
        total_cost: float,
        now: datetime | None = None,
    ) -> None:
        ts = now or datetime.now(UTC)
        day = ts.strftime("%Y-%m-%d")
        month = ts.strftime("%Y-%m")
        daily_key = f"usage:{api_key}:daily:{day}"
        monthly_key = f"usage:{api_key}:monthly:{month}"

        total_tokens = prompt_tokens + completion_tokens
        daily_ttl = 7 * 24 * 60 * 60
        monthly_ttl = 90 * 24 * 60 * 60

        async with self._client.pipeline(transaction=True) as pipe:
            # Daily aggregate
            pipe.hincrby(daily_key, "prompt_tokens", int(prompt_tokens))
            pipe.hincrby(daily_key, "completion_tokens", int(completion_tokens))
            pipe.hincrby(daily_key, "total_tokens", int(total_tokens))
            pipe.hincrbyfloat(daily_key, "total_cost", float(total_cost))
            pipe.hincrby(daily_key, "request_count", 1)
            pipe.expire(daily_key, daily_ttl)

            # Monthly aggregate
            pipe.hincrby(monthly_key, "prompt_tokens", int(prompt_tokens))
            pipe.hincrby(monthly_key, "completion_tokens", int(completion_tokens))
            pipe.hincrby(monthly_key, "total_tokens", int(total_tokens))
            pipe.hincrbyfloat(monthly_key, "total_cost", float(total_cost))
            pipe.hincrby(monthly_key, "request_count", 1)
            pipe.expire(monthly_key, monthly_ttl)

            await pipe.execute()
