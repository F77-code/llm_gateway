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
        max_retries: int = 5,
    ) -> RateLimitResult:
        ts = float(now if now is not None else time.time())
        member = f"{ts:.6f}:{uuid.uuid4().hex}"
        ttl = max(window_seconds * 2, window_seconds + 1)
        min_score = ts - window_seconds

        for _ in range(max_retries):
            async with self._client.pipeline(transaction=True) as pipe:
                try:
                    await pipe.watch(key)
                    await pipe.zremrangebyscore(key, "-inf", min_score)
                    current_count = int(await pipe.zcard(key))

                    if current_count >= limit:
                        oldest = await pipe.zrange(key, 0, 0, withscores=True)
                        await pipe.reset()
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

                    pipe.multi()
                    pipe.zadd(key, {member: ts})
                    pipe.expire(key, ttl)
                    await pipe.execute()
                    return RateLimitResult(
                        allowed=True,
                        limit=limit,
                        remaining=max(0, limit - (current_count + 1)),
                        reset_ts=int(ts + window_seconds),
                    )
                except redis.WatchError:
                    continue

        # Under high contention retries can exhaust; treat as unavailable.
        raise redis.RedisError("Rate-limit transaction retries exhausted")

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
