from __future__ import annotations

import time
import logging

from fastapi import Depends, HTTPException, Request, Response, status
import redis.asyncio as redis

from app.config import get_settings
from app.dependencies import get_api_key

WINDOW_SECONDS = 60
logger = logging.getLogger(__name__)


def _headers(limit: int, remaining: int, reset_ts: int) -> dict[str, str]:
    return {
        "X-RateLimit-Limit": str(limit),
        "X-RateLimit-Remaining": str(max(0, remaining)),
        "X-RateLimit-Reset": str(reset_ts),
    }


async def enforce_rate_limit(
    request: Request,
    response: Response,
    api_key: str = Depends(get_api_key),
) -> str:
    settings = get_settings()
    limit = settings.rate_limit_rpm
    redis_service = request.app.state.redis_service
    key = f"ratelimit:{api_key}"
    try:
        result = await redis_service.check_and_increment_sliding_window(
            key=key,
            limit=limit,
            window_seconds=WINDOW_SECONDS,
            now=time.time(),
        )
    except redis.RedisError as exc:
        # Fail-open: Redis issues should not block traffic.
        logger.warning("Redis unavailable for rate limiting, allowing request: %s", exc)
        return api_key

    if not result.allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Rate limit exceeded",
            headers=_headers(result.limit, result.remaining, result.reset_ts),
        )

    for h, v in _headers(result.limit, result.remaining, result.reset_ts).items():
        response.headers[h] = v
    return api_key
