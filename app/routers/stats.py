from __future__ import annotations

from datetime import UTC, datetime

import redis.asyncio as redis
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.dependencies import get_api_key

router = APIRouter(prefix="/stats", tags=["stats"])


class UsageBucket(BaseModel):
    request_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_cost_usd: float = Field(default=0.0)


class StatsResponse(BaseModel):
    api_key: str
    today: UsageBucket
    current_month: UsageBucket


def _mask_api_key(api_key: str) -> str:
    if len(api_key) <= 4:
        return "****" + api_key
    return f"***masked***{api_key[-4:]}"


def _to_bucket(payload: dict[str, str]) -> UsageBucket:
    return UsageBucket(
        request_count=int(payload.get("request_count", 0) or 0),
        prompt_tokens=int(payload.get("prompt_tokens", 0) or 0),
        completion_tokens=int(payload.get("completion_tokens", 0) or 0),
        total_cost_usd=float(payload.get("total_cost", 0.0) or 0.0),
    )


@router.get("/{api_key}", response_model=StatsResponse)
async def get_stats(
    api_key: str,
    request: Request,
    requester_api_key: str = Depends(get_api_key),
) -> StatsResponse:
    if requester_api_key != api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization key must match requested api_key",
        )

    now = datetime.now(UTC)
    day = now.strftime("%Y-%m-%d")
    month = now.strftime("%Y-%m")
    daily_key = f"usage:{api_key}:daily:{day}"
    monthly_key = f"usage:{api_key}:monthly:{month}"

    redis_service = request.app.state.redis_service
    try:
        daily_raw = await redis_service.client.hgetall(daily_key)
        monthly_raw = await redis_service.client.hgetall(monthly_key)
    except redis.RedisError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Stats storage unavailable: {exc}",
        ) from exc

    return StatsResponse(
        api_key=_mask_api_key(api_key),
        today=_to_bucket(daily_raw),
        current_month=_to_bucket(monthly_raw),
    )
