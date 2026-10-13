from __future__ import annotations

from datetime import UTC, datetime

import redis.asyncio as redis
from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from app.dependencies import get_client_id
from app.exceptions import ServiceUnavailableError

router = APIRouter(prefix="/me", tags=["usage"])


class UsageBucket(BaseModel):
    request_count: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_cost_usd: float = Field(default=0.0)


class UsageResponse(BaseModel):
    client_id: str
    today: UsageBucket
    current_month: UsageBucket


def _to_bucket(payload: dict[str, str]) -> UsageBucket:
    return UsageBucket(
        request_count=int(payload.get("request_count", 0) or 0),
        prompt_tokens=int(payload.get("prompt_tokens", 0) or 0),
        completion_tokens=int(payload.get("completion_tokens", 0) or 0),
        total_cost_usd=float(payload.get("total_cost", 0.0) or 0.0),
    )


@router.get("/usage", response_model=UsageResponse)
async def get_usage(
    request: Request,
    client_id: str = Depends(get_client_id),
) -> UsageResponse:
    now = datetime.now(UTC)
    day = now.strftime("%Y-%m-%d")
    month = now.strftime("%Y-%m")
    daily_key = f"usage:{client_id}:daily:{day}"
    monthly_key = f"usage:{client_id}:monthly:{month}"

    redis_service = request.app.state.redis_service
    try:
        daily_raw = await redis_service.client.hgetall(daily_key)
        monthly_raw = await redis_service.client.hgetall(monthly_key)
    except redis.RedisError as exc:
        raise ServiceUnavailableError(
            "Usage storage unavailable",
            code="usage_storage_unavailable",
        ) from exc

    return UsageResponse(
        client_id=client_id,
        today=_to_bucket(daily_raw),
        current_month=_to_bucket(monthly_raw),
    )
