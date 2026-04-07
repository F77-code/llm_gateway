from __future__ import annotations

import asyncio
import logging

import redis.asyncio as redis
from fastapi import APIRouter, Depends, HTTPException, status
from app.middleware.ratelimit import enforce_rate_limit
from app.models.chat_completion import ChatCompletionRequest, ChatCompletionResponse
from app.providers.base import (
    ProviderError,
    ProviderRateLimitError,
)
from app.providers.registry import UnknownModelError, get_provider
from app.services.cost import calculate_cost

router = APIRouter(prefix="/v1", tags=["chat"])
logger = logging.getLogger(__name__)


async def _persist_usage_stats(
    *,
    redis_service: object,
    api_key: str,
    prompt_tokens: int,
    completion_tokens: int,
    total_cost: float,
) -> None:
    try:
        await redis_service.increment_usage_stats(
            api_key=api_key,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_cost=total_cost,
        )
    except redis.RedisError as exc:
        logger.warning("Failed to persist usage stats to Redis: %s", exc)


@router.post(
    "/chat/completions",
    response_model=ChatCompletionResponse,
)
async def chat_completions(
    request: ChatCompletionRequest,
    api_key: str = Depends(enforce_rate_limit),
) -> ChatCompletionResponse:
    try:
        provider = get_provider(request.model)
    except UnknownModelError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown model: {exc.model}",
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Provider registry is not initialized",
        ) from exc

    try:
        response = await provider.chat_completion(request)
    except ProviderRateLimitError as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(exc),
        ) from exc
    except ProviderError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    usage = response.usage
    if usage is not None:
        total_cost = calculate_cost(
            request.model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
        )
        asyncio.create_task(
            _persist_usage_stats(
                redis_service=request.app.state.redis_service,
                api_key=api_key,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_cost=total_cost,
            ),
        )

    return response
