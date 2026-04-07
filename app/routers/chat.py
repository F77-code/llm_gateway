from __future__ import annotations

import asyncio
import logging
import time

import redis.asyncio as redis
from fastapi import APIRouter, Depends, HTTPException, Request, status
from app.config import Settings
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


def _mask_api_key(api_key: str) -> str:
    if len(api_key) <= 4:
        return "****" + api_key
    return f"***masked***{api_key[-4:]}"


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
    body: ChatCompletionRequest,
    http_request: Request,
    api_key: str = Depends(enforce_rate_limit),
) -> ChatCompletionResponse:
    started = time.perf_counter()
    request_id = getattr(http_request.state, "request_id", None)
    provider_name = (
        Settings.provider_by_model.get(body.model).value
        if Settings.provider_by_model.get(body.model) is not None
        else "unknown"
    )
    usage_payload: dict[str, int] = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    total_cost = 0.0

    try:
        provider = get_provider(request.model)
    except UnknownModelError as exc:
        logger.warning(
            "chat.request",
            extra={
                "payload": {
                    "request_id": request_id,
                    "api_key": _mask_api_key(api_key),
                    "model": body.model,
                    "provider": provider_name,
                    "latency_ms": int((time.perf_counter() - started) * 1000),
                    "tokens": usage_payload,
                    "cost": total_cost,
                    "status": "error",
                    "error": f"Unknown model: {exc.model}",
                },
            },
        )
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown model: {exc.model}",
        ) from exc
    except RuntimeError as exc:
        logger.warning(
            "chat.request",
            extra={
                "payload": {
                    "request_id": request_id,
                    "api_key": _mask_api_key(api_key),
                    "model": body.model,
                    "provider": provider_name,
                    "latency_ms": int((time.perf_counter() - started) * 1000),
                    "tokens": usage_payload,
                    "cost": total_cost,
                    "status": "error",
                    "error": "Provider registry is not initialized",
                },
            },
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Provider registry is not initialized",
        ) from exc

    try:
        response = await provider.chat_completion(body)
    except ProviderRateLimitError as exc:
        logger.warning(
            "chat.request",
            extra={
                "payload": {
                    "request_id": request_id,
                    "api_key": _mask_api_key(api_key),
                    "model": body.model,
                    "provider": provider_name,
                    "latency_ms": int((time.perf_counter() - started) * 1000),
                    "tokens": usage_payload,
                    "cost": total_cost,
                    "status": "error",
                    "error": str(exc),
                },
            },
        )
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(exc),
        ) from exc
    except ProviderError as exc:
        logger.warning(
            "chat.request",
            extra={
                "payload": {
                    "request_id": request_id,
                    "api_key": _mask_api_key(api_key),
                    "model": body.model,
                    "provider": provider_name,
                    "latency_ms": int((time.perf_counter() - started) * 1000),
                    "tokens": usage_payload,
                    "cost": total_cost,
                    "status": "error",
                    "error": str(exc),
                },
            },
        )
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc

    usage = response.usage
    if usage is not None:
        usage_payload = {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        }
        total_cost = calculate_cost(
            body.model,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
        )
        asyncio.create_task(
            _persist_usage_stats(
                redis_service=http_request.app.state.redis_service,
                api_key=api_key,
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                total_cost=total_cost,
            ),
        )

    logger.info(
        "chat.request",
        extra={
            "payload": {
                "request_id": request_id,
                "api_key": _mask_api_key(api_key),
                "model": body.model,
                "provider": provider_name,
                "latency_ms": int((time.perf_counter() - started) * 1000),
                "tokens": usage_payload,
                "cost": total_cost,
                "status": "ok",
            },
        },
    )
    return response
