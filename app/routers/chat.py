from __future__ import annotations

import logging
import time

import redis.asyncio as redis
from fastapi import APIRouter, BackgroundTasks, Depends, Request
from app.config import Settings
from app.exceptions import BadRequestError, ModelNotFound, ProviderError, RateLimitExceeded
from app.middleware.ratelimit import enforce_rate_limit
from app.models.chat_completion import ChatCompletionRequest, ChatCompletionResponse
from app.providers.base import (
    ProviderError as UpstreamProviderError,
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


def _log_chat_event(
    *,
    level: int,
    request_id: str | None,
    api_key: str,
    model: str,
    provider: str,
    latency_ms: int,
    tokens: dict[str, int],
    cost: float,
    status: str,
    error: str | None = None,
    error_context: dict[str, object] | None = None,
    exc_info: bool = False,
) -> None:
    payload: dict[str, object] = {
        "request_id": request_id,
        "api_key": _mask_api_key(api_key),
        "model": model,
        "provider": provider,
        "latency_ms": latency_ms,
        "tokens": tokens,
        "cost": cost,
        "status": status,
    }
    if error is not None:
        payload["error"] = error
    if error_context:
        payload["error_context"] = error_context
    logger.log(level, "chat.request", extra={"payload": payload}, exc_info=exc_info)


@router.post(
    "/chat/completions",
    response_model=ChatCompletionResponse,
)
async def chat_completions(
    body: ChatCompletionRequest,
    http_request: Request,
    background_tasks: BackgroundTasks,
    api_key: str = Depends(enforce_rate_limit),
) -> ChatCompletionResponse:
    if body.stream:
        raise BadRequestError(
            "Streaming is not supported. Set stream=false.",
            code="stream_not_supported",
        )

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
        provider = get_provider(body.model)
    except UnknownModelError as exc:
        _log_chat_event(
            level=logging.WARNING,
            request_id=request_id,
            api_key=api_key,
            model=body.model,
            provider=provider_name,
            latency_ms=int((time.perf_counter() - started) * 1000),
            tokens=usage_payload,
            cost=total_cost,
            status="error",
            error=f"Unknown model: {exc.model}",
        )
        raise ModelNotFound(exc.model) from exc
    except RuntimeError as exc:
        _log_chat_event(
            level=logging.ERROR,
            request_id=request_id,
            api_key=api_key,
            model=body.model,
            provider=provider_name,
            latency_ms=int((time.perf_counter() - started) * 1000),
            tokens=usage_payload,
            cost=total_cost,
            status="error",
            error="Provider registry is not initialized",
            exc_info=True,
        )
        raise ProviderError(
            "Provider registry is not initialized",
            code="registry_not_initialized",
        ) from exc

    try:
        response = await provider.chat_completion(body)
    except ProviderRateLimitError as exc:
        _log_chat_event(
            level=logging.WARNING,
            request_id=request_id,
            api_key=api_key,
            model=body.model,
            provider=provider_name,
            latency_ms=int((time.perf_counter() - started) * 1000),
            tokens=usage_payload,
            cost=total_cost,
            status="error",
            error=str(exc),
            error_context={"upstream_status": getattr(exc, "status_code", None)},
        )
        raise RateLimitExceeded(
            message=str(exc),
            code="upstream_rate_limited",
            context={"upstream_status": getattr(exc, "status_code", None)},
        ) from exc
    except UpstreamProviderError as exc:
        _log_chat_event(
            level=logging.ERROR,
            request_id=request_id,
            api_key=api_key,
            model=body.model,
            provider=provider_name,
            latency_ms=int((time.perf_counter() - started) * 1000),
            tokens=usage_payload,
            cost=total_cost,
            status="error",
            error=str(exc),
            error_context={
                "upstream_status": getattr(exc, "status_code", None),
                "upstream_body_preview": getattr(exc, "body_preview", None),
                "exception_class": exc.__class__.__name__,
            },
            exc_info=True,
        )
        raise ProviderError(
            message="Upstream provider request failed",
            code="provider_request_failed",
            context={
                "upstream_status": getattr(exc, "status_code", None),
                "upstream_body_preview": getattr(exc, "body_preview", None),
                "exception_class": exc.__class__.__name__,
            },
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
        background_tasks.add_task(
            _persist_usage_stats,
            redis_service=http_request.app.state.redis_service,
            api_key=api_key,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_cost=total_cost,
        )

    _log_chat_event(
        level=logging.INFO,
        request_id=request_id,
        api_key=api_key,
        model=body.model,
        provider=provider_name,
        latency_ms=int((time.perf_counter() - started) * 1000),
        tokens=usage_payload,
        cost=total_cost,
        status="ok",
    )
    return response
