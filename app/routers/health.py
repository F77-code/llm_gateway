from __future__ import annotations

import asyncio
import time
from typing import Any

import redis.asyncio as redis
from fastapi import APIRouter, Request, Response, status

from app.config import Provider, Settings
from app.providers.registry import get_provider

router = APIRouter(tags=["health"])


def _representative_models() -> dict[Provider, str]:
    models: dict[Provider, str] = {}
    for model_name, provider in Settings.provider_by_model.items():
        models.setdefault(provider, model_name)
    return models


async def _check_provider(provider: Provider, model_name: str) -> tuple[str, dict[str, Any]]:
    started = time.perf_counter()
    try:
        instance = get_provider(model_name)
        ok = await instance.health_check()
        latency_ms = int((time.perf_counter() - started) * 1000)
        if ok:
            return provider.value, {"status": "ok", "latency_ms": latency_ms}
        return provider.value, {"status": "error", "latency_ms": latency_ms, "error": "health_check returned false"}
    except Exception as exc:
        latency_ms = int((time.perf_counter() - started) * 1000)
        return provider.value, {"status": "error", "latency_ms": latency_ms, "error": str(exc)}


@router.get("/health")
async def health(request: Request, response: Response) -> dict[str, Any]:
    redis_status = "ok"
    try:
        pong = await request.app.state.redis_service.client.ping()
        if not pong:
            redis_status = "error"
    except redis.RedisError as exc:
        redis_status = f"error: {exc}"

    checks = [
        _check_provider(provider, model_name)
        for provider, model_name in _representative_models().items()
    ]
    results = await asyncio.gather(*checks)
    providers = dict(results)

    has_healthy_provider = any(v.get("status") == "ok" for v in providers.values())
    if not has_healthy_provider:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    overall = "healthy" if has_healthy_provider and redis_status == "ok" else "degraded"
    if not has_healthy_provider:
        overall = "unhealthy"

    return {
        "status": overall,
        "redis": redis_status,
        "providers": providers,
    }
