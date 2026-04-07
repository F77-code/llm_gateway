"""Shared fixtures for all tests.

The test app is built from scratch (without lifespan side effects):
  - fakeredis replaces real Redis
  - a bare httpx.AsyncClient is injected into providers; per-test mocks
    patch BaseLLMProvider.chat_completion so no real HTTP calls are made
  - asgi-lifespan triggers FastAPI startup/shutdown so app.state is populated
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator
from unittest.mock import patch
from uuid import uuid4

import fakeredis.aioredis
import httpx
import pytest
import pytest_asyncio
from asgi_lifespan import LifespanManager
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import SecretStr

from app.config import Settings
from app.exceptions import AppError, AuthenticationError, RateLimitExceeded
from app.logging_config import setup_logging
from app.main import _openai_error_payload
from app.providers.registry import configure_registry, reset_registry
from app.routers import api_router
from app.services.redis import RedisService

setup_logging()

GATEWAY_KEY = "test-gateway-key"
AUTH_HEADERS = {"Authorization": f"Bearer {GATEWAY_KEY}"}


def make_test_settings() -> Settings:
    return Settings(
        openai_api_key=SecretStr("sk-test-openai"),
        anthropic_api_key=SecretStr("sk-test-anthropic"),
        gemini_api_key=SecretStr("test-gemini"),
        deepseek_api_key=SecretStr("sk-test-deepseek"),
        perplexity_api_key=SecretStr("pplx-test"),
        xai_api_key=SecretStr("xai-test"),
        redis_url="fake://",
        rate_limit_rpm=20,
        default_api_key=SecretStr(GATEWAY_KEY),
    )


def build_test_app(redis_svc: RedisService, settings: Settings) -> FastAPI:
    """Create a self-contained test app with fake dependencies."""
    mock_http = httpx.AsyncClient()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        app.state.redis_service = redis_svc
        app.state.redis = redis_svc.client
        app.state.http_client = mock_http
        configure_registry(mock_http, settings)
        try:
            yield
        finally:
            reset_registry()
            await mock_http.aclose()

    app = FastAPI(lifespan=lifespan)

    @app.middleware("http")
    async def attach_request_id(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID") or uuid4().hex
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    app.include_router(api_router)

    @app.exception_handler(AppError)
    async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
        headers = None
        if isinstance(exc, AuthenticationError):
            headers = {"WWW-Authenticate": "Bearer"}
        if isinstance(exc, RateLimitExceeded):
            raw = exc.context.get("headers")
            if isinstance(raw, dict):
                headers = {str(k): str(v) for k, v in raw.items()}
        return JSONResponse(
            status_code=exc.status_code,
            content=_openai_error_payload(exc.message, exc.error_type, exc.code),
            headers=headers,
        )

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content=_openai_error_payload(
                "Internal server error", "server_error", "internal_error"
            ),
        )

    return app


@pytest_asyncio.fixture
async def fake_redis():
    r = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield r
    await r.aclose()


@pytest_asyncio.fixture
async def redis_svc(fake_redis) -> RedisService:
    return RedisService(fake_redis)


@pytest_asyncio.fixture
async def client(fake_redis):
    settings = make_test_settings()
    svc = RedisService(fake_redis)
    app = build_test_app(svc, settings)

    # Patch get_settings so that auth/rate-limit middleware uses test settings,
    # not whatever is in the real .env (or the lru_cache from a previous test).
    with (
        patch("app.dependencies.get_settings", return_value=settings),
        patch("app.middleware.ratelimit.get_settings", return_value=settings),
    ):
        async with LifespanManager(app):
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as c:
                yield c
