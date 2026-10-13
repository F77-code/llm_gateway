"""Shared fixtures for all tests.

The test app is built from scratch (without lifespan side effects):
  - fakeredis replaces real Redis
  - a bare httpx.AsyncClient is injected into providers; per-test mocks
    patch BaseLLMProvider.chat_completion so no real HTTP calls are made
  - asgi-lifespan triggers FastAPI startup/shutdown so app.state is populated
"""

from __future__ import annotations

import hashlib
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING
from unittest.mock import patch
from uuid import uuid4

import fakeredis.aioredis
import httpx
import pytest
import pytest_asyncio
from asgi_lifespan import LifespanManager
from fastapi import FastAPI, Request
from pydantic import SecretStr

from app.config import Settings
from app.exceptions import AppError
from app.logging_config import setup_logging
from app.main import app_error_handler, unhandled_exception_handler
from app.providers.registry import configure_registry, reset_registry
from app.routers import api_router
from app.services.redis import RedisService

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

setup_logging()

CLIENT_ID = "client-alpha"
OTHER_CLIENT_ID = "client-beta"
REVOKED_CLIENT_ID = "client-revoked"
GATEWAY_KEY = "gw_test_alpha_Q2jp91PLvTUf68DYsKc0eWn4bRmxA5h7sZQ"
OTHER_GATEWAY_KEY = "gw_test_beta_C83kP4mLX0rG2WqF9aN6vJtZ1yEbH5u8dRC"
ROTATED_GATEWAY_KEY = "gw_test_rotated_V4yF0K2wB7nXa9qG3zL5mHe6tRj1cDu8pQS"
REVOKED_GATEWAY_KEY = "gw_test_revoked_T1xQ8W2nP5fV0hS7jY4bD9mK6cLgA3uZeRC"
AUTH_HEADERS = {"Authorization": f"Bearer {GATEWAY_KEY}"}
OTHER_AUTH_HEADERS = {"Authorization": f"Bearer {OTHER_GATEWAY_KEY}"}
ROTATED_AUTH_HEADERS = {"Authorization": f"Bearer {ROTATED_GATEWAY_KEY}"}


def key_digest(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def client_registrations() -> list[dict[str, str | bool]]:
    return [
        {"client_id": CLIENT_ID, "key_sha256": key_digest(GATEWAY_KEY)},
        {"client_id": CLIENT_ID, "key_sha256": key_digest(ROTATED_GATEWAY_KEY)},
        {"client_id": OTHER_CLIENT_ID, "key_sha256": key_digest(OTHER_GATEWAY_KEY)},
        {
            "client_id": REVOKED_CLIENT_ID,
            "key_sha256": key_digest(REVOKED_GATEWAY_KEY),
            "enabled": False,
        },
    ]


def make_test_settings() -> Settings:
    return Settings(
        _env_file=None,
        openai_api_key=SecretStr("sk-test-openai"),
        anthropic_api_key=SecretStr("sk-test-anthropic"),
        gemini_api_key=SecretStr("test-gemini"),
        deepseek_api_key=SecretStr("sk-test-deepseek"),
        perplexity_api_key=SecretStr("pplx-test"),
        xai_api_key=SecretStr("xai-test"),
        redis_url="fake://",
        rate_limit_rpm=20,
        gateway_clients=client_registrations(),
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

    # Exercise the production handlers, including their structured log payloads.
    app.add_exception_handler(AppError, app_error_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)

    return app


@pytest_asyncio.fixture
async def fake_redis():
    r = fakeredis.aioredis.FakeRedis(decode_responses=True)
    yield r
    await r.aclose()


@pytest_asyncio.fixture
async def redis_svc(fake_redis) -> RedisService:
    return RedisService(fake_redis)


@pytest.fixture
def test_settings() -> Settings:
    return make_test_settings()


@pytest_asyncio.fixture
async def client(fake_redis, test_settings):
    settings = test_settings
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
