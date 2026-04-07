from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator
from uuid import uuid4

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.exceptions import AppError, AuthenticationError, RateLimitExceeded
from app.logging_config import setup_logging
from app.providers.registry import configure_registry, reset_registry
from app.routers import api_router
from app.services.redis import RedisService

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    app.state.redis_service = RedisService.from_url(settings.redis_url)
    app.state.redis = app.state.redis_service.client
    app.state.http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(60.0, connect=10.0),
        limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
    )
    configure_registry(app.state.http_client, settings)
    try:
        yield
    finally:
        reset_registry()
        await app.state.http_client.aclose()
        await app.state.redis_service.aclose()


app = FastAPI(title="LLM Gateway", lifespan=lifespan)


@app.middleware("http")
async def attach_request_id(request: Request, call_next):
    request_id = request.headers.get("X-Request-ID") or uuid4().hex
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


app.include_router(api_router)


def _openai_error_payload(message: str, error_type: str, code: str | None) -> dict[str, object]:
    return {
        "error": {
            "message": message,
            "type": error_type,
            "param": None,
            "code": code,
        },
    }


@app.exception_handler(AppError)
async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    payload = {
        "request_id": getattr(request.state, "request_id", None),
        "path": str(request.url.path),
        "method": request.method,
        "error_type": exc.error_type,
        "code": exc.code,
        "context": exc.context,
    }
    if isinstance(exc, (RateLimitExceeded, AuthenticationError)):
        logger.warning("app.error", extra={"payload": payload})
    else:
        logger.error("app.error", extra={"payload": payload})

    headers: dict[str, str] | None = None
    if isinstance(exc, AuthenticationError):
        headers = {"WWW-Authenticate": "Bearer"}
    if isinstance(exc, RateLimitExceeded):
        raw_headers = exc.context.get("headers")
        if isinstance(raw_headers, dict):
            headers = {str(k): str(v) for k, v in raw_headers.items()}
    return JSONResponse(
        status_code=exc.status_code,
        content=_openai_error_payload(exc.message, exc.error_type, exc.code),
        headers=headers,
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception(
        "unhandled.exception",
        extra={
            "payload": {
                "request_id": getattr(request.state, "request_id", None),
                "path": str(request.url.path),
                "method": request.method,
                "exception_class": exc.__class__.__name__,
            },
        },
    )
    return JSONResponse(
        status_code=500,
        content=_openai_error_payload(
            "Internal server error",
            "server_error",
            "internal_error",
        ),
    )
