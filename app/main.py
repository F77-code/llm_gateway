from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
from fastapi import FastAPI

from app.config import get_settings
from app.providers.registry import configure_registry, reset_registry
from app.routers import api_router
from app.services.redis import RedisService


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
app.include_router(api_router)
