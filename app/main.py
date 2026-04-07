from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

import httpx
import redis.asyncio as redis
from fastapi import FastAPI

from app.config import get_settings
from app.providers.registry import configure_registry, reset_registry
from app.routers import api_router


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    app.state.redis = redis.from_url(
        settings.redis_url,
        decode_responses=True,
    )
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
        await app.state.redis.aclose()


app = FastAPI(title="LLM Gateway", lifespan=lifespan)
app.include_router(api_router)
