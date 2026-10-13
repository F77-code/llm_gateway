"""Explicit, isolated Redis connections; unavailable Redis always fails."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from urllib.parse import urlsplit
from uuid import uuid4

import pytest
import pytest_asyncio
import redis.asyncio as redis

if TYPE_CHECKING:
    from collections.abc import AsyncIterator


@dataclass
class RedisTestScope:
    url: str
    client: redis.Redis
    namespace: str = field(default_factory=lambda: uuid4().hex)
    owned_keys: set[str] = field(default_factory=set)

    def key(self, name: str) -> str:
        key = f"rl:integration:{self.namespace}:{name}"
        self.owned_keys.add(key)
        return key


def _endpoint(url: str) -> tuple[str | None, int]:
    parsed = urlsplit(url)
    return parsed.hostname, parsed.port or 6379


@pytest_asyncio.fixture
async def redis_scope() -> AsyncIterator[RedisTestScope]:
    url = os.environ.get("TEST_REDIS_URL")
    if not url:
        pytest.fail("TEST_REDIS_URL is required and must name a dedicated test Redis")
    if urlsplit(url).scheme not in {"redis", "rediss"}:
        pytest.fail("TEST_REDIS_URL must be an explicit redis:// or rediss:// URL")
    application_url = os.environ.get("REDIS_URL")
    if application_url and _endpoint(application_url) == _endpoint(url):
        pytest.fail("TEST_REDIS_URL must be distinct from application REDIS_URL")

    client = redis.from_url(
        url,
        decode_responses=True,
        socket_connect_timeout=2,
        socket_timeout=2,
    )
    scope = RedisTestScope(url=url, client=client)
    try:
        try:
            await client.ping()
        except redis.RedisError as exc:
            pytest.fail(f"Dedicated test Redis is unavailable ({type(exc).__name__})")
        yield scope
    finally:
        try:
            if scope.owned_keys:
                await client.delete(*scope.owned_keys)
        finally:
            await client.aclose()
