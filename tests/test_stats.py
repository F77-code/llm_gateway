"""Authenticated usage uses stable client identities and never URL secrets."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.services.redis import RedisService
from tests.conftest import (
    AUTH_HEADERS,
    CLIENT_ID,
    GATEWAY_KEY,
    OTHER_AUTH_HEADERS,
    OTHER_CLIENT_ID,
    OTHER_GATEWAY_KEY,
    ROTATED_AUTH_HEADERS,
    ROTATED_GATEWAY_KEY,
    key_digest,
)


async def test_usage_empty_returns_zeros(client):
    response = await client.get("/me/usage", headers=AUTH_HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert data["client_id"] == CLIENT_ID
    assert data["today"] == {
        "request_count": 0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_cost_usd": 0.0,
    }
    assert data["current_month"] == data["today"]


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer unknown-token"}])
async def test_usage_requires_valid_authentication(client, headers):
    response = await client.get("/me/usage", headers=headers)
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.json()["error"]["type"] == "authentication_error"


async def test_old_stats_path_is_removed(client):
    response = await client.get("/stats/removed-key", headers=AUTH_HEADERS)
    assert response.status_code == 404


async def test_usage_reflects_persisted_client_aggregates(client, fake_redis):
    await RedisService(fake_redis).increment_usage_stats(
        client_id=CLIENT_ID,
        prompt_tokens=150,
        completion_tokens=250,
        total_cost=0.02,
    )
    response = await client.get("/me/usage", headers=AUTH_HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert data["client_id"] == CLIENT_ID
    assert data["today"] == {
        "request_count": 1,
        "prompt_tokens": 150,
        "completion_tokens": 250,
        "total_cost_usd": 0.02,
    }
    assert data["current_month"] == data["today"]


async def test_usage_accumulates_multiple_requests(client, fake_redis):
    service = RedisService(fake_redis)
    for _ in range(4):
        await service.increment_usage_stats(
            client_id=CLIENT_ID,
            prompt_tokens=100,
            completion_tokens=100,
            total_cost=0.005,
        )
    response = await client.get("/me/usage", headers=AUTH_HEADERS)
    assert response.status_code == 200
    data = response.json()
    assert data["today"]["request_count"] == 4
    assert data["today"]["prompt_tokens"] == 400
    assert data["current_month"]["completion_tokens"] == 400
    assert data["today"]["total_cost_usd"] == pytest.approx(0.02)


async def test_usage_isolated_between_clients_and_not_selected_by_query(
    client, fake_redis
):
    service = RedisService(fake_redis)
    await service.increment_usage_stats(
        client_id=CLIENT_ID, prompt_tokens=10, completion_tokens=20, total_cost=0.01
    )
    await service.increment_usage_stats(
        client_id=OTHER_CLIENT_ID,
        prompt_tokens=70,
        completion_tokens=80,
        total_cost=0.02,
    )
    first = await client.get(
        "/me/usage", headers=AUTH_HEADERS, params={"client_id": OTHER_CLIENT_ID}
    )
    second = await client.get("/me/usage", headers=OTHER_AUTH_HEADERS)
    assert first.status_code == second.status_code == 200
    assert first.json()["client_id"] == CLIENT_ID
    assert second.json()["client_id"] == OTHER_CLIENT_ID
    assert first.json()["today"]["prompt_tokens"] == 10
    assert second.json()["today"]["prompt_tokens"] == 70
    assert first.json()["current_month"]["completion_tokens"] == 20
    assert second.json()["current_month"]["completion_tokens"] == 80


async def test_key_rotation_reads_same_existing_usage(client, fake_redis):
    await RedisService(fake_redis).increment_usage_stats(
        client_id=CLIENT_ID, prompt_tokens=17, completion_tokens=19, total_cost=0.01
    )
    original = await client.get("/me/usage", headers=AUTH_HEADERS)
    rotated = await client.get("/me/usage", headers=ROTATED_AUTH_HEADERS)
    assert original.status_code == rotated.status_code == 200
    assert original.json() == rotated.json()
    assert rotated.json()["today"]["request_count"] == 1


async def test_usage_response_contains_identity_without_bearer_suffix_or_hash(client):
    response = await client.get("/me/usage", headers=AUTH_HEADERS)
    assert set(response.json()) == {"client_id", "today", "current_month"}
    for secret in (GATEWAY_KEY, OTHER_GATEWAY_KEY, ROTATED_GATEWAY_KEY):
        assert secret not in response.text
        assert secret[-4:] not in response.text
        assert key_digest(secret) not in response.text
    assert "api_key" not in response.json()


async def test_legacy_secret_indexed_history_is_not_silently_transferred(
    client, fake_redis
):
    day = datetime.now(UTC).strftime("%Y-%m-%d")
    legacy_key = f"usage:{GATEWAY_KEY}:daily:{day}"
    await fake_redis.hset(
        legacy_key, mapping={"request_count": 9, "prompt_tokens": 900}
    )
    response = await client.get("/me/usage", headers=AUTH_HEADERS)
    assert response.status_code == 200
    assert response.json()["today"]["request_count"] == 0
    assert await fake_redis.hget(legacy_key, "request_count") == "9"
