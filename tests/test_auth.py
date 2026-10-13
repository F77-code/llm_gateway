"""Bearer authentication, validated registrations and stable client isolation."""

from __future__ import annotations

import json
import logging
from unittest.mock import AsyncMock

import pytest
import redis.asyncio as redis
from pydantic import ValidationError

from app.auth import ClientRegistration, authenticate_bearer
from app.config import Settings
from app.exceptions import AuthenticationError
from app.logging_config import JsonFormatter
from app.providers.openai import OpenAIProvider
from tests.conftest import (
    AUTH_HEADERS,
    CLIENT_ID,
    GATEWAY_KEY,
    OTHER_AUTH_HEADERS,
    OTHER_CLIENT_ID,
    OTHER_GATEWAY_KEY,
    REVOKED_GATEWAY_KEY,
    ROTATED_AUTH_HEADERS,
    ROTATED_GATEWAY_KEY,
    key_digest,
)
from tests.test_chat import _make_response

CHAT_BODY = {
    "model": "gpt-4.1-mini",
    "messages": [{"role": "user", "content": "Hello"}],
}
SECRETS = (GATEWAY_KEY, OTHER_GATEWAY_KEY, ROTATED_GATEWAY_KEY, REVOKED_GATEWAY_KEY)


def _assert_no_credentials(text: str) -> None:
    for secret in SECRETS:
        assert secret not in text
        assert key_digest(secret) not in text


@pytest.mark.parametrize(
    "authorization",
    [
        None,
        "",
        "Bearer",
        "Bearer ",
        "Basic dXNlcjpwYXNz",
        "Bearer unknown-but-well-formed-token",
        f"Bearer {REVOKED_GATEWAY_KEY}",
        f"Bearer\t{GATEWAY_KEY}",
        f"Bearer {GATEWAY_KEY} ",
        f"Bearer {GATEWAY_KEY}\t",
        f"Bearer {GATEWAY_KEY},Bearer {OTHER_GATEWAY_KEY}",
        "Bearer " + "x" * 1_025,
    ],
)
async def test_invalid_authentication_is_uniform_and_has_no_side_effects(
    client, fake_redis, mocker, authorization
):
    provider = mocker.patch.object(
        OpenAIProvider, "chat_completion", new_callable=AsyncMock
    )
    headers = {} if authorization is None else {"Authorization": authorization}
    response = await client.post(
        "/v1/chat/completions", headers=headers, json=CHAT_BODY
    )
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.json()["error"] == {
        "message": "Invalid gateway credentials",
        "type": "authentication_error",
        "param": None,
        "code": "invalid_api_key",
    }
    assert await fake_redis.keys("*") == []
    provider.assert_not_awaited()
    _assert_no_credentials(response.text + json.dumps(dict(response.headers)))


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("second", ["Bearer unknown-token", "Basic opaque-token"])
async def test_duplicate_authorization_headers_are_rejected(
    client, fake_redis, mocker, reverse, second
):
    provider = mocker.patch.object(
        OpenAIProvider, "chat_completion", new_callable=AsyncMock
    )
    headers = [("Authorization", f"Bearer {GATEWAY_KEY}"), ("Authorization", second)]
    if reverse:
        headers.reverse()
    response = await client.post(
        "/v1/chat/completions", headers=headers, json=CHAT_BODY
    )
    assert response.status_code == 401
    assert response.headers["www-authenticate"] == "Bearer"
    assert response.json()["error"]["code"] == "invalid_api_key"
    assert await fake_redis.keys("*") == []
    provider.assert_not_awaited()
    _assert_no_credentials(response.text)


async def test_unconfigured_registry_fails_closed_even_with_legacy_default_key(
    client, fake_redis, test_settings, monkeypatch
):
    monkeypatch.setenv("DEFAULT_API_KEY", GATEWAY_KEY)
    test_settings.gateway_clients = []
    for headers in (AUTH_HEADERS, OTHER_AUTH_HEADERS):
        response = await client.post(
            "/v1/chat/completions", headers=headers, json=CHAT_BODY
        )
        assert response.status_code == 401
        assert response.headers["www-authenticate"] == "Bearer"
    assert await fake_redis.keys("*") == []


def test_default_registry_is_empty_and_legacy_environment_does_not_authorize(
    monkeypatch,
):
    monkeypatch.delenv("GATEWAY_CLIENTS", raising=False)
    monkeypatch.setenv("DEFAULT_API_KEY", GATEWAY_KEY)
    settings = Settings(_env_file=None)
    assert settings.gateway_clients == []
    with pytest.raises(AuthenticationError):
        authenticate_bearer(f"Bearer {GATEWAY_KEY}", settings.gateway_clients)


@pytest.mark.parametrize("scheme_and_spaces", ["Bearer ", "bEaReR ", "Bearer  "])
def test_valid_bearer_scheme_and_spaces_preserve_registered_identity(
    test_settings, scheme_and_spaces
):
    assert (
        authenticate_bearer(
            f"{scheme_and_spaces}{GATEWAY_KEY}", test_settings.gateway_clients
        )
        == CLIENT_ID
    )


@pytest.mark.parametrize(
    "authorization", ["Bearer nonascii-ключ", "Bearer opaque\nkey"]
)
def test_non_ascii_and_control_characters_fail_without_echo(
    test_settings, authorization
):
    with pytest.raises(AuthenticationError) as error:
        authenticate_bearer(authorization, test_settings.gateway_clients)
    assert str(error.value) == "Invalid gateway credentials"
    assert authorization not in str(error.value)


@pytest.mark.parametrize(
    "client_id",
    ["", "has:colon", "has/slash", "has.dot", "has space", "ключ", "a" * 65],
)
def test_registration_rejects_unsafe_client_identifiers(client_id):
    with pytest.raises(ValidationError):
        ClientRegistration(client_id=client_id, key_sha256=key_digest(GATEWAY_KEY))


@pytest.mark.parametrize("digest", ["", "a" * 63, "a" * 65, "g" * 64, GATEWAY_KEY])
def test_registration_rejects_invalid_hash_without_echo(digest):
    with pytest.raises(ValidationError) as error:
        ClientRegistration(client_id=CLIENT_ID, key_sha256=digest)
    if digest:
        assert digest not in str(error.value)


@pytest.mark.parametrize("enabled", [0, 1, "false", "true", None])
def test_registration_requires_boolean_revocation_flag(enabled):
    with pytest.raises(ValidationError):
        ClientRegistration(
            client_id=CLIENT_ID, key_sha256=key_digest(GATEWAY_KEY), enabled=enabled
        )


def test_registration_normalizes_hash_case_and_keeps_it_out_of_repr():
    digest = key_digest(GATEWAY_KEY)
    registration = ClientRegistration(client_id=CLIENT_ID, key_sha256=digest.upper())
    assert registration.key_sha256 == digest
    assert digest not in repr(registration)
    assert digest.upper() not in repr(registration)


@pytest.mark.parametrize("enabled", [False, True])
def test_settings_reject_duplicate_hashes_across_clients_even_if_revoked(enabled):
    digest = key_digest(GATEWAY_KEY)
    with pytest.raises(ValidationError, match="Duplicate client key hashes") as error:
        Settings(
            _env_file=None,
            gateway_clients=[
                {"client_id": CLIENT_ID, "key_sha256": digest},
                {
                    "client_id": OTHER_CLIENT_ID,
                    "key_sha256": digest.upper(),
                    "enabled": enabled,
                },
            ],
        )
    assert digest not in str(error.value)
    assert GATEWAY_KEY not in str(error.value)


def test_settings_reads_explicit_json_registrations_from_environment(monkeypatch):
    monkeypatch.setenv(
        "GATEWAY_CLIENTS",
        json.dumps([{"client_id": CLIENT_ID, "key_sha256": key_digest(GATEWAY_KEY)}]),
    )
    settings = Settings(_env_file=None)
    assert (
        authenticate_bearer(f"Bearer {GATEWAY_KEY}", settings.gateway_clients)
        == CLIENT_ID
    )
    _assert_no_credentials(repr(settings))


@pytest.mark.parametrize("credential_identity", [CLIENT_ID, OTHER_CLIENT_ID])
def test_settings_reject_public_id_as_same_or_other_client_bearer(credential_identity):
    with pytest.raises(
        ValidationError, match="Client identifiers must not also be bearer credentials"
    ):
        Settings(
            _env_file=None,
            gateway_clients=[
                {"client_id": CLIENT_ID, "key_sha256": key_digest(credential_identity)},
                {
                    "client_id": OTHER_CLIENT_ID,
                    "key_sha256": key_digest(OTHER_GATEWAY_KEY),
                },
            ],
        )


async def test_rotation_preserves_shared_quota_and_usage(
    client, fake_redis, test_settings, mocker
):
    test_settings.rate_limit_rpm = 2
    provider = mocker.patch.object(
        OpenAIProvider,
        "chat_completion",
        new_callable=AsyncMock,
        return_value=_make_response(),
    )
    for headers, remaining in ((AUTH_HEADERS, "1"), (ROTATED_AUTH_HEADERS, "0")):
        response = await client.post(
            "/v1/chat/completions", headers=headers, json=CHAT_BODY
        )
        assert response.status_code == 200
        assert response.headers["x-ratelimit-remaining"] == remaining
    blocked = await client.post(
        "/v1/chat/completions", headers=ROTATED_AUTH_HEADERS, json=CHAT_BODY
    )
    assert blocked.status_code == 429
    assert blocked.headers["x-ratelimit-remaining"] == "0"
    original = await client.get("/me/usage", headers=AUTH_HEADERS)
    rotated = await client.get("/me/usage", headers=ROTATED_AUTH_HEADERS)
    assert original.json() == rotated.json()
    assert rotated.json()["client_id"] == CLIENT_ID
    assert rotated.json()["today"]["request_count"] == 2
    assert rotated.json()["today"]["prompt_tokens"] == 20
    assert await fake_redis.zcard(f"ratelimit:{CLIENT_ID}") == 2
    assert provider.await_count == 2


async def test_revoking_old_rotated_key_preserves_identity_and_existing_usage(
    client, test_settings, mocker
):
    mocker.patch.object(
        OpenAIProvider,
        "chat_completion",
        new_callable=AsyncMock,
        return_value=_make_response(),
    )
    admitted = await client.post(
        "/v1/chat/completions", headers=AUTH_HEADERS, json=CHAT_BODY
    )
    assert admitted.status_code == 200
    test_settings.gateway_clients = [
        registration.model_copy(update={"enabled": False})
        if registration.key_sha256 == key_digest(GATEWAY_KEY)
        else registration
        for registration in test_settings.gateway_clients
    ]
    old = await client.get("/me/usage", headers=AUTH_HEADERS)
    new = await client.get("/me/usage", headers=ROTATED_AUTH_HEADERS)
    assert old.status_code == 401
    assert new.status_code == 200
    assert new.json()["client_id"] == CLIENT_ID
    assert new.json()["today"]["request_count"] == 1


async def test_different_clients_have_independent_quota_and_persisted_usage(
    client, fake_redis, test_settings, mocker
):
    test_settings.rate_limit_rpm = 1
    mocker.patch.object(
        OpenAIProvider,
        "chat_completion",
        new_callable=AsyncMock,
        return_value=_make_response(),
    )
    first = await client.post(
        "/v1/chat/completions", headers=AUTH_HEADERS, json=CHAT_BODY
    )
    denied = await client.post(
        "/v1/chat/completions", headers=ROTATED_AUTH_HEADERS, json=CHAT_BODY
    )
    second = await client.post(
        "/v1/chat/completions", headers=OTHER_AUTH_HEADERS, json=CHAT_BODY
    )
    assert (first.status_code, denied.status_code, second.status_code) == (
        200,
        429,
        200,
    )
    assert await fake_redis.zcard(f"ratelimit:{CLIENT_ID}") == 1
    assert await fake_redis.zcard(f"ratelimit:{OTHER_CLIENT_ID}") == 1
    for headers, identity in (
        (AUTH_HEADERS, CLIENT_ID),
        (OTHER_AUTH_HEADERS, OTHER_CLIENT_ID),
    ):
        response = await client.get("/me/usage", headers=headers)
        assert response.json()["client_id"] == identity
        assert response.json()["today"]["request_count"] == 1
        assert response.json()["current_month"]["prompt_tokens"] == 10
    keys = await fake_redis.keys("*")
    assert len(keys) == 6
    _assert_no_credentials(json.dumps(keys))


async def test_credentials_absent_from_success_rejection_and_production_logs(
    client, fake_redis, test_settings, mocker, caplog
):
    test_settings.rate_limit_rpm = 1
    mocker.patch.object(
        OpenAIProvider,
        "chat_completion",
        new_callable=AsyncMock,
        return_value=_make_response(),
    )
    with caplog.at_level(logging.INFO):
        success = await client.post(
            "/v1/chat/completions", headers=AUTH_HEADERS, json=CHAT_BODY
        )
        rejected = await client.post(
            "/v1/chat/completions", headers=ROTATED_AUTH_HEADERS, json=CHAT_BODY
        )
        unauthorized = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {REVOKED_GATEWAY_KEY}"},
            json=CHAT_BODY,
        )
        usage = await client.get("/me/usage", headers=AUTH_HEADERS)
    assert (
        success.status_code,
        rejected.status_code,
        unauthorized.status_code,
        usage.status_code,
    ) == (200, 429, 401, 200)
    for response in (success, rejected, unauthorized, usage):
        _assert_no_credentials(response.text + json.dumps(dict(response.headers)))
    _assert_no_credentials(json.dumps(await fake_redis.keys("*")))
    rendered = "\n".join(JsonFormatter().format(record) for record in caplog.records)
    _assert_no_credentials(rendered)
    application_errors = [
        record
        for record in caplog.records
        if record.name == "app.main" and record.getMessage() == "app.error"
    ]
    assert {record.payload["code"] for record in application_errors} == {
        "rate_limit_exceeded",
        "invalid_api_key",
    }
    chat_records = [
        record for record in caplog.records if record.getMessage() == "chat.request"
    ]
    assert chat_records and all(
        record.payload["client_id"] == CLIENT_ID for record in chat_records
    )


async def test_usage_storage_error_is_generic_and_hides_backend_details(
    client, fake_redis, mocker
):
    mocker.patch.object(
        fake_redis,
        "hgetall",
        new_callable=AsyncMock,
        side_effect=redis.ConnectionError(
            f"backend detail {GATEWAY_KEY} {key_digest(GATEWAY_KEY)}"
        ),
    )
    response = await client.get("/me/usage", headers=AUTH_HEADERS)
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "usage_storage_unavailable"
    assert response.json()["error"]["message"] == "Usage storage unavailable"
    _assert_no_credentials(response.text)
