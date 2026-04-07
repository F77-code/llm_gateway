"""Integration tests for POST /v1/chat/completions.

All provider HTTP calls are mocked at the BaseLLMProvider.chat_completion level
so no real network requests are made.
"""

from __future__ import annotations

import time

import pytest
from unittest.mock import AsyncMock

from app.models.chat_completion import ChatCompletionChoice, ChatCompletionResponse
from app.models.message import Message
from app.models.usage import Usage
from app.providers.anthropic import AnthropicProvider
from app.providers.base import ProviderRateLimitError, ProviderServerError
from app.providers.openai import OpenAIProvider

from tests.conftest import AUTH_HEADERS


def _make_response(model: str = "gpt-4.1-mini") -> ChatCompletionResponse:
    return ChatCompletionResponse(
        id="chatcmpl-test123",
        created=int(time.time()),
        model=model,
        choices=[
            ChatCompletionChoice(
                index=0,
                message=Message(role="assistant", content="Hello! How can I help you?"),
                finish_reason="stop",
            )
        ],
        usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
    )


async def test_chat_success(client, mocker):
    """Happy path: returns 200 with the assistant message."""
    mocker.patch.object(
        OpenAIProvider,
        "chat_completion",
        new_callable=AsyncMock,
        return_value=_make_response(),
    )

    resp = await client.post(
        "/v1/chat/completions",
        headers=AUTH_HEADERS,
        json={"model": "gpt-4.1-mini", "messages": [{"role": "user", "content": "Hello"}]},
    )

    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == "chatcmpl-test123"
    assert data["choices"][0]["message"]["content"] == "Hello! How can I help you?"
    assert data["usage"]["total_tokens"] == 30


async def test_chat_routes_to_anthropic(client, mocker):
    """Model routing: claude-* should reach AnthropicProvider."""
    mock = mocker.patch.object(
        AnthropicProvider,
        "chat_completion",
        new_callable=AsyncMock,
        return_value=_make_response(model="claude-opus-4-6"),
    )

    resp = await client.post(
        "/v1/chat/completions",
        headers=AUTH_HEADERS,
        json={
            "model": "claude-opus-4-6",
            "messages": [{"role": "user", "content": "Hello Claude"}],
        },
    )

    assert resp.status_code == 200
    mock.assert_called_once()


async def test_chat_unknown_model(client):
    """Requesting an unknown model returns 400 model_not_found."""
    resp = await client.post(
        "/v1/chat/completions",
        headers=AUTH_HEADERS,
        json={
            "model": "totally-unknown-model-xyz",
            "messages": [{"role": "user", "content": "Hi"}],
        },
    )

    assert resp.status_code == 400
    data = resp.json()
    assert data["error"]["code"] == "model_not_found"
    assert data["error"]["type"] == "invalid_request_error"


async def test_chat_missing_auth(client):
    """Missing Authorization header → 422 (FastAPI validation error)."""
    resp = await client.post(
        "/v1/chat/completions",
        json={"model": "gpt-4.1-mini", "messages": [{"role": "user", "content": "Hi"}]},
    )
    assert resp.status_code == 422


async def test_chat_wrong_auth(client):
    """Wrong Bearer token → 401 authentication_error."""
    resp = await client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer totally-wrong-key"},
        json={"model": "gpt-4.1-mini", "messages": [{"role": "user", "content": "Hi"}]},
    )

    assert resp.status_code == 401
    data = resp.json()
    assert data["error"]["type"] == "authentication_error"


async def test_chat_invalid_auth_scheme(client):
    """Non-Bearer scheme → 401 with invalid_authorization_header code."""
    resp = await client.post(
        "/v1/chat/completions",
        headers={"Authorization": "Basic dXNlcjpwYXNz"},
        json={"model": "gpt-4.1-mini", "messages": [{"role": "user", "content": "Hi"}]},
    )

    assert resp.status_code == 401
    data = resp.json()
    assert data["error"]["code"] == "invalid_authorization_header"


async def test_chat_stream_not_supported(client):
    """stream=True must return 400, not 502."""
    resp = await client.post(
        "/v1/chat/completions",
        headers=AUTH_HEADERS,
        json={
            "model": "gpt-4.1-mini",
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": True,
        },
    )

    assert resp.status_code == 400
    data = resp.json()
    assert data["error"]["code"] == "stream_not_supported"
    assert data["error"]["type"] == "invalid_request_error"


async def test_chat_upstream_server_error(client, mocker):
    """Provider 5xx → gateway returns 502 provider_error."""
    mocker.patch.object(
        OpenAIProvider,
        "chat_completion",
        new_callable=AsyncMock,
        side_effect=ProviderServerError("HTTP 500", status_code=500),
    )

    resp = await client.post(
        "/v1/chat/completions",
        headers=AUTH_HEADERS,
        json={"model": "gpt-4.1-mini", "messages": [{"role": "user", "content": "Hi"}]},
    )

    assert resp.status_code == 502
    data = resp.json()
    assert data["error"]["type"] == "provider_error"


async def test_chat_upstream_rate_limit(client, mocker):
    """Provider 429 → gateway returns 429 with upstream_rate_limited code."""
    mocker.patch.object(
        OpenAIProvider,
        "chat_completion",
        new_callable=AsyncMock,
        side_effect=ProviderRateLimitError("HTTP 429", status_code=429),
    )

    resp = await client.post(
        "/v1/chat/completions",
        headers=AUTH_HEADERS,
        json={"model": "gpt-4.1-mini", "messages": [{"role": "user", "content": "Hi"}]},
    )

    assert resp.status_code == 429
    data = resp.json()
    assert data["error"]["code"] == "upstream_rate_limited"


async def test_chat_response_has_request_id_header(client, mocker):
    """Every response must carry an X-Request-ID header."""
    mocker.patch.object(
        OpenAIProvider,
        "chat_completion",
        new_callable=AsyncMock,
        return_value=_make_response(),
    )

    resp = await client.post(
        "/v1/chat/completions",
        headers=AUTH_HEADERS,
        json={"model": "gpt-4.1-mini", "messages": [{"role": "user", "content": "Hi"}]},
    )

    assert "x-request-id" in resp.headers


async def test_chat_custom_request_id_is_echoed(client, mocker):
    """If client sends X-Request-ID, the same value must be returned."""
    mocker.patch.object(
        OpenAIProvider,
        "chat_completion",
        new_callable=AsyncMock,
        return_value=_make_response(),
    )

    resp = await client.post(
        "/v1/chat/completions",
        headers={**AUTH_HEADERS, "X-Request-ID": "my-custom-id-42"},
        json={"model": "gpt-4.1-mini", "messages": [{"role": "user", "content": "Hi"}]},
    )

    assert resp.headers["x-request-id"] == "my-custom-id-42"


async def test_chat_usage_in_response(client, mocker):
    """Response must include token usage statistics."""
    mocker.patch.object(
        OpenAIProvider,
        "chat_completion",
        new_callable=AsyncMock,
        return_value=_make_response(),
    )

    resp = await client.post(
        "/v1/chat/completions",
        headers=AUTH_HEADERS,
        json={"model": "gpt-4.1-mini", "messages": [{"role": "user", "content": "Hi"}]},
    )

    data = resp.json()
    assert data["usage"]["prompt_tokens"] == 10
    assert data["usage"]["completion_tokens"] == 20
    assert data["usage"]["total_tokens"] == 30
