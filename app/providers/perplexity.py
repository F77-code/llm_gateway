from __future__ import annotations

import time
from typing import Any

import httpx
from pydantic import SecretStr

from app.models.chat_completion import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from app.models.message import Message
from app.models.usage import Usage
from app.providers.base import BaseLLMProvider, ProviderError

PERPLEXITY_MODELS: tuple[str, ...] = (
    "sonar",
    "sonar-pro",
    "sonar-reasoning-pro",
    "sonar-deep-research",
)

_FINISH_REASON_MAP: dict[str, str] = {
    "stop": "stop",
    "length": "length",
}

_EXTRA_REQUEST_KEYS: frozenset[str] = frozenset(
    {
        "search_domain_filter",
        "search_language_filter",
        "search_recency_filter",
        "search_after_date_filter",
        "search_before_date_filter",
        "last_updated_before_filter",
        "last_updated_after_filter",
        "return_images",
        "return_related_questions",
        "search_mode",
        "web_search_options",
        "enable_search_classifier",
        "disable_search",
        "reasoning_effort",
        "language_preference",
        "image_format_filter",
        "image_domain_filter",
    },
)


def _secret_value(key: str | SecretStr) -> str:
    if isinstance(key, SecretStr):
        return key.get_secret_value()
    return key


def _serialize_message(m: Message) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": m.role, "content": m.content or ""}
    return msg


def _build_perplexity_body(request: ChatCompletionRequest) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": request.model,
        "messages": [_serialize_message(m) for m in request.messages],
    }
    if request.temperature is not None:
        body["temperature"] = request.temperature
    if request.max_tokens is not None:
        body["max_tokens"] = request.max_tokens
    if request.top_p is not None:
        body["top_p"] = request.top_p
    if request.stop is not None:
        body["stop"] = request.stop
    if request.response_format is not None:
        body["response_format"] = request.response_format

    extra = request.model_extra or {}
    for key in _EXTRA_REQUEST_KEYS:
        val = extra.get(key)
        if val is not None:
            body[key] = val

    return body


def _parse_choice(idx: int, raw: dict[str, Any]) -> ChatCompletionChoice:
    raw_msg = raw.get("message")
    if not isinstance(raw_msg, dict):
        raw_msg = {"role": "assistant", "content": None}

    msg_payload: dict[str, Any] = {
        "role": raw_msg.get("role", "assistant"),
        "content": raw_msg.get("content"),
    }

    finish = str(raw.get("finish_reason", "stop") or "stop")
    finish = _FINISH_REASON_MAP.get(finish, finish)

    return ChatCompletionChoice(
        index=raw.get("index", idx),
        message=Message.model_validate(msg_payload),
        finish_reason=finish,
    )


def _parse_usage(data: dict[str, Any]) -> Usage | None:
    raw = data.get("usage")
    if not isinstance(raw, dict):
        return None
    prompt = int(raw.get("prompt_tokens", 0) or 0)
    completion = int(raw.get("completion_tokens", 0) or 0)
    total = int(raw.get("total_tokens", 0) or 0)
    if total == 0:
        total = prompt + completion
    return Usage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=total,
    )


def _perplexity_response_to_openai(data: dict[str, Any]) -> ChatCompletionResponse:
    if not isinstance(data, dict):
        raise ProviderError("Perplexity returned a non-object JSON response.")

    choices_raw = data.get("choices")
    if not isinstance(choices_raw, list) or not choices_raw:
        raise ProviderError("Perplexity returned no choices.")

    choices = [
        _parse_choice(i, c)
        for i, c in enumerate(choices_raw)
        if isinstance(c, dict)
    ]
    if not choices:
        raise ProviderError("Perplexity choices could not be parsed.")

    resp = ChatCompletionResponse(
        id=str(data.get("id", "")),
        created=int(data.get("created") or int(time.time())),
        model=str(data.get("model", "")),
        choices=choices,
        usage=_parse_usage(data),
    )

    citations = data.get("citations")
    if isinstance(citations, list):
        resp.citations = citations  # type: ignore[attr-defined]

    search_results = data.get("search_results")
    if isinstance(search_results, list):
        resp.search_results = search_results  # type: ignore[attr-defined]

    images = data.get("images")
    if isinstance(images, list):
        resp.images = images  # type: ignore[attr-defined]

    related = data.get("related_questions")
    if isinstance(related, list):
        resp.related_questions = related  # type: ignore[attr-defined]

    return resp


class PerplexityProvider(BaseLLMProvider):
    """
    Perplexity Sonar Chat Completions API.

    Endpoint: ``/chat/completions`` (OpenAI-compatible alias).
    Perplexity-specific response fields (``citations``, ``search_results``,
    ``images``, ``related_questions``) are forwarded as extra attributes on
    the :class:`ChatCompletionResponse`.
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        api_key: str | SecretStr,
        *,
        timeout: httpx.Timeout | float | None = None,
        base_url: str = "https://api.perplexity.ai",
    ) -> None:
        super().__init__(client, timeout=timeout)
        self._api_key = _secret_value(api_key)
        self._base_url = base_url.rstrip("/")

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    async def chat_completion(
        self,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        if request.stream:
            raise ProviderError(
                "PerplexityProvider does not support stream=true; "
                "use a non-streaming request.",
            )
        if (request.n or 1) != 1:
            raise ProviderError("PerplexityProvider only supports n=1.")
        if request.tools:
            raise ProviderError(
                "PerplexityProvider does not support tool calling.",
            )

        url = f"{self._base_url}/chat/completions"
        payload = _build_perplexity_body(request)
        response = await self._http_request(
            "POST",
            url,
            headers=self._headers(),
            json=payload,
        )
        data = response.json()
        return _perplexity_response_to_openai(data)

    async def health_check(self) -> bool:
        url = f"{self._base_url}/chat/completions"
        try:
            await self._http_request(
                "POST",
                url,
                headers=self._headers(),
                json={
                    "model": "sonar",
                    "messages": [{"role": "user", "content": "ping"}],
                    "max_tokens": 1,
                },
            )
        except ProviderError:
            return False
        return True
