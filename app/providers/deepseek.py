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

DEEPSEEK_MODELS: tuple[str, ...] = (
    "deepseek-chat",
    "deepseek-reasoner",
)

_FINISH_REASON_MAP: dict[str, str] = {
    "stop": "stop",
    "length": "length",
    "tool_calls": "tool_calls",
    "content_filter": "content_filter",
    "insufficient_system_resource": "stop",
}


def _secret_value(key: str | SecretStr) -> str:
    if isinstance(key, SecretStr):
        return key.get_secret_value()
    return key


def _serialize_message(m: Message) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": m.role}
    if m.content is not None:
        msg["content"] = m.content
    if m.name is not None:
        msg["name"] = m.name
    if m.tool_calls is not None:
        msg["tool_calls"] = m.tool_calls
    if m.tool_call_id is not None:
        msg["tool_call_id"] = m.tool_call_id
    return msg


def _build_deepseek_body(request: ChatCompletionRequest) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": request.model,
        "messages": [_serialize_message(m) for m in request.messages],
    }
    if request.temperature is not None:
        body["temperature"] = request.temperature
    if request.max_tokens is not None:
        body["max_tokens"] = request.max_tokens
    if request.max_completion_tokens is not None:
        body["max_completion_tokens"] = request.max_completion_tokens
    if request.top_p is not None:
        body["top_p"] = request.top_p
    if request.stop is not None:
        body["stop"] = request.stop
    if request.frequency_penalty is not None:
        body["frequency_penalty"] = request.frequency_penalty
    if request.presence_penalty is not None:
        body["presence_penalty"] = request.presence_penalty
    if request.tools:
        body["tools"] = request.tools
    if request.tool_choice is not None:
        body["tool_choice"] = request.tool_choice
    if request.response_format is not None:
        body["response_format"] = request.response_format
    if request.logit_bias is not None:
        body["logit_bias"] = request.logit_bias

    extra = request.model_extra or {}
    thinking = extra.get("thinking")
    if thinking is not None:
        body["thinking"] = thinking

    return body


def _parse_tool_calls(raw_calls: Any) -> list[dict[str, Any]] | None:
    if not isinstance(raw_calls, list) or not raw_calls:
        return None
    calls: list[dict[str, Any]] = []
    for tc in raw_calls:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function")
        fn = fn if isinstance(fn, dict) else {}
        calls.append(
            {
                "id": str(tc.get("id", "")),
                "type": "function",
                "function": {
                    "name": str(fn.get("name", "")),
                    "arguments": str(fn.get("arguments", "{}")),
                },
            },
        )
    return calls if calls else None


def _parse_choice(idx: int, raw: dict[str, Any]) -> ChatCompletionChoice:
    raw_msg = raw.get("message")
    if not isinstance(raw_msg, dict):
        raw_msg = {"role": "assistant", "content": None}

    msg_payload: dict[str, Any] = {
        "role": raw_msg.get("role", "assistant"),
        "content": raw_msg.get("content"),
    }

    reasoning = raw_msg.get("reasoning_content")
    if reasoning:
        msg_payload["reasoning_content"] = reasoning

    tool_calls = _parse_tool_calls(raw_msg.get("tool_calls"))
    if tool_calls:
        msg_payload["tool_calls"] = tool_calls

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


def _deepseek_response_to_openai(data: dict[str, Any]) -> ChatCompletionResponse:
    if not isinstance(data, dict):
        raise ProviderError("DeepSeek returned a non-object JSON response.")

    choices_raw = data.get("choices")
    if not isinstance(choices_raw, list) or not choices_raw:
        raise ProviderError("DeepSeek returned no choices.")

    choices = [
        _parse_choice(i, c)
        for i, c in enumerate(choices_raw)
        if isinstance(c, dict)
    ]
    if not choices:
        raise ProviderError("DeepSeek choices could not be parsed.")

    return ChatCompletionResponse(
        id=str(data.get("id", "")),
        created=int(data.get("created") or int(time.time())),
        model=str(data.get("model", "")),
        choices=choices,
        usage=_parse_usage(data),
        system_fingerprint=data.get("system_fingerprint"),
    )


class DeepSeekProvider(BaseLLMProvider):
    """
    DeepSeek Chat Completions API (`/v1/chat/completions`).

    Handles both ``deepseek-chat`` and ``deepseek-reasoner``.
    For reasoner models the response ``reasoning_content`` field is forwarded
    as an extra attribute on the assistant :class:`Message`.
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        api_key: str | SecretStr,
        *,
        timeout: httpx.Timeout | float | None = None,
        base_url: str = "https://api.deepseek.com/v1",
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
                "DeepSeekProvider does not support stream=true; use a non-streaming request.",
            )

        url = f"{self._base_url}/chat/completions"
        payload = _build_deepseek_body(request)
        response = await self._http_request(
            "POST",
            url,
            headers=self._headers(),
            json=payload,
        )
        data = response.json()
        return _deepseek_response_to_openai(data)

    async def health_check(self) -> bool:
        url = f"{self._base_url}/models"
        try:
            await self._http_request(
                "GET",
                url,
                headers=self._headers(),
                params={"limit": 1},
            )
        except ProviderError:
            return False
        return True
