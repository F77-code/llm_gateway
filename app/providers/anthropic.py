from __future__ import annotations

import json
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

ANTHROPIC_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 4096

_STOP_REASON_TO_OPENAI: dict[str, str] = {
    "end_turn": "stop",
    "max_tokens": "length",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
    "pause_turn": "stop",
    "refusal": "content_filter",
}


def _secret_value(key: str | SecretStr) -> str:
    if isinstance(key, SecretStr):
        return key.get_secret_value()
    return key


def _normalize_role(role: str) -> str:
    return (role or "").strip().lower()


def _text_from_openai_content(content: str | list[dict[str, Any]] | None) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts: list[str] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            parts.append(str(block.get("text", "")))
        elif "text" in block:
            parts.append(str(block["text"]))
    return "\n".join(parts).strip()


def _openai_user_content_to_anthropic(
    content: str | list[dict[str, Any]] | None,
) -> str | list[dict[str, Any]]:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    blocks: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            blocks.append({"type": "text", "text": str(block.get("text", ""))})
        elif btype == "image_url":
            src = (block.get("image_url") or {}) if isinstance(block.get("image_url"), dict) else {}
            url = src.get("url")
            if url:
                blocks.append(
                    {
                        "type": "image",
                        "source": {"type": "url", "url": url},
                    },
                )
        else:
            blocks.append(block)
    return blocks if blocks else ""


def _parse_tool_arguments(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _assistant_blocks(msg: Message) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    text = _text_from_openai_content(msg.content)
    if text:
        blocks.append({"type": "text", "text": text})
    if msg.tool_calls:
        for tc in msg.tool_calls:
            fn = tc.get("function") if isinstance(tc, dict) else None
            fn = fn if isinstance(fn, dict) else {}
            name = str(fn.get("name", ""))
            args = _parse_tool_arguments(fn.get("arguments"))
            blocks.append(
                {
                    "type": "tool_use",
                    "id": str(tc.get("id", "")),
                    "name": name,
                    "input": args,
                },
            )
    if not blocks:
        blocks.append({"type": "text", "text": ""})
    return blocks


def _content_to_blocks(content: str | list[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    return list(content)


def _merge_anthropic_turns(
    a: str | list[dict[str, Any]],
    b: str | list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return _content_to_blocks(a) + _content_to_blocks(b)


def _openai_messages_to_anthropic(
    messages: list[Message],
) -> tuple[str | None, list[dict[str, Any]]]:
    system_parts: list[str] = []
    conv: list[Message] = []
    for m in messages:
        role = _normalize_role(str(m.role))
        if role in ("system", "developer"):
            t = _text_from_openai_content(m.content)
            if t:
                system_parts.append(t)
        else:
            conv.append(m)

    system = "\n\n".join(system_parts) if system_parts else None

    anth: list[dict[str, Any]] = []
    pending_tool_results: list[dict[str, Any]] = []

    def flush_tools() -> None:
        nonlocal pending_tool_results
        if pending_tool_results:
            anth.append({"role": "user", "content": list(pending_tool_results)})
            pending_tool_results = []

    for m in conv:
        role = _normalize_role(str(m.role))
        if role == "tool":
            tid = m.tool_call_id or ""
            body = _text_from_openai_content(m.content) or ""
            pending_tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": tid,
                    "content": body,
                },
            )
            continue

        flush_tools()

        if role == "user":
            ac = _openai_user_content_to_anthropic(m.content)
            anth.append({"role": "user", "content": ac})
        elif role == "assistant":
            anth.append({"role": "assistant", "content": _assistant_blocks(m)})
        else:
            anth.append(
                {
                    "role": "user",
                    "content": _text_from_openai_content(m.content) or "(message)",
                },
            )

    flush_tools()

    merged: list[dict[str, Any]] = []
    for msg in anth:
        if merged and msg["role"] == merged[-1]["role"]:
            prev = merged[-1]
            prev["content"] = _merge_anthropic_turns(prev["content"], msg["content"])
        else:
            merged.append(dict(msg))

    if merged and merged[0]["role"] != "user":
        merged.insert(0, {"role": "user", "content": "Please continue the conversation."})

    return system, merged


def _openai_tools_to_anthropic(tools: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        if t.get("type") != "function":
            continue
        fn = t.get("function")
        fn = fn if isinstance(fn, dict) else {}
        name = str(fn.get("name", ""))
        if not name:
            continue
        params = fn.get("parameters")
        if not isinstance(params, dict):
            params = {"type": "object", "properties": {}}
        out.append(
            {
                "name": name,
                "description": fn.get("description") or "",
                "input_schema": params,
            },
        )
    return out


def _map_tool_choice(
    tool_choice: str | dict[str, Any] | None,
) -> dict[str, Any] | str | None:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        low = tool_choice.lower()
        if low == "auto":
            return {"type": "auto"}
        if low == "none":
            return {"type": "none"}
        return {"type": "auto"}
    if isinstance(tool_choice, dict):
        t = tool_choice.get("type")
        if t == "function":
            fn = tool_choice.get("function")
            fn = fn if isinstance(fn, dict) else {}
            name = fn.get("name")
            if name:
                return {"type": "tool", "name": str(name)}
        if t in ("auto", "none", "tool", "any"):
            return tool_choice
    return {"type": "auto"}


def _build_anthropic_body(request: ChatCompletionRequest) -> dict[str, Any]:
    max_out = request.max_tokens or request.max_completion_tokens or DEFAULT_MAX_TOKENS
    system, messages = _openai_messages_to_anthropic(request.messages)
    if not messages:
        raise ProviderError("Anthropic requires at least one user/assistant message.")

    body: dict[str, Any] = {
        "model": request.model,
        "max_tokens": max_out,
        "messages": messages,
    }
    if system:
        body["system"] = system
    if request.temperature is not None:
        body["temperature"] = request.temperature
    if request.top_p is not None:
        body["top_p"] = request.top_p
    if request.stop is not None:
        body["stop_sequences"] = (
            [request.stop] if isinstance(request.stop, str) else list(request.stop)
        )
    if request.tools:
        at = _openai_tools_to_anthropic(request.tools)
        if at:
            body["tools"] = at
    tc = _map_tool_choice(request.tool_choice)
    if tc is not None:
        body["tool_choice"] = tc
    if request.user:
        body["metadata"] = {"user_id": str(request.user)}
    return body


def _anthropic_response_to_openai(data: dict[str, Any]) -> ChatCompletionResponse:
    msg_id = str(data.get("id", ""))
    model = str(data.get("model", ""))
    content = data.get("content")
    blocks = content if isinstance(content, list) else []

    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            text_parts.append(str(block.get("text", "")))
        elif btype == "tool_use":
            tid = str(block.get("id", ""))
            name = str(block.get("name", ""))
            inp = block.get("input")
            args = json.dumps(inp if isinstance(inp, dict) else {}, ensure_ascii=False)
            tool_calls.append(
                {
                    "id": tid,
                    "type": "function",
                    "function": {"name": name, "arguments": args},
                },
            )
        elif btype == "refusal":
            text_parts.append(str(block.get("refusal", "")))

    assistant_text = "\n".join(text_parts).strip()
    stop_reason = data.get("stop_reason")
    finish = _STOP_REASON_TO_OPENAI.get(
        str(stop_reason).lower() if stop_reason is not None else "",
        "stop",
    )

    message_payload: dict[str, Any] = {
        "role": "assistant",
        "content": assistant_text if assistant_text else None,
    }
    if tool_calls:
        message_payload["tool_calls"] = tool_calls

    usage_out: Usage | None = None
    raw_usage = data.get("usage")
    if isinstance(raw_usage, dict):
        inp = int(raw_usage.get("input_tokens", 0) or 0)
        out_t = int(raw_usage.get("output_tokens", 0) or 0)
        usage_out = Usage(
            prompt_tokens=inp,
            completion_tokens=out_t,
            total_tokens=inp + out_t,
        )

    choice = ChatCompletionChoice(
        index=0,
        message=Message.model_validate(message_payload),
        finish_reason=finish,
    )

    return ChatCompletionResponse(
        id=msg_id,
        created=int(time.time()),
        model=model,
        choices=[choice],
        usage=usage_out,
    )


class AnthropicProvider(BaseLLMProvider):
    """Anthropic Messages API (`/v1/messages`) mapped to OpenAI chat types."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        api_key: str | SecretStr,
        *,
        timeout: httpx.Timeout | float | None = None,
        base_url: str = "https://api.anthropic.com/v1",
    ) -> None:
        super().__init__(client, timeout=timeout)
        self._api_key = _secret_value(api_key)
        self._base_url = base_url.rstrip("/")

    def _headers(self) -> dict[str, str]:
        return {
            "x-api-key": self._api_key,
            "anthropic-version": ANTHROPIC_VERSION,
            "Content-Type": "application/json",
        }

    async def chat_completion(
        self,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        if request.stream:
            raise ProviderError(
                "AnthropicProvider does not support stream=true; use a non-streaming request.",
            )
        if (request.n or 1) != 1:
            raise ProviderError("AnthropicProvider only supports n=1.")

        url = f"{self._base_url}/messages"
        payload = _build_anthropic_body(request)
        response = await self._http_request(
            "POST",
            url,
            headers=self._headers(),
            json=payload,
        )
        data = response.json()
        if not isinstance(data, dict):
            raise ProviderError("Anthropic returned a non-object JSON response.")
        return _anthropic_response_to_openai(data)

    async def health_check(self) -> bool:
        url = f"{self._base_url}/models"
        try:
            await self._http_request(
                "GET",
                url,
                headers=self._headers(),
            )
        except ProviderError:
            return False
        return True
