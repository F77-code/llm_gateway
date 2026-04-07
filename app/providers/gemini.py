from __future__ import annotations

import base64
import binascii
import json
import re
import time
import uuid
from typing import Any
from urllib.parse import quote

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

DEFAULT_MAX_OUTPUT_TOKENS = 4096
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

_DATA_URL_RE = re.compile(
    r"^data:(?P<mime>[\w/+.-]+);base64,(?P<b64>.+)$",
    re.IGNORECASE | re.DOTALL,
)

_FINISH_REASON_MAP: dict[str, str] = {
    "STOP": "stop",
    "MAX_TOKENS": "length",
    "SAFETY": "content_filter",
    "BLOCKLIST": "content_filter",
    "PROHIBITED_CONTENT": "content_filter",
    "SPII": "content_filter",
    "RECITATION": "content_filter",
    "OTHER": "stop",
    "FINISH_REASON_UNSPECIFIED": "stop",
    "MALFORMED_FUNCTION_CALL": "stop",
    "NO_IMAGE": "content_filter",
    "IMAGE_SAFETY": "content_filter",
    "IMAGE_PROHIBITED_CONTENT": "content_filter",
    "IMAGE_OTHER": "stop",
}


def _secret_value(key: str | SecretStr) -> str:
    if isinstance(key, SecretStr):
        return key.get_secret_value()
    return key


def _normalize_role(role: str) -> str:
    return (role or "").strip().lower()


def _strip_models_prefix(model: str) -> str:
    m = model.strip()
    if m.startswith("models/"):
        return m[7:]
    return m


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


def _openai_image_part_to_gemini(block: dict[str, Any]) -> dict[str, Any] | None:
    src = block.get("image_url")
    src = src if isinstance(src, dict) else {}
    url = src.get("url")
    if not url or not isinstance(url, str):
        return None
    m = _DATA_URL_RE.match(url.strip())
    if m:
        mime = m.group("mime")
        b64 = m.group("b64").strip()
        try:
            raw = base64.b64decode(b64, validate=False)
        except (binascii.Error, ValueError):
            return {"text": "[invalid image data]"}
        return {
            "inlineData": {
                "mimeType": mime or "image/png",
                "data": base64.b64encode(raw).decode("ascii"),
            },
        }
    if url.startswith(("http://", "https://")):
        mime = str(src.get("mime_type") or "image/jpeg")
        return {"fileData": {"mimeType": mime, "fileUri": url}}
    return None


def _openai_user_parts(content: str | list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    if content is None:
        return [{"text": ""}]
    if isinstance(content, str):
        return [{"text": content}] if content else [{"text": ""}]
    parts: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            parts.append({"text": str(block.get("text", ""))})
        elif btype == "image_url":
            img = _openai_image_part_to_gemini(block)
            if img:
                parts.append(img)
        else:
            parts.append({"text": json.dumps(block, ensure_ascii=False)})
    return parts if parts else [{"text": ""}]


def _model_parts_from_assistant(msg: Message) -> list[dict[str, Any]]:
    parts: list[dict[str, Any]] = []
    text = _text_from_openai_content(msg.content)
    if text:
        parts.append({"text": text})
    if msg.tool_calls:
        for tc in msg.tool_calls:
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function")
            fn = fn if isinstance(fn, dict) else {}
            name = str(fn.get("name", ""))
            args = _parse_tool_arguments(fn.get("arguments"))
            parts.append({"functionCall": {"name": name, "args": args}})
    if not parts:
        parts.append({"text": ""})
    return parts


def _merge_consecutive_contents(
    contents: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not contents:
        return contents
    out: list[dict[str, Any]] = [{"role": contents[0]["role"], "parts": list(contents[0]["parts"])}]
    for c in contents[1:]:
        if c["role"] == out[-1]["role"]:
            out[-1]["parts"] = list(out[-1]["parts"]) + list(c["parts"])
        else:
            out.append({"role": c["role"], "parts": list(c["parts"])})
    return out


def _openai_messages_to_gemini_contents(
    messages: list[Message],
) -> tuple[dict[str, Any] | None, list[dict[str, Any]]]:
    system_bits: list[str] = []
    conv: list[Message] = []
    for m in messages:
        role = _normalize_role(str(m.role))
        if role in ("system", "developer"):
            t = _text_from_openai_content(m.content)
            if t:
                system_bits.append(t)
        else:
            conv.append(m)

    system_instruction: dict[str, Any] | None = None
    if system_bits:
        system_instruction = {
            "parts": [{"text": "\n\n".join(system_bits)}],
        }

    tool_name_by_call_id: dict[str, str] = {}
    contents: list[dict[str, Any]] = []
    pending_fr: list[dict[str, Any]] = []

    def flush_fr() -> None:
        if pending_fr:
            contents.append({"role": "user", "parts": list(pending_fr)})
            pending_fr.clear()

    for m in conv:
        role = _normalize_role(str(m.role))
        if role == "tool":
            tid = str(m.tool_call_id or "")
            name = tool_name_by_call_id.get(tid, "unknown")
            body = _text_from_openai_content(m.content)
            pending_fr.append(
                {
                    "functionResponse": {
                        "name": name,
                        "response": {"result": body},
                    },
                },
            )
            continue

        flush_fr()

        if role == "user":
            contents.append({"role": "user", "parts": _openai_user_parts(m.content)})
        elif role == "assistant":
            if m.tool_calls:
                for tc in m.tool_calls:
                    if not isinstance(tc, dict):
                        continue
                    tid = str(tc.get("id", ""))
                    fn = tc.get("function")
                    fn = fn if isinstance(fn, dict) else {}
                    nm = str(fn.get("name", ""))
                    if tid and nm:
                        tool_name_by_call_id[tid] = nm
            contents.append({"role": "model", "parts": _model_parts_from_assistant(m)})
        else:
            contents.append(
                {
                    "role": "user",
                    "parts": [{"text": _text_from_openai_content(m.content) or "(message)"}],
                },
            )

    flush_fr()

    merged = _merge_consecutive_contents(contents)
    if merged and merged[0]["role"] != "user":
        merged.insert(0, {"role": "user", "parts": [{"text": "Please continue the conversation."}]})
    return system_instruction, merged


def _openai_tools_to_gemini(tools: list[dict[str, Any]]) -> list[dict[str, Any]] | None:
    decls: list[dict[str, Any]] = []
    for t in tools:
        if not isinstance(t, dict) or t.get("type") != "function":
            continue
        fn = t.get("function")
        fn = fn if isinstance(fn, dict) else {}
        name = str(fn.get("name", ""))
        if not name:
            continue
        params = fn.get("parameters")
        if not isinstance(params, dict):
            params = {"type": "object", "properties": {}}
        decls.append(
            {
                "name": name,
                "description": str(fn.get("description", "")),
                "parameters": params,
            },
        )
    if not decls:
        return None
    return [{"functionDeclarations": decls}]


def _map_tool_config(
    tool_choice: str | dict[str, Any] | None,
) -> dict[str, Any] | None:
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        low = tool_choice.lower()
        if low == "none":
            return {"functionCallingConfig": {"mode": "NONE"}}
        if low == "auto":
            return {"functionCallingConfig": {"mode": "AUTO"}}
        if low == "required":
            return {"functionCallingConfig": {"mode": "ANY"}}
        return {"functionCallingConfig": {"mode": "AUTO"}}
    if isinstance(tool_choice, dict):
        t = tool_choice.get("type")
        if t == "function":
            fn = tool_choice.get("function")
            fn = fn if isinstance(fn, dict) else {}
            name = fn.get("name")
            if name:
                return {
                    "functionCallingConfig": {
                        "mode": "ANY",
                        "allowedFunctionNames": [str(name)],
                    },
                }
        return {"functionCallingConfig": {"mode": "AUTO"}}
    return {"functionCallingConfig": {"mode": "AUTO"}}


def _build_gemini_body(request: ChatCompletionRequest) -> dict[str, Any]:
    sys_inst, contents = _openai_messages_to_gemini_contents(request.messages)
    if not contents:
        raise ProviderError("Gemini requires at least one content turn.")

    max_out = (
        request.max_tokens or request.max_completion_tokens or DEFAULT_MAX_OUTPUT_TOKENS
    )
    gen: dict[str, Any] = {"maxOutputTokens": max_out}
    if request.temperature is not None:
        gen["temperature"] = request.temperature
    if request.top_p is not None:
        gen["topP"] = request.top_p
    if request.stop is not None:
        gen["stopSequences"] = (
            [request.stop] if isinstance(request.stop, str) else list(request.stop)
        )
    n = request.n or 1
    if n > 1:
        gen["candidateCount"] = min(n, 8)

    body: dict[str, Any] = {
        "contents": contents,
        "generationConfig": gen,
    }
    if sys_inst:
        body["systemInstruction"] = sys_inst
    gt = _openai_tools_to_gemini(request.tools or [])
    if gt:
        body["tools"] = gt
    tc = _map_tool_config(request.tool_choice)
    if tc is not None:
        body["toolConfig"] = tc
    return body


def _parts_from_candidate_content(content: Any) -> list[dict[str, Any]]:
    if not isinstance(content, dict):
        return []
    parts = content.get("parts")
    if not isinstance(parts, list):
        return []
    return [p for p in parts if isinstance(p, dict)]


def _openai_finish_reason(
    gemini_reason: str | None,
    has_function_call: bool,
) -> str:
    if has_function_call:
        return "tool_calls"
    if not gemini_reason:
        return "stop"
    return _FINISH_REASON_MAP.get(gemini_reason.upper(), "stop")


def _choice_from_candidate(index: int, cand: dict[str, Any]) -> ChatCompletionChoice:
    content = cand.get("content")
    parts = _parts_from_candidate_content(content)

    text_chunks: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for p in parts:
        if "text" in p:
            text_chunks.append(str(p.get("text", "")))
        fc = p.get("functionCall")
        if isinstance(fc, dict):
            name = str(fc.get("name", ""))
            args = fc.get("args")
            if not isinstance(args, dict):
                args = {}
            tool_calls.append(
                {
                    "id": f"call_{uuid.uuid4().hex[:12]}",
                    "type": "function",
                    "function": {
                        "name": name,
                        "arguments": json.dumps(args, ensure_ascii=False),
                    },
                },
            )

    assistant_text = "\n".join(text_chunks).strip()
    has_fc = bool(tool_calls)
    raw_finish = cand.get("finishReason")
    raw_str = str(raw_finish) if raw_finish is not None else None
    finish = _openai_finish_reason(raw_str, has_fc)

    msg_payload: dict[str, Any] = {
        "role": "assistant",
        "content": assistant_text if assistant_text else None,
    }
    if tool_calls:
        msg_payload["tool_calls"] = tool_calls

    return ChatCompletionChoice(
        index=index,
        message=Message.model_validate(msg_payload),
        finish_reason=finish,
    )


def _usage_from_gemini(data: dict[str, Any]) -> Usage | None:
    um = data.get("usageMetadata")
    if not isinstance(um, dict):
        return None
    prompt = int(um.get("promptTokenCount", 0) or 0)
    candidates = int(um.get("candidatesTokenCount", 0) or 0)
    total = int(um.get("totalTokenCount", 0) or 0)
    if total == 0:
        total = prompt + candidates
    return Usage(
        prompt_tokens=prompt,
        completion_tokens=candidates,
        total_tokens=total,
    )


def _gemini_response_to_openai(
    data: dict[str, Any],
    request_model: str,
) -> ChatCompletionResponse:
    if not isinstance(data, dict):
        raise ProviderError("Gemini returned a non-object JSON response.")

    candidates = data.get("candidates")
    if not isinstance(candidates, list):
        candidates = []

    pf = data.get("promptFeedback")
    block_reason = None
    if isinstance(pf, dict):
        block_reason = pf.get("blockReason")

    if not candidates:
        if block_reason:
            choice = ChatCompletionChoice(
                index=0,
                message=Message.model_validate(
                    {
                        "role": "assistant",
                        "content": None,
                    },
                ),
                finish_reason="content_filter",
            )
            return ChatCompletionResponse(
                id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
                created=int(time.time()),
                model=request_model,
                choices=[choice],
                usage=_usage_from_gemini(data),
            )
        raise ProviderError("Gemini returned no candidates.")

    choices = [
        _choice_from_candidate(i, c)
        for i, c in enumerate(candidates)
        if isinstance(c, dict)
    ]
    if not choices:
        raise ProviderError("Gemini candidates could not be parsed.")

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex[:24]}",
        created=int(time.time()),
        model=request_model,
        choices=choices,
        usage=_usage_from_gemini(data),
    )


class GeminiProvider(BaseLLMProvider):
    """Google Gemini `generateContent` mapped to OpenAI chat types."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        api_key: str | SecretStr,
        *,
        timeout: httpx.Timeout | float | None = None,
        base_url: str = GEMINI_BASE_URL,
    ) -> None:
        super().__init__(client, timeout=timeout)
        self._api_key = _secret_value(api_key)
        self._base_url = base_url.rstrip("/")

    def _headers(self) -> dict[str, str]:
        return {
            "x-goog-api-key": self._api_key,
            "Content-Type": "application/json",
        }

    def _generate_url(self, model: str) -> str:
        mid = quote(_strip_models_prefix(model), safe="")
        return f"{self._base_url}/models/{mid}:generateContent"

    async def chat_completion(
        self,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        if request.stream:
            raise ProviderError(
                "GeminiProvider does not support stream=true; use a non-streaming request.",
            )

        url = self._generate_url(request.model)
        payload = _build_gemini_body(request)
        response = await self._http_request(
            "POST",
            url,
            headers=self._headers(),
            json=payload,
        )
        data = response.json()
        return _gemini_response_to_openai(data, request.model)

    async def health_check(self) -> bool:
        url = f"{self._base_url}/models?pageSize=1"
        try:
            await self._http_request("GET", url, headers=self._headers())
        except ProviderError:
            return False
        return True
