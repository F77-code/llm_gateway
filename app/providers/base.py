from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import httpx

from app.models.chat_completion import ChatCompletionRequest, ChatCompletionResponse


class ProviderError(Exception):
    """Base error for upstream LLM / HTTP failures."""


class ProviderTimeoutError(ProviderError):
    """Request exceeded httpx timeout (connect, read, write, pool)."""


class ProviderHTTPError(ProviderError):
    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        body_preview: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body_preview = body_preview


class ProviderUnauthorizedError(ProviderHTTPError):
    """HTTP 401 — invalid or missing API credentials."""


class ProviderRateLimitError(ProviderHTTPError):
    """HTTP 429 — rate limited by upstream."""


class ProviderServerError(ProviderHTTPError):
    """HTTP 5xx — upstream server error."""


class ProviderRequestError(ProviderError):
    """Network / transport failure (no HTTP response or not classified elsewhere)."""


def _body_preview(response: httpx.Response, limit: int = 512) -> str | None:
    try:
        text = response.text
    except Exception:
        return None
    text = text.strip()
    if not text:
        return None
    return text if len(text) <= limit else f"{text[:limit]}…"


class BaseLLMProvider(ABC):
    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        timeout: httpx.Timeout | float | None = None,
    ) -> None:
        self._client = client
        self._timeout: httpx.Timeout | float | None = timeout

    @abstractmethod
    async def chat_completion(
        self,
        request: ChatCompletionRequest,
    ) -> ChatCompletionResponse:
        """Perform a chat completion against the provider API."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Return True if the provider endpoint is reachable (provider-specific)."""

    def _effective_timeout(self) -> httpx.Timeout | float | None:
        return self._timeout

    async def _http_request(
        self,
        method: str,
        url: str,
        *,
        timeout: httpx.Timeout | float | None = None,
        **kwargs: Any,
    ) -> httpx.Response:
        """
        Run an httpx request with shared timeout defaults and error translation.

        Subclasses should use this for outbound calls so timeouts and transport
        failures are mapped to `Provider*` errors consistently.
        """
        t = self._effective_timeout() if timeout is None else timeout
        try:
            response = await self._client.request(method, url, timeout=t, **kwargs)
            response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise ProviderTimeoutError(str(exc) or "upstream request timed out") from exc
        except httpx.HTTPStatusError as exc:
            prev = exc.response
            preview = _body_preview(prev)
            msg = f"HTTP {prev.status_code}"
            if preview:
                msg = f"{msg}: {preview}"
            code = prev.status_code
            kwargs: dict[str, Any] = {
                "message": msg,
                "status_code": code,
                "body_preview": preview,
            }
            if code == 401:
                raise ProviderUnauthorizedError(**kwargs) from exc
            if code == 429:
                raise ProviderRateLimitError(**kwargs) from exc
            if code >= 500:
                raise ProviderServerError(**kwargs) from exc
            raise ProviderHTTPError(**kwargs) from exc
        except httpx.RequestError as exc:
            raise ProviderRequestError(str(exc) or "upstream request failed") from exc
        else:
            return response
