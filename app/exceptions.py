from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class AppError(Exception):
    message: str
    status_code: int
    error_type: str
    code: str | None = None
    context: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.message


class ProviderError(AppError):
    def __init__(
        self,
        message: str = "Provider request failed",
        *,
        code: str | None = "provider_error",
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=502,
            error_type="provider_error",
            code=code,
            context=context or {},
        )


class RateLimitExceeded(AppError):
    def __init__(
        self,
        message: str = "Rate limit exceeded",
        *,
        code: str | None = "rate_limit_exceeded",
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=429,
            error_type="rate_limit_error",
            code=code,
            context=context or {},
        )


class ModelNotFound(AppError):
    def __init__(
        self,
        model: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=f"Unknown model: {model}",
            status_code=400,
            error_type="invalid_request_error",
            code="model_not_found",
            context={"model": model, **(context or {})},
        )


class AuthenticationError(AppError):
    def __init__(
        self,
        message: str = "Invalid authentication credentials",
        *,
        code: str | None = "authentication_error",
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=401,
            error_type="authentication_error",
            code=code,
            context=context or {},
        )


class BadRequestError(AppError):
    def __init__(
        self,
        message: str = "Bad request",
        *,
        code: str | None = "bad_request",
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=400,
            error_type="invalid_request_error",
            code=code,
            context=context or {},
        )


class ServiceUnavailableError(AppError):
    def __init__(
        self,
        message: str = "Service temporarily unavailable",
        *,
        code: str | None = "service_unavailable",
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=503,
            error_type="service_unavailable_error",
            code=code,
            context=context or {},
        )
