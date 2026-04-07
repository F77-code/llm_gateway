from __future__ import annotations

from fastapi import Header

from app.config import get_settings
from app.exceptions import AuthenticationError


def get_api_key(authorization: str = Header()) -> str:
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise AuthenticationError(
            message="Invalid Authorization header, expected Bearer token",
            code="invalid_authorization_header",
        )

    api_key = token.strip()
    configured = get_settings().default_api_key
    if configured is not None and api_key != configured.get_secret_value():
        raise AuthenticationError(
            message="Invalid gateway API key",
            code="invalid_api_key",
        )
    return api_key
