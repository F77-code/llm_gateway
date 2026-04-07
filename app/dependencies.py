from __future__ import annotations

from fastapi import Header, HTTPException, status

from app.config import get_settings


def get_api_key(authorization: str = Header()) -> str:
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header, expected Bearer token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    api_key = token.strip()
    configured = get_settings().default_api_key
    if configured is not None and api_key != configured.get_secret_value():
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid gateway API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return api_key
