from __future__ import annotations

from typing import Annotated

from fastapi import Header

from app.auth import authenticate_bearer
from app.config import get_settings


def get_client_id(
    authorization: Annotated[list[str] | None, Header()] = None,
) -> str:
    # Duplicate headers are ambiguous and must never select one valid credential.
    header = authorization[0] if authorization and len(authorization) == 1 else None
    return authenticate_bearer(header, get_settings().gateway_clients)
