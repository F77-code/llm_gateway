from __future__ import annotations

import hashlib
import hmac
import re
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator

from app.exceptions import AuthenticationError

if TYPE_CHECKING:
    from collections.abc import Sequence


MAX_BEARER_TOKEN_LENGTH = 1024
_MAX_AUTHORIZATION_HEADER_LENGTH = 4096
_BEARER_HEADER = re.compile(r"(?i:Bearer) +([A-Za-z0-9._~+/-]+=*)")


class ClientRegistration(BaseModel):
    """A stable identity and one hashed bearer credential; repeat IDs for rotation."""

    model_config = ConfigDict(extra="forbid", frozen=True, hide_input_in_errors=True)

    client_id: str = Field(
        pattern=r"^[a-zA-Z0-9_-]{1,64}$", min_length=1, max_length=64
    )
    key_sha256: str = Field(
        pattern=r"^[a-fA-F0-9]{64}$", min_length=64, max_length=64, repr=False
    )
    enabled: StrictBool = True

    @field_validator("key_sha256")
    @classmethod
    def normalize_key_hash(cls, value: str) -> str:
        return value.lower()


def authenticate_bearer(
    authorization: str | None,
    registrations: Sequence[ClientRegistration],
) -> str:
    """Return only the registered client ID, using constant-time hash comparisons."""
    if authorization is None or len(authorization) > _MAX_AUTHORIZATION_HEADER_LENGTH:
        raise AuthenticationError("Invalid gateway credentials", code="invalid_api_key")
    match = _BEARER_HEADER.fullmatch(authorization)
    if match is None:
        raise AuthenticationError("Invalid gateway credentials", code="invalid_api_key")

    token = match.group(1)
    if len(token) > MAX_BEARER_TOKEN_LENGTH:
        raise AuthenticationError("Invalid gateway credentials", code="invalid_api_key")
    key_hash = hashlib.sha256(token.encode("ascii")).hexdigest()

    client_id: str | None = None
    for registration in registrations:
        matches = hmac.compare_digest(key_hash, registration.key_sha256)
        if matches and registration.enabled:
            client_id = registration.client_id
    if client_id is None:
        raise AuthenticationError("Invalid gateway credentials", code="invalid_api_key")
    return client_id
