from __future__ import annotations

import hashlib
from enum import StrEnum
from functools import lru_cache
from typing import ClassVar

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.auth import ClientRegistration


class Provider(StrEnum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GEMINI = "gemini"
    DEEPSEEK = "deepseek"
    PERPLEXITY = "perplexity"
    XAI = "xai"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        hide_input_in_errors=True,
    )

    openai_api_key: SecretStr | None = None
    anthropic_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None
    deepseek_api_key: SecretStr | None = None
    perplexity_api_key: SecretStr | None = None
    xai_api_key: SecretStr | None = None

    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL",
    )
    rate_limit_rpm: int = Field(
        default=60,
        ge=1,
        description="Max chat completion requests per client per minute (sliding window)",
    )
    gateway_clients: list[ClientRegistration] = Field(
        default_factory=list,
        description="Client registrations as JSON: client_id, key_sha256 and enabled",
        repr=False,
    )

    @field_validator("gateway_clients")
    @classmethod
    def reject_duplicate_key_hashes(
        cls, clients: list[ClientRegistration]
    ) -> list[ClientRegistration]:
        hashes = [client.key_sha256 for client in clients]
        if len(hashes) != len(set(hashes)):
            raise ValueError("Duplicate client key hashes are not allowed")
        identity_hashes = {
            hashlib.sha256(client.client_id.encode("ascii")).hexdigest()
            for client in clients
        }
        if identity_hashes.intersection(hashes):
            raise ValueError("Client identifiers must not also be bearer credentials")
        return clients

    provider_by_model: ClassVar[dict[str, Provider]] = {
        "deepseek-chat": Provider.DEEPSEEK,
        "deepseek-reasoner": Provider.DEEPSEEK,
        "gemini-2.5-pro": Provider.GEMINI,
        "gemini-2.5-flash": Provider.GEMINI,
        "gemini-2.5-flash-lite": Provider.GEMINI,
        "gemini-2.0-flash": Provider.GEMINI,
        "gemini-2.0-flash-lite": Provider.GEMINI,
        "grok-4-1-fast-non-reasoning": Provider.XAI,
        "grok-4-1-fast-reasoning": Provider.XAI,
        "sonar": Provider.PERPLEXITY,
        "sonar-pro": Provider.PERPLEXITY,
        "sonar-reasoning-pro": Provider.PERPLEXITY,
        "sonar-deep-research": Provider.PERPLEXITY,
        "claude-opus-4-6": Provider.ANTHROPIC,
        "claude-sonnet-4-5-20250929": Provider.ANTHROPIC,
        "claude-haiku-4-5-20251001": Provider.ANTHROPIC,
        "gpt-5.4": Provider.OPENAI,
        "gpt-5.3-chat-latest": Provider.OPENAI,
        "gpt-5.2": Provider.OPENAI,
        "gpt-5.1": Provider.OPENAI,
        "gpt-5": Provider.OPENAI,
        "gpt-5-mini": Provider.OPENAI,
        "gpt-5-nano": Provider.OPENAI,
        "gpt-4.1-mini": Provider.OPENAI,
    }


@lru_cache
def get_settings() -> Settings:
    return Settings()
