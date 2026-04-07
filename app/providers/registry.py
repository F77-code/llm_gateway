"""Shared provider registry (`model` → upstream `BaseLLMProvider` instance)."""

from __future__ import annotations

import httpx

from app.config import Provider, Settings, get_settings
from app.providers.anthropic import AnthropicProvider
from app.providers.base import BaseLLMProvider, ProviderError
from app.providers.deepseek import DeepSeekProvider
from app.providers.gemini import GeminiProvider
from app.providers.openai import OpenAIProvider
from app.providers.perplexity import PerplexityProvider
from app.providers.xai import XAIProvider

# Gateway model id → other model ids to consider for cross-backend fallback.
FALLBACK_MODELS_BY_MODEL: dict[str, list[str]] = {}


class UnknownModelError(LookupError):
    """No `Provider` mapping exists for the requested model id."""

    def __init__(self, model: str) -> None:
        self.model = model
        super().__init__(f"Unknown model: {model!r}")


class ProviderRegistry:
    """
    Maps chat model names to shared `BaseLLMProvider` instances (one per `Provider`).

    Call `configure_registry()` from application lifespan after creating `httpx.AsyncClient`.
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        settings: Settings | None = None,
    ) -> None:
        self._client = client
        self._settings = settings or get_settings()
        self._by_backend: dict[Provider, BaseLLMProvider] = {}

    def _require_secret(self, value: object | None, name: str) -> None:
        if value is None:
            raise ProviderError(f"{name} is not set in the environment / settings.")

    def _build(self, backend: Provider) -> BaseLLMProvider:
        s = self._settings
        c = self._client
        if backend is Provider.OPENAI:
            self._require_secret(s.openai_api_key, "OPENAI_API_KEY")
            assert s.openai_api_key is not None
            return OpenAIProvider(c, s.openai_api_key)
        if backend is Provider.ANTHROPIC:
            self._require_secret(s.anthropic_api_key, "ANTHROPIC_API_KEY")
            assert s.anthropic_api_key is not None
            return AnthropicProvider(c, s.anthropic_api_key)
        if backend is Provider.GEMINI:
            self._require_secret(s.gemini_api_key, "GEMINI_API_KEY")
            assert s.gemini_api_key is not None
            return GeminiProvider(c, s.gemini_api_key)
        if backend is Provider.DEEPSEEK:
            self._require_secret(s.deepseek_api_key, "DEEPSEEK_API_KEY")
            assert s.deepseek_api_key is not None
            return DeepSeekProvider(c, s.deepseek_api_key)
        if backend is Provider.PERPLEXITY:
            self._require_secret(s.perplexity_api_key, "PERPLEXITY_API_KEY")
            assert s.perplexity_api_key is not None
            return PerplexityProvider(c, s.perplexity_api_key)
        if backend is Provider.XAI:
            self._require_secret(s.xai_api_key, "XAI_API_KEY")
            assert s.xai_api_key is not None
            return XAIProvider(c, s.xai_api_key)
        raise ProviderError(f"Unsupported provider backend: {backend!r}")

    def _backend_for_model(self, model: str) -> Provider:
        kind = Settings.provider_by_model.get(model)
        if kind is None:
            raise UnknownModelError(model)
        return kind

    def get_provider(self, model: str) -> BaseLLMProvider:
        """Resolve the upstream provider instance for a gateway model name."""
        backend = self._backend_for_model(model)
        if backend not in self._by_backend:
            self._by_backend[backend] = self._build(backend)
        return self._by_backend[backend]

    def get_fallback_providers(self, model: str) -> list[BaseLLMProvider]:
        """
        Providers for alternative models, in order, for cross-backend fallback.

        Configure `FALLBACK_MODELS_BY_MODEL` with model ids (values must exist in
        `Settings.provider_by_model`). The primary backend for `model` is skipped so each
        entry refers to a different `BaseLLMProvider` instance where possible.
        """
        primary = self.get_provider(model)
        primary_id = id(primary)
        names = FALLBACK_MODELS_BY_MODEL.get(model, [])
        seen: set[int] = set()
        out: list[BaseLLMProvider] = []
        for name in names:
            if name == model:
                continue
            try:
                p = self.get_provider(name)
            except UnknownModelError:
                continue
            if id(p) == primary_id:
                continue
            pid = id(p)
            if pid in seen:
                continue
            seen.add(pid)
            out.append(p)
        return out


_registry: ProviderRegistry | None = None


def configure_registry(
    client: httpx.AsyncClient,
    settings: Settings | None = None,
) -> ProviderRegistry:
    """Create or replace the process-global registry (call once from app lifespan)."""
    global _registry
    _registry = ProviderRegistry(client, settings)
    return _registry


def get_registry() -> ProviderRegistry:
    if _registry is None:
        msg = (
            "Provider registry is not configured. "
            "Call configure_registry(httpx_client) during application startup."
        )
        raise RuntimeError(msg)
    return _registry


def get_provider(model: str) -> BaseLLMProvider:
    """Return the shared upstream provider for a model name."""
    return get_registry().get_provider(model)


def get_fallback_providers(model: str) -> list[BaseLLMProvider]:
    """Ordered distinct fallback providers (other backends) for `FALLBACK_MODELS_BY_MODEL`."""
    return get_registry().get_fallback_providers(model)


def reset_registry() -> None:
    """Clear the global registry (for tests)."""
    global _registry
    _registry = None
