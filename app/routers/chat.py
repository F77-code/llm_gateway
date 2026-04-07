from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from app.dependencies import get_api_key
from app.models.chat_completion import ChatCompletionRequest, ChatCompletionResponse
from app.providers.base import (
    ProviderError,
    ProviderHTTPError,
    ProviderRateLimitError,
    ProviderRequestError,
    ProviderServerError,
    ProviderTimeoutError,
    ProviderUnauthorizedError,
)
from app.providers.registry import UnknownModelError, get_provider

router = APIRouter(prefix="/v1", tags=["chat"])


@router.post(
    "/chat/completions",
    response_model=ChatCompletionResponse,
)
async def chat_completions(
    request: ChatCompletionRequest,
    _: str = Depends(get_api_key),
) -> ChatCompletionResponse:
    try:
        provider = get_provider(request.model)
    except UnknownModelError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unknown model: {exc.model}",
        ) from exc
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Provider registry is not initialized",
        ) from exc

    try:
        return await provider.chat_completion(request)
    except ProviderUnauthorizedError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
        ) from exc
    except ProviderRateLimitError as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(exc),
        ) from exc
    except ProviderServerError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
    except ProviderTimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail=str(exc),
        ) from exc
    except ProviderHTTPError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=str(exc),
        ) from exc
    except ProviderRequestError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except ProviderError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc),
        ) from exc
