from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from app.models.message import Message
from app.models.usage import Usage

FinishReason = Literal["stop", "length", "tool_calls", "content_filter", "function_call"]


class ChatCompletionChoice(BaseModel):
    model_config = ConfigDict(extra="allow")

    index: int = Field(..., ge=0)
    message: Message
    finish_reason: FinishReason | str | None = None
    logprobs: dict[str, Any] | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")

    model: str
    messages: list[Message]

    temperature: float | None = Field(
        default=None,
        ge=0,
        le=2,
        description="Sampling temperature.",
    )
    max_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Legacy max tokens to generate.",
    )
    max_completion_tokens: int | None = Field(
        default=None,
        ge=1,
        description="Max tokens in the completion (newer APIs).",
    )
    top_p: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Nucleus sampling mass.",
    )
    n: int | None = Field(default=1, ge=1, description="Number of completions to sample.")
    stream: bool = False
    stop: str | list[str] | None = None
    presence_penalty: float | None = Field(default=None, ge=-2, le=2)
    frequency_penalty: float | None = Field(default=None, ge=-2, le=2)
    logit_bias: dict[str, int] | None = None
    user: str | None = None
    seed: int | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    response_format: dict[str, Any] | None = None


class ChatCompletionResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(..., description="Unix timestamp (seconds).")
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage | None = None
    system_fingerprint: str | None = None
