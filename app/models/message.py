from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

ChatRole = Literal["system", "user", "assistant", "tool", "developer"]


class Message(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: ChatRole | str = Field(
        ...,
        description="Role of the message author.",
    )
    content: str | list[dict[str, Any]] | None = Field(
        default=None,
        description="Text or structured content (multimodal parts).",
    )
    name: str | None = Field(
        default=None,
        description="Optional name for the participant.",
    )
    tool_calls: list[dict[str, Any]] | None = Field(
        default=None,
        description="Assistant tool calls, when applicable.",
    )
    tool_call_id: str | None = Field(
        default=None,
        description="For role=tool, id of the call this message answers.",
    )
