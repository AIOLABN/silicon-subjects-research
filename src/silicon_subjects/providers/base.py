from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, Protocol


@dataclass(frozen=True)
class ChatMessage:
    role: Literal["user", "assistant"]
    content: str

    raw: list[dict[str, Any]] | None = None


@dataclass(frozen=True)
class Completion:
    text: str
    stop_reason: str | None

    served_model: str | None

    platform_refusal: bool = False
    input_tokens: int | None = None
    output_tokens: int | None = None
    request_id: str | None = None
    latency_s: float | None = None
    extra: dict[str, object] = field(default_factory=dict)
    raw_content: list[dict[str, Any]] | None = None


class TransientProviderError(Exception):
    def __init__(self, message: str, *, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


def parse_retry_after(value: str | None) -> float | None:
    try:
        seconds = float(value) if value is not None else None
    except ValueError:
        return None
    return seconds if seconds is not None and seconds >= 0 else None


class Provider(Protocol):
    name: str
    # Whether `seed` reaches the model. When it does not, a trial's seed only identifies it, and
    # the stored transcript is the record of what the model said.
    sends_seed: bool

    async def complete(
        self,
        system: str,
        messages: list[ChatMessage],
        *,
        seed: int,
        response_schema: dict[str, Any] | None = None,
    ) -> Completion: ...

    async def aclose(self) -> None: ...
