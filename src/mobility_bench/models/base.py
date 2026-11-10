"""Common model client interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True)
class ModelResponse:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    raw_response: dict[str, Any] | None = None
    token_counts_estimated: bool = False
    reasoning_text: str | None = None
    finish_reason: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    usage: dict[str, Any] | None = None
    timings: dict[str, Any] | None = None


class ModelClient(Protocol):
    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> ModelResponse:
        """Generate a response for chat messages."""


def generation_kwargs(temperature: float | None) -> dict[str, Any]:
    """Return per-call generation overrides.

    When the benchmark does not fix a temperature, each model keeps the value
    from its own ``generation`` block.
    """
    return {} if temperature is None else {"temperature": temperature}
