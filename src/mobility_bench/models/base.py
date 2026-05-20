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


class ModelClient(Protocol):
    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> ModelResponse:
        """Generate a response for chat messages."""
