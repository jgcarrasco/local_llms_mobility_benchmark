"""Deterministic model client for tests and local development."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from mobility_bench.models.base import ModelResponse


class MockClient:
    def __init__(self, responses: Iterable[str] | None = None) -> None:
        self._responses = list(responses or [""])
        self.calls: list[list[dict[str, str]]] = []

    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> ModelResponse:
        del kwargs
        self.calls.append(messages)
        if self._responses:
            text = self._responses.pop(0)
        else:
            text = ""
        return ModelResponse(text=text, token_counts_estimated=True)
