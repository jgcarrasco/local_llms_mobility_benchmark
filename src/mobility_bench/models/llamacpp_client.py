"""Client for llama.cpp OpenAI-compatible chat endpoints."""

from __future__ import annotations

from typing import Any

import requests

from mobility_bench.models.base import ModelResponse


class LlamaCppClient:
    def __init__(
        self,
        base_url: str,
        model_name: str,
        timeout_seconds: int = 3600,
        generation_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.generation_kwargs = generation_kwargs or {}

    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> ModelResponse:
        payload = {
            "model": self.model_name,
            "messages": messages,
            **self.generation_kwargs,
            **kwargs,
        }
        response = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            timeout=self.timeout_seconds,
        )
        response.raise_for_status()
        data = response.json()
        text = data["choices"][0]["message"]["content"]
        usage = data.get("usage") or {}
        return ModelResponse(
            text=text,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
            raw_response=data,
            token_counts_estimated=not bool(usage),
        )
