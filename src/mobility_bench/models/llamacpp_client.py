"""Client for llama.cpp OpenAI-compatible chat endpoints."""

from __future__ import annotations

import json
from typing import Any

import requests

from mobility_bench.models.base import ModelResponse


def _normalise_tool_calls(raw: object) -> list[dict[str, Any]] | None:
    """Convert the server's tool_calls into {id, name, arguments} dicts."""
    if not isinstance(raw, list) or not raw:
        return None
    calls: list[dict[str, Any]] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        function = entry.get("function") or {}
        arguments = function.get("arguments")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        if not isinstance(arguments, dict):
            arguments = {}
        calls.append(
            {
                "id": entry.get("id"),
                "name": function.get("name") or "__invalid__",
                "arguments": arguments,
            }
        )
    return calls or None


class LlamaCppClient:
    """OpenAI-compatible chat client.

    Serves both a local llama.cpp server (no auth) and hosted gateways that
    require a bearer token and extra headers.
    """

    def __init__(
        self,
        base_url: str,
        model_name: str,
        timeout_seconds: int = 3600,
        generation_kwargs: dict[str, Any] | None = None,
        api_key: str | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.generation_kwargs = generation_kwargs or {}
        self.headers = {"User-Agent": "mobility-bench/1.0"}
        if extra_headers:
            self.headers.update(extra_headers)
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"

    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> ModelResponse:
        payload = {
            "model": self.model_name,
            "messages": messages,
            **self.generation_kwargs,
            **kwargs,
        }
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers=self.headers,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as exc:
            # A single server-side failure must not abort a long sweep. The
            # attempt loop treats this as a recoverable protocol error.
            body = getattr(getattr(exc, "response", None), "text", "") or ""
            return ModelResponse(
                text="",
                raw_response={"error": f"{type(exc).__name__}: {exc}", "body": body[:2000]},
                token_counts_estimated=True,
                finish_reason="transport_error",
            )
        choice = data["choices"][0]
        message = choice.get("message") or {}
        # Reasoning models return an empty content field when the response is
        # truncated mid-thought, so coerce to a string before the parsers see it.
        text = message.get("content") or ""
        reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
        usage = data.get("usage") or {}
        return ModelResponse(
            text=text,
            input_tokens=usage.get("prompt_tokens"),
            output_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
            raw_response=data,
            token_counts_estimated=not bool(usage),
            reasoning_text=reasoning or None,
            finish_reason=choice.get("finish_reason"),
            tool_calls=_normalise_tool_calls(message.get("tool_calls")),
            usage=usage or None,
            timings=data.get("timings") or None,
        )
