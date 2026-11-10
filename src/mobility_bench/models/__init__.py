"""Model client implementations."""

from __future__ import annotations

import os
import uuid

from mobility_bench.config import ModelConfig
from mobility_bench.models.base import ModelClient
from mobility_bench.models.llamacpp_client import LlamaCppClient
from mobility_bench.models.mock_client import MockClient

# Backends served through an OpenAI-compatible chat-completions endpoint.
_OPENAI_COMPATIBLE_BACKENDS = {"llamacpp", "openai_compatible"}


def default_session_id(model_name: str) -> str:
    """A stable identifier for gateways that require a session header."""
    return f"sophia-{model_name}-{uuid.uuid4().hex[:12]}"


def create_model_client(config: ModelConfig) -> ModelClient:
    if config.backend == "mock":
        return MockClient()
    if config.backend in _OPENAI_COMPATIBLE_BACKENDS:
        if not config.base_url:
            raise ValueError(f"{config.backend} model config requires base_url")
        api_key = None
        if config.api_key_env:
            api_key = os.environ.get(config.api_key_env)
            if not api_key:
                raise ValueError(
                    f"model {config.model_name!r} needs an API key in "
                    f"${config.api_key_env}, which is unset or empty"
                )
        return LlamaCppClient(
            base_url=config.base_url,
            model_name=config.model_name,
            generation_kwargs=config.generation,
            api_key=api_key,
            extra_headers=dict(config.extra_headers),
        )
    raise ValueError(f"unsupported model backend: {config.backend}")
