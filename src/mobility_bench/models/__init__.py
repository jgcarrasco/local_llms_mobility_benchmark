"""Model client implementations."""

from __future__ import annotations

from mobility_bench.config import ModelConfig
from mobility_bench.models.base import ModelClient
from mobility_bench.models.llamacpp_client import LlamaCppClient
from mobility_bench.models.mock_client import MockClient


def create_model_client(config: ModelConfig) -> ModelClient:
    if config.backend == "mock":
        return MockClient()
    if config.backend == "llamacpp":
        if not config.base_url:
            raise ValueError("llama.cpp model config requires base_url")
        return LlamaCppClient(
            base_url=config.base_url,
            model_name=config.model_name,
            generation_kwargs=config.generation,
        )
    raise ValueError(f"unsupported model backend: {config.backend}")
