"""Configuration loading for benchmark, model, and task settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class RuntimeConfig(BaseModel):
    # None means each model uses its own generation.temperature.
    temperature: float | None = None
    max_iterations: int = 150
    max_tool_calls: int = 150
    max_consecutive_protocol_errors: int = 8
    max_repeated_tool_calls: int = 20
    max_repeated_failed_tool_calls: int = 8
    timeout_seconds_per_task: int = 5400
    bash_timeout_seconds: int = 60
    repetitions: int = 3
    network_enabled: bool = False
    clean_workspace_per_attempt: bool = True


class PathsConfig(BaseModel):
    data_dir: Path = Path("data")
    runs_dir: Path = Path("runs")
    reports_dir: Path = Path("reports")


class ExecutionConfig(BaseModel):
    backend: str = "local"
    docker_image: str = "mobility-bench-runner:latest"
    docker_network: str = "none"
    expose_sensitive_env: bool = False


class BenchmarkConfig(BaseModel):
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)


class ModelConfig(BaseModel):
    family: str
    size_b: float
    architecture: str
    local: bool
    backend: str
    base_url: str | None = None
    model_name: str
    quantization: str
    context_length: int
    # How the agent interface is spoken: the model's native function-calling
    # channel, or the benchmark's text protocol for templates that have none.
    tool_protocol: str = "text"
    source_repo: str | None = None
    artifact_repo: str | None = None
    # Hosted endpoints need a bearer token. The key itself never goes in a
    # config file: name the environment variable that holds it.
    api_key_env: str | None = None
    # Extra HTTP headers for gateways with non-standard requirements.
    extra_headers: dict[str, str] = Field(default_factory=dict)
    generation: dict[str, Any] = Field(default_factory=dict)


class ModelsConfig(BaseModel):
    models: dict[str, ModelConfig]


class TasksConfig(BaseModel):
    tasks: dict[str, list[str]]


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    return data or {}


def load_benchmark_config(path: str | Path) -> BenchmarkConfig:
    return BenchmarkConfig.model_validate(load_yaml(path))


def load_models_config(path: str | Path) -> ModelsConfig:
    return ModelsConfig.model_validate(load_yaml(path))


def load_tasks_config(path: str | Path) -> TasksConfig:
    return TasksConfig.model_validate(load_yaml(path))
