"""One-shot benchmark mode that receives a structural summary of the inputs."""

from __future__ import annotations

from pathlib import Path

from mobility_bench.models.base import ModelClient
from mobility_bench.modes.chat_only import run_chat_only_attempt
from mobility_bench.runtime.input_schema import build_input_schema
from mobility_bench.runtime.logger import EventLogger
from mobility_bench.runtime.runner import AttemptSummary
from mobility_bench.runtime.workspace import Workspace
from mobility_bench.tasks.base import TaskSpec


def run_chat_schema_attempt(
    *,
    task: TaskSpec,
    model: ModelClient,
    workspace: Workspace,
    run_id: str,
    model_id: str,
    repetition: int,
    logger: EventLogger | None = None,
    temperature: float = 0.0,
    timeout_seconds: int = 5400,
    bash_timeout_seconds: int = 60,
    execution_backend: str = "local",
    docker_image: str = "mobility-bench-runner:latest",
    docker_network: str = "none",
    data_dir: Path | None = None,
) -> AttemptSummary:
    """Run the one-shot condition with a generated input schema in the prompt.

    The summary is built from the seeded inputs, so every attempt of a task
    receives identical text, and it is logged verbatim with the attempt.
    """
    input_schema = build_input_schema(Path(workspace.root), task)
    if logger:
        logger.log("input_schema", text=input_schema, chars=len(input_schema))
    return run_chat_only_attempt(
        task=task,
        model=model,
        workspace=workspace,
        run_id=run_id,
        model_id=model_id,
        repetition=repetition,
        logger=logger,
        temperature=temperature,
        timeout_seconds=timeout_seconds,
        bash_timeout_seconds=bash_timeout_seconds,
        execution_backend=execution_backend,
        docker_image=docker_image,
        docker_network=docker_network,
        data_dir=data_dir,
        mode="chat_schema",
        input_schema=input_schema,
    )
