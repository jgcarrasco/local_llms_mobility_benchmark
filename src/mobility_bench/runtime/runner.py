"""Common attempt result structures."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mobility_bench.config import BenchmarkConfig, ModelConfig
from mobility_bench.models.base import ModelClient
from mobility_bench.runtime.logger import EventLogger
from mobility_bench.runtime.workspace import create_attempt_workspace
from mobility_bench.tasks.base import TaskSpec


@dataclass(frozen=True)
class AttemptSummary:
    run_id: str
    task_id: str
    model_id: str
    mode: str
    repetition: int
    success: bool
    partial_score: float
    failure_category: str | None
    checker_errors: list[str] = field(default_factory=list)
    time_to_submission_seconds: float | None = None
    timeout_reached: bool = False
    stop_reason: str | None = None
    cap_exhausted: bool = False
    n_model_calls: int = 0
    n_tool_calls_total: int = 0
    n_read_calls: int = 0
    n_write_calls: int = 0
    n_edit_calls: int = 0
    n_bash_calls: int = 0
    n_submit_calls: int = 0
    n_failed_tool_calls: int = 0
    n_code_executions: int = 0
    n_error_messages_observed: int = 0
    input_tokens_total: int = 0
    output_tokens_total: int = 0
    total_tokens: int = 0
    token_counts_estimated: bool = False
    first_silent_checker_success: bool = False
    final_checker_success: bool = False
    infrastructure_failure: bool = False
    notes: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "model_id": self.model_id,
            "mode": self.mode,
            "repetition": self.repetition,
            "success": self.success,
            "partial_score": self.partial_score,
            "failure_category": self.failure_category,
            "checker_errors": self.checker_errors,
            "time_to_submission_seconds": self.time_to_submission_seconds,
            "timeout_reached": self.timeout_reached,
            "stop_reason": self.stop_reason,
            "cap_exhausted": self.cap_exhausted,
            "n_model_calls": self.n_model_calls,
            "n_tool_calls_total": self.n_tool_calls_total,
            "n_read_calls": self.n_read_calls,
            "n_write_calls": self.n_write_calls,
            "n_edit_calls": self.n_edit_calls,
            "n_bash_calls": self.n_bash_calls,
            "n_submit_calls": self.n_submit_calls,
            "n_failed_tool_calls": self.n_failed_tool_calls,
            "n_code_executions": self.n_code_executions,
            "n_error_messages_observed": self.n_error_messages_observed,
            "input_tokens_total": self.input_tokens_total,
            "output_tokens_total": self.output_tokens_total,
            "total_tokens": self.total_tokens,
            "token_counts_estimated": self.token_counts_estimated,
            "first_silent_checker_success": self.first_silent_checker_success,
            "final_checker_success": self.final_checker_success,
            "infrastructure_failure": self.infrastructure_failure,
            "notes": self.notes,
        }


def _copy_inputs(data_dir: Path, workspace_root: Path) -> None:
    source = data_dir / "inputs"
    target = workspace_root / "inputs"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(source, target)


def _copy_reference_file(data_dir: Path, workspace_root: Path, relative_path: str) -> None:
    source = data_dir / "reference_outputs" / relative_path
    if not source.exists():
        return
    target = workspace_root / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)


def _remove_expected_outputs(workspace_root: Path, expected_outputs: list[str]) -> None:
    for relative_path in expected_outputs:
        target = workspace_root / relative_path
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()


def _copy_reference_inputs(task: TaskSpec, data_dir: Path, workspace_root: Path) -> None:
    reference_root = data_dir / "reference_outputs"
    if not reference_root.exists():
        return
    if "all previous outputs" in task.input_files:
        for child in ("outputs", "src"):
            source = reference_root / child
            target = workspace_root / child
            if source.exists():
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(source, target)
        _remove_expected_outputs(workspace_root, task.expected_outputs)
        return
    for relative_path in task.input_files:
        if relative_path.startswith(("outputs/", "src/", "reports/", "figures/")):
            _copy_reference_file(data_dir, workspace_root, relative_path)


def effective_timeout_seconds(task: TaskSpec, benchmark: BenchmarkConfig) -> int:
    """Return the attempt wall-clock cap used by the benchmark run."""
    if benchmark.runtime.timeout_seconds_per_task > 0:
        return benchmark.runtime.timeout_seconds_per_task
    return task.timeout_seconds


def run_task_attempt(
    *,
    task: TaskSpec,
    model_client: ModelClient,
    model_config: ModelConfig,
    model_id: str,
    mode: str,
    repetition: int,
    benchmark: BenchmarkConfig,
) -> AttemptSummary:
    run_id = f"{model_id}__{mode}__{task.task_id}__rep{repetition:02d}"
    workspace = create_attempt_workspace(
        run_id,
        benchmark.paths.runs_dir,
        clean=benchmark.runtime.clean_workspace_per_attempt,
    )
    _copy_inputs(benchmark.paths.data_dir, workspace.root)
    _copy_reference_inputs(task, benchmark.paths.data_dir, workspace.root)
    logger = EventLogger(benchmark.paths.runs_dir / "raw_logs" / f"{run_id}.jsonl")
    timeout_seconds = effective_timeout_seconds(task, benchmark)

    if mode == "chat_only":
        from mobility_bench.modes.chat_only import run_chat_only_attempt

        summary = run_chat_only_attempt(
            task=task,
            model=model_client,
            workspace=workspace,
            run_id=run_id,
            model_id=model_id,
            repetition=repetition,
            logger=logger,
            temperature=benchmark.runtime.temperature,
            timeout_seconds=timeout_seconds,
            bash_timeout_seconds=benchmark.runtime.bash_timeout_seconds,
            execution_backend=benchmark.execution.backend,
            docker_image=benchmark.execution.docker_image,
            docker_network=benchmark.execution.docker_network,
        )
    elif mode == "agent_tools":
        from mobility_bench.modes.agent_tools import run_agent_tools_attempt

        summary = run_agent_tools_attempt(
            task=task,
            model=model_client,
            workspace=workspace,
            run_id=run_id,
            model_id=model_id,
            repetition=repetition,
            logger=logger,
            max_iterations=benchmark.runtime.max_iterations,
            max_tool_calls=benchmark.runtime.max_tool_calls,
            max_consecutive_protocol_errors=benchmark.runtime.max_consecutive_protocol_errors,
            max_protocol_response_tokens=benchmark.runtime.max_protocol_response_tokens,
            max_repeated_tool_calls=benchmark.runtime.max_repeated_tool_calls,
            max_repeated_failed_tool_calls=benchmark.runtime.max_repeated_failed_tool_calls,
            temperature=benchmark.runtime.temperature,
            timeout_seconds=timeout_seconds,
            bash_timeout_seconds=benchmark.runtime.bash_timeout_seconds,
            execution_backend=benchmark.execution.backend,
            docker_image=benchmark.execution.docker_image,
            docker_network=benchmark.execution.docker_network,
        )
    else:
        raise ValueError(f"unsupported mode: {mode}")

    enriched = {
        **summary.to_dict(),
        "task_family": task.family,
        "model_family": model_config.family,
        "model_size_b": model_config.size_b,
        "architecture": model_config.architecture,
        "quantization": model_config.quantization,
        "context_length": model_config.context_length,
        "source_repo": model_config.source_repo,
        "artifact_repo": model_config.artifact_repo,
    }
    results_dir = benchmark.paths.runs_dir / "results_jsonl"
    results_dir.mkdir(parents=True, exist_ok=True)
    with (results_dir / f"{run_id}.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(enriched, ensure_ascii=True, sort_keys=True) + "\n")
    return summary
