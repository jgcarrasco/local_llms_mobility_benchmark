"""Chat-only benchmark mode."""

from __future__ import annotations

import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from mobility_bench.models.base import ModelClient, generation_kwargs
from mobility_bench.runtime.extraction import extract_solution_files
from mobility_bench.runtime.inputs import restore_seeded_inputs
from mobility_bench.runtime.logger import EventLogger
from mobility_bench.runtime.runner import AttemptSummary
from mobility_bench.runtime.workspace import Workspace
from mobility_bench.tasks.base import TaskSpec
from mobility_bench.tools.bash import run_bash
from mobility_bench.tools.write import write_file


CHAT_ONLY_SYSTEM_PROMPT = """You are solving a controlled benchmark task.
You cannot inspect files interactively.
You cannot run code.
Return the solution as one or more <solution_file path="relative/path"> blocks.
Do not invent unavailable data.
The harness will write your files, run your solution once, and evaluate the outputs."""


def entry_scripts(task: TaskSpec) -> list[str]:
    """Return the script paths the harness executes for a chat-style attempt.

    A task that declares a script in its expected outputs is executed at exactly
    that path. Every other task is executed at ``src/solution.py``. Helper
    modules written alongside the entry point are imported by it, not run
    separately.
    """
    declared = [path for path in task.expected_outputs if path.endswith(".py")]
    return declared or ["src/solution.py"]


def _chat_only_user_prompt(task: TaskSpec, input_schema: str | None = None) -> str:
    outputs = "\n".join(f"- {path}" for path in task.expected_outputs)
    schema_section = ""
    if input_schema:
        schema_section = (
            "\nInput data structure, generated from inputs/ (column names, types, sizes, and a few "
            "sample rows; the files themselves are not shown):\n"
            + input_schema
            + "\n"
        )
    return (
        task.render_prompt()
        + schema_section
        + "\nOutput format instructions:\n"
        + "- Every solution block must include a workspace-relative path attribute.\n"
        + "- Prefer a single executable Python script that writes all required outputs.\n"
        + "- When the required outputs list a script, write it at exactly that path; otherwise use `src/solution.py`.\n"
        + "- Do not put shell commands or prose inside a solution block.\n"
        + "- Example:\n"
        + '<solution_file path="src/solution.py">\n'
        + "# Python code that reads inputs/ and writes the required outputs.\n"
        + "</solution_file>\n\n"
        + "Required outputs to create:\n"
        + outputs
        + "\n"
    )


def run_chat_only_attempt(
    *,
    task: TaskSpec,
    model: ModelClient,
    workspace: Workspace,
    run_id: str,
    model_id: str,
    repetition: int,
    logger: EventLogger | None = None,
    temperature: float | None = None,
    timeout_seconds: int = 5400,
    bash_timeout_seconds: int = 60,
    execution_backend: str = "local",
    docker_image: str = "mobility-bench-runner:latest",
    docker_network: str = "none",
    data_dir: Path | None = None,
    mode: str = "chat_only",
    input_schema: str | None = None,
) -> AttemptSummary:
    start = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    entry_missing = 0
    messages = [
        {"role": "system", "content": CHAT_ONLY_SYSTEM_PROMPT},
        {"role": "user", "content": _chat_only_user_prompt(task, input_schema)},
    ]
    if logger:
        logger.log("attempt_started", run_id=run_id, task_id=task.task_id, mode=mode)
        logger.log("model_request", messages=messages)
    call_started = time.monotonic()
    response = model.generate(messages, **generation_kwargs(temperature))
    call_elapsed = time.monotonic() - call_started
    timings = response.timings or {}
    cached_tokens = int(((response.usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    if logger:
        logger.log(
            "model_response",
            text=response.text,
            raw_response=response.raw_response,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            total_tokens=response.total_tokens,
            token_counts_estimated=response.token_counts_estimated,
        )

    timing_fields = {
        "started_at": started_at,
        "model_seconds": call_elapsed,
        "max_call_seconds": call_elapsed,
        "prefill_ms_total": float(timings.get("prompt_ms") or 0.0),
        "decode_ms_total": float(timings.get("predicted_ms") or 0.0),
        "decode_tokens_per_second": (
            float(timings["predicted_per_second"]) if timings.get("predicted_per_second") else None
        ),
        "prompt_tokens_cached_total": cached_tokens,
        "reasoning_chars_total": len(response.reasoning_text or ""),
        "n_responses_with_reasoning": 1 if response.reasoning_text else 0,
    }
    finish_reason = response.finish_reason or "unknown"
    response_tokens = {
        "input_tokens_total": response.input_tokens or 0,
        "output_tokens_total": response.output_tokens or 0,
        "total_tokens": response.total_tokens or 0,
        "token_counts_estimated": response.token_counts_estimated,
        "n_truncated_responses": 1 if response.finish_reason == "length" else 0,
        "finish_reasons": {finish_reason: 1},
        "max_prompt_tokens": response.input_tokens or 0,
        "prompt_chars": sum(len(message["content"]) for message in messages),
        **timing_fields,
    }

    if time.monotonic() - start > timeout_seconds:
        return AttemptSummary(
            run_id=run_id,
            task_id=task.task_id,
            model_id=model_id,
            mode=mode,
            repetition=repetition,
            success=False,
            partial_score=0.0,
            failure_category="timeout",
            checker_errors=["attempt timeout reached"],
            time_to_submission_seconds=time.monotonic() - start,
            timeout_reached=True,
            n_model_calls=1,
            **response_tokens,
        )

    solution_files = extract_solution_files(response.text)
    if not solution_files:
        truncated = response.finish_reason == "length"
        return AttemptSummary(
            run_id=run_id,
            task_id=task.task_id,
            model_id=model_id,
            mode=mode,
            repetition=repetition,
            success=False,
            partial_score=0.0,
            failure_category="truncated" if truncated else "format_error",
            checker_errors=[
                "response reached the output token limit without a complete <solution_file> block"
                if truncated
                else "no valid <solution_file> blocks found"
            ],
            time_to_submission_seconds=time.monotonic() - start,
            n_model_calls=1,
            **response_tokens,
        )

    for solution_file in solution_files:
        result = write_file(workspace, solution_file.path, solution_file.content)
        if not result.ok:
            return AttemptSummary(
                run_id=run_id,
                task_id=task.task_id,
                model_id=model_id,
                mode=mode,
                repetition=repetition,
                success=False,
                partial_score=0.0,
                failure_category="schema_error",
                checker_errors=[result.stderr],
                time_to_submission_seconds=time.monotonic() - start,
                n_model_calls=1,
                **response_tokens,
            )

    code_executions = 0
    tool_elapsed = 0.0
    for entry_path in entry_scripts(task):
        script = Path(workspace.root) / entry_path
        if not script.is_file():
            if logger:
                logger.log("entry_point_missing", path=entry_path)
            entry_missing += 1
            continue
        if time.monotonic() - start > timeout_seconds:
            return AttemptSummary(
                run_id=run_id,
                task_id=task.task_id,
                model_id=model_id,
                mode=mode,
                repetition=repetition,
                success=False,
                partial_score=0.0,
                failure_category="timeout",
                checker_errors=["attempt timeout reached"],
                time_to_submission_seconds=time.monotonic() - start,
                timeout_reached=True,
                n_model_calls=1,
                n_bash_calls=code_executions,
                n_code_executions=code_executions,
                tool_seconds=tool_elapsed,
                **response_tokens,
            )
        relative = script.relative_to(workspace.root)
        command = f"{sys.executable} {relative.as_posix()}"
        tool_started = time.monotonic()
        tool_result = run_bash(
            workspace,
            command,
            timeout_seconds=bash_timeout_seconds,
            backend=execution_backend,
            docker_image=docker_image,
            docker_network=docker_network,
        )
        tool_elapsed += time.monotonic() - tool_started
        code_executions += 1
        if logger:
            logger.log("tool_result", tool="bash", result=tool_result.to_dict())
        if not tool_result.ok:
            return AttemptSummary(
                run_id=run_id,
                task_id=task.task_id,
                model_id=model_id,
                mode=mode,
                repetition=repetition,
                success=False,
                partial_score=0.0,
                failure_category="timeout" if tool_result.metadata.get("timeout_reached") else "runtime_error",
                checker_errors=[tool_result.stderr or tool_result.stdout],
                time_to_submission_seconds=time.monotonic() - start,
                timeout_reached=bool(tool_result.metadata.get("timeout_reached")),
                n_model_calls=1,
                n_bash_calls=code_executions,
                n_code_executions=code_executions,
                tool_seconds=tool_elapsed,
                **response_tokens,
            )

    if time.monotonic() - start > timeout_seconds:
        return AttemptSummary(
            run_id=run_id,
            task_id=task.task_id,
            model_id=model_id,
            mode=mode,
            repetition=repetition,
            success=False,
            partial_score=0.0,
            failure_category="timeout",
            checker_errors=["attempt timeout reached"],
            time_to_submission_seconds=time.monotonic() - start,
            timeout_reached=True,
            n_model_calls=1,
            n_bash_calls=code_executions,
            n_code_executions=code_executions,
            tool_seconds=tool_elapsed,
            **response_tokens,
        )

    if data_dir is not None:
        restored = restore_seeded_inputs(task, data_dir, Path(workspace.root))
        if logger and restored:
            logger.log("inputs_restored", paths=restored)
    checker_started = time.monotonic()
    checker_result = task.checker(Path(workspace.root))
    checker_seconds = time.monotonic() - checker_started
    if logger:
        logger.log("final_checker_result", result=checker_result.__dict__)
        logger.log("attempt_finished", summary=json.dumps(checker_result.__dict__, sort_keys=True))

    return AttemptSummary(
        run_id=run_id,
        task_id=task.task_id,
        model_id=model_id,
        mode=mode,
        repetition=repetition,
        success=checker_result.success,
        partial_score=checker_result.partial_score,
        failure_category=checker_result.failure_category,
        checker_errors=checker_result.checker_errors,
        time_to_submission_seconds=time.monotonic() - start,
        n_model_calls=1,
        n_bash_calls=code_executions,
        n_code_executions=code_executions,
        **response_tokens,
        finished_at=datetime.now(timezone.utc).isoformat(),
        tool_seconds=tool_elapsed,
        checker_seconds=checker_seconds,
        cache_hit_rate=(
            cached_tokens / (response.input_tokens or 1) if response.input_tokens else None
        ),
        n_entry_point_missing=entry_missing,
        final_checker_success=checker_result.success,
    )
