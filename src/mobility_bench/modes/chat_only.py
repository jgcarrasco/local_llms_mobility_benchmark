"""Chat-only benchmark mode."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from mobility_bench.models.base import ModelClient
from mobility_bench.runtime.extraction import extract_solution_files
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


def _chat_only_user_prompt(task: TaskSpec) -> str:
    outputs = "\n".join(f"- {path}" for path in task.expected_outputs)
    return (
        task.render_prompt()
        + "\nOutput format instructions:\n"
        + "- Every solution block must include a workspace-relative path attribute.\n"
        + "- Prefer a single executable Python script at `src/solution.py` that writes all required outputs.\n"
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
    temperature: float = 0.0,
    timeout_seconds: int = 600,
    bash_timeout_seconds: int = 60,
    execution_backend: str = "local",
    docker_image: str = "mobility-bench-runner:latest",
    docker_network: str = "none",
) -> AttemptSummary:
    start = time.monotonic()
    messages = [
        {"role": "system", "content": CHAT_ONLY_SYSTEM_PROMPT},
        {"role": "user", "content": _chat_only_user_prompt(task)},
    ]
    if logger:
        logger.log("attempt_started", run_id=run_id, task_id=task.task_id, mode="chat_only")
        logger.log("model_request", messages=messages)
    response = model.generate(messages, temperature=temperature)
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

    response_tokens = {
        "input_tokens_total": response.input_tokens or 0,
        "output_tokens_total": response.output_tokens or 0,
        "total_tokens": response.total_tokens or 0,
        "token_counts_estimated": response.token_counts_estimated,
    }

    if time.monotonic() - start > timeout_seconds:
        return AttemptSummary(
            run_id=run_id,
            task_id=task.task_id,
            model_id=model_id,
            mode="chat_only",
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
        return AttemptSummary(
            run_id=run_id,
            task_id=task.task_id,
            model_id=model_id,
            mode="chat_only",
            repetition=repetition,
            success=False,
            partial_score=0.0,
            failure_category="format_error",
            checker_errors=["no valid <solution_file> blocks found"],
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
                mode="chat_only",
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
    for script in sorted(workspace.root.glob("src/**/*.py")):
        if time.monotonic() - start > timeout_seconds:
            return AttemptSummary(
                run_id=run_id,
                task_id=task.task_id,
                model_id=model_id,
                mode="chat_only",
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
                **response_tokens,
            )
        relative = script.relative_to(workspace.root)
        command = f"{sys.executable} {relative.as_posix()}"
        tool_result = run_bash(
            workspace,
            command,
            timeout_seconds=bash_timeout_seconds,
            backend=execution_backend,
            docker_image=docker_image,
            docker_network=docker_network,
        )
        code_executions += 1
        if logger:
            logger.log("tool_result", tool="bash", result=tool_result.to_dict())
        if not tool_result.ok:
            return AttemptSummary(
                run_id=run_id,
                task_id=task.task_id,
                model_id=model_id,
                mode="chat_only",
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
                **response_tokens,
            )

    if time.monotonic() - start > timeout_seconds:
        return AttemptSummary(
            run_id=run_id,
            task_id=task.task_id,
            model_id=model_id,
            mode="chat_only",
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
            **response_tokens,
        )

    checker_result = task.checker(Path(workspace.root))
    if logger:
        logger.log("final_checker_result", result=checker_result.__dict__)
        logger.log("attempt_finished", summary=json.dumps(checker_result.__dict__, sort_keys=True))

    return AttemptSummary(
        run_id=run_id,
        task_id=task.task_id,
        model_id=model_id,
        mode="chat_only",
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
        final_checker_success=checker_result.success,
    )
