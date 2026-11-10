"""Tool-using benchmark mode."""

from __future__ import annotations

import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from mobility_bench.models.base import ModelClient, generation_kwargs
from mobility_bench.runtime.extraction import ParsedToolCall, extract_tool_calls
from mobility_bench.runtime.inputs import restore_seeded_inputs
from mobility_bench.runtime.logger import EventLogger
from mobility_bench.runtime.runner import AttemptSummary
from mobility_bench.runtime.workspace import Workspace
from mobility_bench.tasks.base import TaskSpec
from mobility_bench.tools import ALLOWED_TOOLS, execute_tool, normalize_tool_call
from mobility_bench.tools.schemas import TOOL_SCHEMAS


AGENT_SYSTEM_PROMPT = """You are solving a controlled benchmark task.
You may only use the available tools.
Do not assume internet access.
Do not access files outside the workspace.
Do not invent unavailable data.
Produce the required output files.
When finished, call submit.

Protocol rules:
- Reply with exactly one <tool_call> block per turn.
- Do not write prose, markdown, or code outside the <tool_call> block.
- The JSON inside <tool_call> must be one object with exactly two top-level fields: "tool" and "args".
- The "tool" field must be a string: "read", "write", "edit", "bash", or "submit".
- The "args" field must be an object.
- Tool results can be long. Do not copy tool-result content into your reply; choose the next tool call.
- For multi-line Python, use write to create a script, then bash to run it; avoid giant inline bash commands.

Available tools and argument schemas:
- read: {"path": "relative/path.txt"} or {"path": "relative/path.txt", "max_chars": 12000}
- write: {"path": "relative/path.txt", "content": "complete file content"}
- edit: {"path": "relative/path.txt", "old": "text to replace", "new": "replacement text", "replace_all": true}
- bash: {"command": "python src/solution.py"}
- submit: {"message": "done"}

Valid examples:
<tool_call>
{"tool": "read", "args": {"path": "inputs/example.csv"}}
</tool_call>

<tool_call>
{"tool": "write", "args": {"path": "outputs/example.txt", "content": "complete file content"}}
</tool_call>

Invalid examples:
- {"tool": {"args": {"path": "inputs/example.csv"}}}
- {"read": {"path": "inputs/example.csv"}}
- prose before or after the <tool_call> block
- copying a CSV or script into the response without using the write tool

Use this exact shape for every response:
<tool_call>
{"tool": "...", "args": {...}}
</tool_call>"""


AGENT_NATIVE_SYSTEM_PROMPT = """You are solving a controlled benchmark task.
You may only use the provided tools.
Do not assume internet access.
Do not access files outside the workspace.
Do not invent unavailable data.
Produce the required output files with the tools; do not describe files in prose.
Inspect the inputs before transforming them, and verify the outputs before finishing.
When every required artifact exists, call submit."""


def _tool_result_message(result: dict[str, object]) -> str:
    return "<tool_result>\n" + json.dumps(result, ensure_ascii=True, indent=2) + "\n</tool_result>"


def run_agent_tools_attempt(
    *,
    task: TaskSpec,
    model: ModelClient,
    workspace: Workspace,
    run_id: str,
    model_id: str,
    repetition: int,
    logger: EventLogger | None = None,
    max_iterations: int = 150,
    max_tool_calls: int = 150,
    max_consecutive_protocol_errors: int = 8,
    max_repeated_tool_calls: int = 20,
    max_repeated_failed_tool_calls: int = 8,
    temperature: float | None = None,
    timeout_seconds: int = 5400,
    bash_timeout_seconds: int = 60,
    execution_backend: str = "local",
    docker_image: str = "mobility-bench-runner:latest",
    docker_network: str = "none",
    data_dir: Path | None = None,
    tool_protocol: str = "text",
) -> AttemptSummary:
    native = tool_protocol == "native"
    start = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    model_seconds = 0.0
    tool_seconds = 0.0
    checker_seconds = 0.0
    max_call_seconds = 0.0
    prefill_ms = 0.0
    decode_ms = 0.0
    decode_tps: list[float] = []
    cached_tokens = 0
    reasoning_chars = 0
    reasoning_responses = 0
    parallel_calls = 0
    messages = [
        {"role": "system", "content": AGENT_NATIVE_SYSTEM_PROMPT if native else AGENT_SYSTEM_PROMPT},
        {"role": "user", "content": task.render_prompt()},
    ]
    counters = {
        "read": 0,
        "write": 0,
        "edit": 0,
        "bash": 0,
        "submit": 0,
        "failed": 0,
    }
    model_calls = 0
    tool_calls = 0
    input_tokens = 0
    output_tokens = 0
    total_tokens = 0
    token_counts_estimated = False
    consecutive_protocol_errors = 0
    tool_call_counts: dict[str, int] = {}
    failed_tool_call_counts: dict[str, int] = {}
    finish_reasons: dict[str, int] = {}
    n_truncated_responses = 0
    max_prompt_tokens = 0
    prompt_chars = sum(len(message["content"]) for message in messages)

    if logger:
        logger.log(
            "attempt_started",
            run_id=run_id,
            task_id=task.task_id,
            mode="agent_tools",
            system_prompt=AGENT_SYSTEM_PROMPT,
            task_prompt=task.render_prompt(),
        )

    failure_category: str | None = None
    checker_errors: list[str] = []
    submitted = False
    timeout_reached = False
    stop_reason: str | None = None
    cap_exhausted = False
    silent_checker_result = None

    for _iteration in range(max_iterations):
        if time.monotonic() - start > timeout_seconds:
            timeout_reached = True
            stop_reason = "timeout"
            failure_category = "timeout"
            checker_errors = ["attempt timeout reached"]
            break
        if tool_calls >= max_tool_calls:
            stop_reason = "max_tool_calls"
            cap_exhausted = True
            failure_category = "protocol_error"
            checker_errors = ["maximum tool calls reached"]
            break
        if max_consecutive_protocol_errors > 0 and consecutive_protocol_errors >= max_consecutive_protocol_errors:
            stop_reason = "protocol_stall"
            cap_exhausted = True
            failure_category = "protocol_error"
            checker_errors = [f"maximum consecutive protocol errors reached: {max_consecutive_protocol_errors}"]
            break

        if logger:
            logger.log("model_request", n_messages=len(messages))
        call_started = time.monotonic()
        if native:
            response = model.generate(
                messages, tools=TOOL_SCHEMAS, tool_choice="auto", **generation_kwargs(temperature)
            )
        else:
            response = model.generate(messages, **generation_kwargs(temperature))
        call_elapsed = time.monotonic() - call_started
        model_seconds += call_elapsed
        max_call_seconds = max(max_call_seconds, call_elapsed)
        timings = response.timings or {}
        prefill_ms += timings.get("prompt_ms") or 0.0
        decode_ms += timings.get("predicted_ms") or 0.0
        if timings.get("predicted_per_second"):
            decode_tps.append(float(timings["predicted_per_second"]))
        cached_tokens += int(((response.usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
        if response.reasoning_text:
            reasoning_responses += 1
            reasoning_chars += len(response.reasoning_text)
        if response.tool_calls and len(response.tool_calls) > 1:
            parallel_calls += 1
        model_calls += 1
        input_tokens += response.input_tokens or 0
        output_tokens += response.output_tokens or 0
        total_tokens += response.total_tokens or 0
        token_counts_estimated = token_counts_estimated or response.token_counts_estimated
        max_prompt_tokens = max(max_prompt_tokens, response.input_tokens or 0)
        finish_reason = response.finish_reason or "unknown"
        finish_reasons[finish_reason] = finish_reasons.get(finish_reason, 0) + 1
        # Only the answer is fed back into the conversation; reasoning traces are
        # not replayed, or the context would overflow after a few turns.
        if native and response.tool_calls:
            messages.append(
                {
                    "role": "assistant",
                    "content": response.text,
                    "tool_calls": [
                        {
                            "id": call["id"],
                            "type": "function",
                            "function": {
                                "name": call["name"],
                                "arguments": json.dumps(call["arguments"], ensure_ascii=True),
                            },
                        }
                        for call in response.tool_calls
                    ],
                }
            )
        else:
            messages.append({"role": "assistant", "content": response.text})
        if logger:
            logger.log(
                "model_response",
                text=response.text,
                reasoning_text=response.reasoning_text,
                finish_reason=response.finish_reason,
                raw_response=response.raw_response,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                total_tokens=response.total_tokens,
                token_counts_estimated=response.token_counts_estimated,
            )

        if native:
            calls = [
                ParsedToolCall(tool=call["name"], args=call["arguments"], raw=call)
                for call in (response.tool_calls or [])
            ]
        else:
            try:
                calls = extract_tool_calls(response.text)
            except ValueError as exc:
                calls = []
                if logger:
                    logger.log("protocol_recovery", error=str(exc), text=response.text)
        if not calls and response.finish_reason == "length":
            # The model ran out of output budget, usually inside a reasoning
            # trace. Retrying the same prompt would spend another full call for
            # the same result, so the attempt ends here and the condition is
            # reported separately from model errors.
            n_truncated_responses += 1
            stop_reason = "truncated_response"
            failure_category = "truncated"
            checker_errors = [
                f"response reached the output token limit ({response.output_tokens or 0} tokens) without a tool call"
            ]
            break
        if not calls:
            result = {
                "ok": False,
                "stdout": "",
                "stderr": (
                    "response did not contain a tool call; use one of the provided tools"
                    if native
                    else "response did not contain a valid <tool_call>; continue with one allowed tool call"
                ),
                "metadata": {"recoverable_protocol_error": True},
            }
            counters["failed"] += 1
            consecutive_protocol_errors += 1
            if logger:
                logger.log("tool_result", tool="__invalid__", result=result)
            messages.append({"role": "user", "content": _tool_result_message(result)})
            continue

        saw_valid_tool_call = False
        for call in calls:
            if tool_calls >= max_tool_calls:
                break
            tool_calls += 1
            tool, args = normalize_tool_call(call.tool, call.args)
            if tool not in ALLOWED_TOOLS:
                result = {"ok": False, "stdout": "", "stderr": f"unknown tool: {call.tool}", "metadata": call.args}
            else:
                saw_valid_tool_call = True
                tool_started = time.monotonic()
                result = execute_tool(
                    workspace,
                    tool,
                    args,
                    bash_timeout_seconds=bash_timeout_seconds,
                    execution_backend=execution_backend,
                    docker_image=docker_image,
                    docker_network=docker_network,
                )
                tool_seconds += time.monotonic() - tool_started
            counters[tool] = counters.get(tool, 0) + 1
            call_key = json.dumps({"tool": tool, "args": args}, sort_keys=True, ensure_ascii=True)
            tool_call_counts[call_key] = tool_call_counts.get(call_key, 0) + 1
            if max_repeated_tool_calls > 0 and tool_call_counts[call_key] >= max_repeated_tool_calls:
                stop_reason = "repeated_tool_call"
                cap_exhausted = True
                failure_category = "tool_loop"
                checker_errors = [f"same tool call repeated {tool_call_counts[call_key]} times"]
            if not result.get("ok"):
                counters["failed"] += 1
                failed_tool_call_counts[call_key] = failed_tool_call_counts.get(call_key, 0) + 1
                if (
                    max_repeated_failed_tool_calls > 0
                    and failed_tool_call_counts[call_key] >= max_repeated_failed_tool_calls
                ):
                    stop_reason = "repeated_failed_tool_call"
                    cap_exhausted = True
                    failure_category = "tool_error_loop"
                    checker_errors = [
                        f"same failed tool call repeated {failed_tool_call_counts[call_key]} times"
                    ]
            if logger:
                logger.log("tool_call", tool=tool, args=args, raw_tool=call.tool, raw_args=call.args)
                logger.log("tool_result", tool=tool, result=result)
            if native and call.raw.get("id"):
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.raw["id"],
                        "name": tool,
                        "content": json.dumps(result, ensure_ascii=True),
                    }
                )
            else:
                messages.append({"role": "user", "content": _tool_result_message(result)})
            if stop_reason in {"repeated_tool_call", "repeated_failed_tool_call"}:
                break
            if tool == "submit" and result.get("ok"):
                submitted = True
                stop_reason = "submitted"
                break
        if saw_valid_tool_call:
            consecutive_protocol_errors = 0
        else:
            consecutive_protocol_errors += 1
        if submitted or stop_reason in {"repeated_tool_call", "repeated_failed_tool_call"}:
            break

    if not submitted and failure_category is None:
        stop_reason = "max_iterations"
        cap_exhausted = True
        failure_category = "incomplete_solution"
        checker_errors = ["agent stopped without submit"]

    if data_dir is not None:
        restored = restore_seeded_inputs(task, data_dir, Path(workspace.root))
        if logger and restored:
            logger.log("inputs_restored", paths=restored)

    if submitted:
        checker_started = time.monotonic()
        checker_result = task.checker(Path(workspace.root))
        checker_seconds += time.monotonic() - checker_started
    else:
        checker_result = None
        if cap_exhausted:
            checker_started = time.monotonic()
            silent_checker_result = task.checker(Path(workspace.root))
            checker_seconds += time.monotonic() - checker_started
            if logger:
                logger.log("silent_checker_result", result=silent_checker_result.__dict__)
    if logger and checker_result:
        logger.log("final_checker_result", result=checker_result.__dict__)

    success = bool(checker_result and checker_result.success)
    notes = None
    if silent_checker_result:
        notes = (
            "unsubmitted_checker_success="
            f"{str(silent_checker_result.success).lower()}; "
            f"unsubmitted_failure_category={silent_checker_result.failure_category}"
        )
    summary = AttemptSummary(
        run_id=run_id,
        task_id=task.task_id,
        model_id=model_id,
        mode="agent_tools",
        repetition=repetition,
        success=success,
        partial_score=checker_result.partial_score if checker_result else 0.0,
        failure_category=checker_result.failure_category if checker_result else failure_category,
        checker_errors=checker_result.checker_errors if checker_result else checker_errors,
        time_to_submission_seconds=time.monotonic() - start,
        timeout_reached=timeout_reached,
        stop_reason=stop_reason,
        cap_exhausted=cap_exhausted,
        n_model_calls=model_calls,
        n_tool_calls_total=tool_calls,
        n_read_calls=counters["read"],
        n_write_calls=counters["write"],
        n_edit_calls=counters["edit"],
        n_bash_calls=counters["bash"],
        n_submit_calls=counters["submit"],
        n_failed_tool_calls=counters["failed"],
        n_code_executions=counters["bash"],
        input_tokens_total=input_tokens,
        output_tokens_total=output_tokens,
        total_tokens=total_tokens,
        token_counts_estimated=token_counts_estimated,
        n_truncated_responses=n_truncated_responses,
        finish_reasons=finish_reasons,
        max_prompt_tokens=max_prompt_tokens,
        prompt_chars=prompt_chars,
        tool_protocol=tool_protocol,
        started_at=started_at,
        finished_at=datetime.now(timezone.utc).isoformat(),
        model_seconds=model_seconds,
        tool_seconds=tool_seconds,
        checker_seconds=checker_seconds,
        max_call_seconds=max_call_seconds,
        prefill_ms_total=prefill_ms,
        decode_ms_total=decode_ms,
        decode_tokens_per_second=(statistics.median(decode_tps) if decode_tps else None),
        prompt_tokens_cached_total=cached_tokens,
        cache_hit_rate=(cached_tokens / input_tokens if input_tokens else None),
        reasoning_chars_total=reasoning_chars,
        n_responses_with_reasoning=reasoning_responses,
        n_parallel_tool_calls=parallel_calls,
        first_silent_checker_success=bool(silent_checker_result and silent_checker_result.success),
        final_checker_success=success,
        notes=notes,
    )
    if logger:
        logger.log("attempt_finished", summary=summary.to_dict())
    return summary
