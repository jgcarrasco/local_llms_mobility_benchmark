from pathlib import Path
from typing import Any

from mobility_bench.models.base import ModelResponse
from mobility_bench.models.mock_client import MockClient
from mobility_bench.modes.agent_tools import AGENT_SYSTEM_PROMPT, run_agent_tools_attempt
from mobility_bench.modes.chat_only import run_chat_only_attempt
from mobility_bench.runtime.workspace import Workspace
from mobility_bench.scoring.base import CheckerResult
from mobility_bench.tasks.base import TaskSpec


class RecordingClient:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.kwargs: list[dict[str, Any]] = []

    def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> ModelResponse:
        del messages
        self.kwargs.append(kwargs)
        return ModelResponse(text=self.responses.pop(0), token_counts_estimated=True)


def toy_checker(root: Path) -> CheckerResult:
    output = root / "outputs" / "answer.txt"
    if output.exists() and output.read_text(encoding="utf-8").strip() == "ok":
        return CheckerResult(success=True, partial_score=1.0)
    return CheckerResult(
        success=False,
        partial_score=0.0,
        checker_errors=["missing or invalid output"],
        failure_category="logic_error",
    )


def toy_task() -> TaskSpec:
    return TaskSpec(
        task_id="TX",
        title="Toy task",
        family="test",
        input_files=[],
        expected_outputs=["outputs/answer.txt"],
        prompt_template="Write the expected output.",
        checker=toy_checker,
        timeout_seconds=60,
    )


def test_chat_only_mock_run_passes(tmp_path: Path) -> None:
    model = MockClient(
        [
            """<solution_file path="src/solution.py">
from pathlib import Path
Path("outputs").mkdir(exist_ok=True)
Path("outputs/answer.txt").write_text("ok")
</solution_file>"""
        ]
    )
    summary = run_chat_only_attempt(
        task=toy_task(),
        model=model,
        workspace=Workspace(tmp_path),
        run_id="run01",
        model_id="mock",
        repetition=1,
    )
    assert summary.success is True
    assert summary.final_checker_success is True
    assert '<solution_file path="src/solution.py">' in model.calls[0][1]["content"]
    assert "Do not put shell commands or prose inside a solution block." in model.calls[0][1]["content"]


def test_chat_only_passes_temperature_to_model(tmp_path: Path) -> None:
    model = RecordingClient(
        [
            """<solution_file path="src/solution.py">
from pathlib import Path
Path("outputs").mkdir(exist_ok=True)
Path("outputs/answer.txt").write_text("ok")
</solution_file>"""
        ]
    )
    run_chat_only_attempt(
        task=toy_task(),
        model=model,
        workspace=Workspace(tmp_path),
        run_id="run-temp-chat",
        model_id="mock",
        repetition=1,
        temperature=0.2,
    )
    assert model.kwargs == [{"temperature": 0.2}]


def test_chat_only_timeout_is_reported(tmp_path: Path) -> None:
    model = MockClient(
        [
            """<solution_file path="outputs/answer.txt">
ok
</solution_file>"""
        ]
    )
    summary = run_chat_only_attempt(
        task=toy_task(),
        model=model,
        workspace=Workspace(tmp_path),
        run_id="run-timeout",
        model_id="mock",
        repetition=1,
        timeout_seconds=0,
    )
    assert summary.success is False
    assert summary.timeout_reached is True
    assert summary.failure_category == "timeout"


def test_agent_tools_mock_run_passes(tmp_path: Path) -> None:
    model = MockClient(
        [
            '<tool_call>\n{"tool": "write", "args": {"path": "outputs/answer.txt", "content": "ok"}}\n</tool_call>',
            '<tool_call>\n{"tool": "submit", "args": {"message": "done"}}\n</tool_call>',
        ]
    )
    summary = run_agent_tools_attempt(
        task=toy_task(),
        model=model,
        workspace=Workspace(tmp_path),
        run_id="run02",
        model_id="mock",
        repetition=1,
    )
    assert summary.success is True
    assert summary.n_write_calls == 1
    assert summary.n_submit_calls == 1


def test_agent_tools_records_valid_artifact_without_submit(tmp_path: Path) -> None:
    model = MockClient(
        [
            '<tool_call>\n{"tool": "write", "args": {"path": "outputs/answer.txt", "content": "ok"}}\n</tool_call>',
        ]
    )
    summary = run_agent_tools_attempt(
        task=toy_task(),
        model=model,
        workspace=Workspace(tmp_path),
        run_id="run-valid-no-submit",
        model_id="mock",
        repetition=1,
        max_iterations=1,
    )
    assert summary.success is False
    assert summary.failure_category == "incomplete_solution"
    assert summary.stop_reason == "max_iterations"
    assert summary.cap_exhausted is True
    assert summary.first_silent_checker_success is True
    assert summary.final_checker_success is False
    assert summary.notes == "unsubmitted_checker_success=true; unsubmitted_failure_category=None"


def test_agent_tools_stops_protocol_stall(tmp_path: Path) -> None:
    model = MockClient(["not a tool call", "still not a tool call", "again not a tool call"])
    summary = run_agent_tools_attempt(
        task=toy_task(),
        model=model,
        workspace=Workspace(tmp_path),
        run_id="run-protocol-stall",
        model_id="mock",
        repetition=1,
        max_iterations=10,
        max_consecutive_protocol_errors=2,
    )

    assert summary.success is False
    assert summary.failure_category == "protocol_error"
    assert summary.stop_reason == "protocol_stall"
    assert summary.cap_exhausted is True
    assert summary.n_model_calls == 2
    assert summary.n_failed_tool_calls == 2


def test_agent_tools_counts_invalid_tool_calls_as_protocol_stall(tmp_path: Path) -> None:
    model = MockClient(
        [
            '<tool_call>\n{"tool": {"args": {"path": "inputs/file.csv"}}\n</tool_call>',
            '<tool_call>\n{"tool": {"args": {"path": "inputs/file.csv"}}\n</tool_call>',
            '<tool_call>\n{"tool": {"args": {"path": "inputs/file.csv"}}\n</tool_call>',
        ]
    )
    summary = run_agent_tools_attempt(
        task=toy_task(),
        model=model,
        workspace=Workspace(tmp_path),
        run_id="run-invalid-tool-protocol-stall",
        model_id="mock",
        repetition=1,
        max_iterations=10,
        max_consecutive_protocol_errors=2,
    )

    assert summary.success is False
    assert summary.failure_category == "protocol_error"
    assert summary.stop_reason == "protocol_stall"
    assert summary.cap_exhausted is True
    assert summary.n_model_calls == 2
    assert summary.n_tool_calls_total == 2
    assert summary.n_failed_tool_calls == 2


def test_agent_tools_stops_oversized_response_without_tool_call(tmp_path: Path) -> None:
    class TokenClient:
        def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> ModelResponse:
            del messages, kwargs
            return ModelResponse(
                text="not a tool call",
                output_tokens=10,
                total_tokens=10,
                token_counts_estimated=False,
            )

    summary = run_agent_tools_attempt(
        task=toy_task(),
        model=TokenClient(),
        workspace=Workspace(tmp_path),
        run_id="run-protocol-bloat",
        model_id="mock",
        repetition=1,
        max_iterations=10,
        max_protocol_response_tokens=10,
    )

    assert summary.success is False
    assert summary.failure_category == "protocol_error"
    assert summary.stop_reason == "protocol_bloat"
    assert summary.cap_exhausted is True
    assert summary.n_model_calls == 1
    assert summary.n_failed_tool_calls == 0


def test_agent_tools_stops_repeated_failed_tool_call(tmp_path: Path) -> None:
    model = MockClient(
        [
            '<tool_call>\n{"tool": "bash", "args": {"command": "python missing.py"}}\n</tool_call>',
            '<tool_call>\n{"tool": "bash", "args": {"command": "python missing.py"}}\n</tool_call>',
            '<tool_call>\n{"tool": "bash", "args": {"command": "python missing.py"}}\n</tool_call>',
        ]
    )
    summary = run_agent_tools_attempt(
        task=toy_task(),
        model=model,
        workspace=Workspace(tmp_path),
        run_id="run-repeated-failed-tool",
        model_id="mock",
        repetition=1,
        max_iterations=10,
        max_repeated_failed_tool_calls=2,
    )

    assert summary.success is False
    assert summary.failure_category == "tool_error_loop"
    assert summary.stop_reason == "repeated_failed_tool_call"
    assert summary.cap_exhausted is True
    assert summary.n_model_calls == 2
    assert summary.n_bash_calls == 2
    assert summary.n_failed_tool_calls == 2


def test_agent_tools_stops_repeated_tool_call(tmp_path: Path) -> None:
    model = MockClient(
        [
            '<tool_call>\n{"tool": "bash", "args": {"command": "true"}}\n</tool_call>',
            '<tool_call>\n{"tool": "bash", "args": {"command": "true"}}\n</tool_call>',
            '<tool_call>\n{"tool": "bash", "args": {"command": "true"}}\n</tool_call>',
        ]
    )
    summary = run_agent_tools_attempt(
        task=toy_task(),
        model=model,
        workspace=Workspace(tmp_path),
        run_id="run-repeated-tool",
        model_id="mock",
        repetition=1,
        max_iterations=10,
        max_repeated_tool_calls=2,
    )

    assert summary.success is False
    assert summary.failure_category == "tool_loop"
    assert summary.stop_reason == "repeated_tool_call"
    assert summary.cap_exhausted is True
    assert summary.n_model_calls == 2
    assert summary.n_bash_calls == 2
    assert summary.n_failed_tool_calls == 0


def test_agent_prompt_includes_tool_argument_schemas() -> None:
    assert '- read: {"path": "relative/path.txt"}' in AGENT_SYSTEM_PROMPT
    assert '{"path": "relative/path.txt", "max_chars": 12000}' in AGENT_SYSTEM_PROMPT
    assert '- bash: {"command": "python src/solution.py"}' in AGENT_SYSTEM_PROMPT
    assert '- submit: {"message": "done"}' in AGENT_SYSTEM_PROMPT
    assert '"tool" field must be a string' in AGENT_SYSTEM_PROMPT
    assert "Reply with exactly one <tool_call> block per turn." in AGENT_SYSTEM_PROMPT
    assert "Do not copy tool-result content into your reply" in AGENT_SYSTEM_PROMPT
    assert "use write to create a script, then bash to run it" in AGENT_SYSTEM_PROMPT


def test_agent_tools_passes_temperature_to_model(tmp_path: Path) -> None:
    model = RecordingClient(
        [
            '<tool_call>\n{"tool": "submit", "args": {"message": "done"}}\n</tool_call>',
        ]
    )
    run_agent_tools_attempt(
        task=toy_task(),
        model=model,
        workspace=Workspace(tmp_path),
        run_id="run-temp-agent",
        model_id="mock",
        repetition=1,
        temperature=0.3,
    )
    assert model.kwargs == [{"temperature": 0.3}]
