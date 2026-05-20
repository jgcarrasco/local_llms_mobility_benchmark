from pathlib import Path
from typing import Any

from mobility_bench.runtime.extraction import extract_solution_files, extract_tool_calls
from mobility_bench.runtime.logger import EventLogger
from mobility_bench.runtime.sandbox import build_docker_command
from mobility_bench.runtime.workspace import Workspace, WorkspacePathError, create_attempt_workspace
from mobility_bench.tools import execute_tool
from mobility_bench.tools.bash import run_bash
from mobility_bench.tools.edit import edit_file
from mobility_bench.tools.read import read_file
from mobility_bench.tools.submit import submit
from mobility_bench.tools.write import write_file


def test_workspace_rejects_path_escape(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path)
    try:
        workspace.resolve_path("../outside.txt")
    except WorkspacePathError:
        pass
    else:
        raise AssertionError("expected workspace path escape to be rejected")


def test_create_attempt_workspace_makes_base_dirs(tmp_path: Path) -> None:
    workspace = create_attempt_workspace("run01", tmp_path)
    assert (workspace.root / "inputs").is_dir()
    assert (workspace.root / "outputs").is_dir()


def test_read_write_edit_tools(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path)
    result = write_file(workspace, "src/solution.py", "print('wrong')\n")
    assert result.ok

    edited = edit_file(workspace, "src/solution.py", "wrong", "right")
    assert edited.ok

    read = read_file(workspace, "src/solution.py")
    assert read["ok"] is True
    assert "right" in str(read["content"])


def test_read_tool_truncates_with_targeted_inspection_hint(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path)
    write_file(workspace, "large.txt", "abcdef" * 2000)
    result = read_file(workspace, "large.txt", max_chars=10)

    assert result["ok"] is True
    assert result["content"] == "abcdefabcd"
    assert result["truncated"] is True
    assert result["metadata"]["total_chars"] == 12000
    assert "targeted inspection" in result["metadata"]["hint"]


def test_bash_tool_runs_inside_workspace(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path)
    write_file(workspace, "hello.txt", "hello")
    result = run_bash(workspace, "python3 -c \"from pathlib import Path; print(Path('hello.txt').read_text())\"")
    assert result.ok
    assert result.stdout.strip() == "hello"


def test_bash_tool_accepts_cmd_alias(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path)
    result = execute_tool(workspace, "bash", {"cmd": "python3 -c \"print('ok')\""})
    assert result["ok"] is True
    assert result["stdout"].strip() == "ok"


def test_bash_tool_strips_sensitive_environment(tmp_path: Path, monkeypatch: Any) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret-value")
    workspace = Workspace(tmp_path)
    result = run_bash(workspace, "python3 -c \"import os; print(os.getenv('OPENAI_API_KEY'))\"")
    assert result.ok
    assert "secret-value" not in result.stdout
    assert result.stdout.strip() == "None"


def test_bash_tool_rejects_host_absolute_paths(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path)
    result = run_bash(workspace, "bash -c \"source /home/other/project/venv/bin/activate\"")
    assert result.ok is False
    assert "outside workspace" in result.stderr or "host Python environment" in result.stderr
    assert result.metadata["recoverable_policy_error"] is True


def test_bash_tool_rejects_filesystem_root_scan(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path)
    result = run_bash(workspace, "find / -maxdepth 2 -type f")
    assert result.ok is False
    assert "filesystem root" in result.stderr


def test_bash_tool_rejects_pip_install(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path)
    result = run_bash(workspace, "python3 -m pip install pandas")
    assert result.ok is False
    assert "host Python environment" in result.stderr


def test_submit_tool_marks_submission() -> None:
    result = submit("done")
    assert result.ok
    assert result.metadata["submitted"] is True


def test_extract_tool_calls() -> None:
    calls = extract_tool_calls(
        '<tool_call>\n{"tool": "read", "args": {"path": "inputs/a.csv"}}\n</tool_call>'
    )
    assert calls[0].tool == "read"
    assert calls[0].args["path"] == "inputs/a.csv"


def test_extract_tool_calls_recovers_malformed_command_json() -> None:
    calls = extract_tool_calls(
        '<tool_call>\n{"tool":": {"command": "python src/solution.py"}}\n</tool_call>'
    )
    assert calls[0].tool == "bash"
    assert calls[0].args["command"] == "python src/solution.py"


def test_extract_tool_calls_accepts_top_level_args() -> None:
    calls = extract_tool_calls(
        '<tool_call>\n{"tool": "bash", "command": "head -20 outputs/data.csv"}\n</tool_call>'
    )
    assert calls[0].tool == "bash"
    assert calls[0].args["command"] == "head -20 outputs/data.csv"


def test_extract_tool_calls_recovers_malformed_edit_json() -> None:
    calls = extract_tool_calls(
        "<tool_call>\n"
        '{"tool": "edit", {"path": "src/solution.py", "old": "hours > 24", '
        '"new": "hours >= 24", "replace_all": false}}\n'
        "</tool_call>"
    )
    assert calls[0].tool == "edit"
    assert calls[0].args["path"] == "src/solution.py"
    assert calls[0].args["old"] == "hours > 24"
    assert calls[0].args["new"] == "hours >= 24"
    assert calls[0].args["replace_all"] is False


def test_extract_tool_calls_returns_invalid_call_for_unrecoverable_json() -> None:
    calls = extract_tool_calls('<tool_call>\n{"tool": "write", "args": }\n</tool_call>')
    assert calls[0].tool == "__invalid__"
    assert "invalid tool-call JSON" in calls[0].args["error"]


def test_tool_aliases_and_argument_aliases_are_accepted(tmp_path: Path) -> None:
    workspace = Workspace(tmp_path)
    write_file(workspace, "hello.txt", "hello")
    read = execute_tool(workspace, "read_file", {"file": "hello.txt"})
    assert read["ok"] is True
    assert read["content"] == "hello"
    bash = execute_tool(workspace, "run", {"cmd": "python3 -c \"print('ok')\""})
    assert bash["ok"] is True
    assert bash["stdout"].strip() == "ok"


def test_extract_solution_files() -> None:
    files = extract_solution_files(
        '<solution_file path="src/solution.py">\nprint("ok")\n</solution_file>'
    )
    assert files[0].path == "src/solution.py"
    assert 'print("ok")' in files[0].content


def test_event_logger_writes_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    logger = EventLogger(path)
    logger.log("attempt_started", run_id="run01")
    assert path.read_text(encoding="utf-8").count("\n") == 1


def test_event_logger_resets_existing_log_by_default(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text('{"event_type": "old"}\n', encoding="utf-8")
    logger = EventLogger(path)
    logger.log("attempt_started", run_id="run02")
    text = path.read_text(encoding="utf-8")
    assert "old" not in text
    assert text.count("\n") == 1


def test_docker_command_is_network_disabled_by_default(tmp_path: Path) -> None:
    command = build_docker_command(command="python src/solution.py", workspace=tmp_path, image="runner")
    assert "--network" in command
    assert "none" in command
    assert f"{tmp_path.resolve()}:/workspace" in command
