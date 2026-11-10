from __future__ import annotations

from pathlib import Path

from mobility_bench.models.mock_client import MockClient
from mobility_bench.modes.chat_only import run_chat_only_attempt
from mobility_bench.runtime.inputs import restore_seeded_inputs
from mobility_bench.runtime.workspace import Workspace
from mobility_bench.scoring.base import CheckerResult
from mobility_bench.tasks.base import TaskSpec


def _task(input_files: list[str], expected_outputs: list[str]) -> TaskSpec:
    return TaskSpec(
        task_id="TX",
        title="test",
        family="test",
        input_files=input_files,
        expected_outputs=expected_outputs,
        prompt_template="test",
        checker=lambda root: CheckerResult(success=True, partial_score=1.0),
        timeout_seconds=1,
    )


def test_restore_seeded_inputs_rewrites_only_modified_files(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    (data_dir / "reference_outputs" / "outputs").mkdir(parents=True)
    (data_dir / "inputs" / "eurostat").mkdir(parents=True)
    (data_dir / "reference_outputs" / "outputs" / "upstream.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (data_dir / "inputs" / "eurostat" / "sample.csv").write_text("x\n1\n", encoding="utf-8")

    workspace = tmp_path / "workspace"
    (workspace / "outputs").mkdir(parents=True)
    (workspace / "inputs" / "eurostat").mkdir(parents=True)
    (workspace / "outputs" / "upstream.csv").write_text("tampered\n", encoding="utf-8")
    (workspace / "inputs" / "eurostat" / "sample.csv").write_text("x\n1\n", encoding="utf-8")

    task = _task(
        ["outputs/upstream.csv", "inputs/eurostat/sample.csv"],
        ["outputs/result.csv"],
    )

    restored = restore_seeded_inputs(task, data_dir, workspace)

    assert restored == ["outputs/upstream.csv"]
    assert (workspace / "outputs" / "upstream.csv").read_text(encoding="utf-8") == "a,b\n1,2\n"


def test_restore_seeded_inputs_keeps_the_task_outputs(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    (data_dir / "reference_outputs" / "outputs").mkdir(parents=True)
    (data_dir / "reference_outputs" / "src").mkdir(parents=True)
    (data_dir / "reference_outputs" / "outputs" / "upstream.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (data_dir / "reference_outputs" / "src" / "loader.py").write_text("# reference loader\n", encoding="utf-8")

    workspace = tmp_path / "workspace"
    (workspace / "outputs").mkdir(parents=True)
    (workspace / "src").mkdir(parents=True)
    (workspace / "outputs" / "upstream.csv").write_text("tampered\n", encoding="utf-8")
    (workspace / "outputs" / "result.csv").write_text("model output\n", encoding="utf-8")
    (workspace / "src" / "loader.py").write_text("# reference loader\n", encoding="utf-8")

    task = _task(["all previous outputs"], ["outputs/result.csv"])

    restored = restore_seeded_inputs(task, data_dir, workspace)

    assert restored == ["outputs/upstream.csv"]
    assert (workspace / "outputs" / "upstream.csv").read_text(encoding="utf-8") == "a,b\n1,2\n"
    assert (workspace / "outputs" / "result.csv").read_text(encoding="utf-8") == "model output\n"

def test_chat_only_restores_a_tampered_input_before_scoring(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    (data_dir / "reference_outputs" / "outputs").mkdir(parents=True)
    (data_dir / "reference_outputs" / "outputs" / "upstream.txt").write_text("reference\n", encoding="utf-8")

    workspace_root = tmp_path / "workspace"
    (workspace_root / "outputs").mkdir(parents=True)
    (workspace_root / "outputs" / "upstream.txt").write_text("reference\n", encoding="utf-8")

    def checker(root: Path) -> CheckerResult:
        upstream = (root / "outputs" / "upstream.txt").read_text(encoding="utf-8")
        answer = root / "outputs" / "answer.txt"
        if upstream == "reference\n" and answer.exists():
            return CheckerResult(success=True, partial_score=1.0)
        return CheckerResult(
            success=False,
            partial_score=0.0,
            checker_errors=["input was modified"],
            failure_category="logic_error",
        )

    task = TaskSpec(
        task_id="TX",
        title="toy",
        family="test",
        input_files=["outputs/upstream.txt"],
        expected_outputs=["outputs/answer.txt"],
        prompt_template="write the answer",
        checker=checker,
        timeout_seconds=60,
    )
    model = MockClient(
        [
            """<solution_file path="src/solution.py">
from pathlib import Path
Path("outputs/upstream.txt").write_text("tampered")
Path("outputs/answer.txt").write_text("ok")
</solution_file>"""
        ]
    )

    summary = run_chat_only_attempt(
        task=task,
        model=model,
        workspace=Workspace(workspace_root),
        run_id="run01",
        model_id="mock",
        repetition=1,
        data_dir=data_dir,
    )

    assert summary.success is True
    assert (workspace_root / "outputs" / "upstream.txt").read_text(encoding="utf-8") == "reference\n"
