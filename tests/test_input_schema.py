from __future__ import annotations

import json
import zipfile
from pathlib import Path

import pandas as pd

from mobility_bench.modes.chat_schema import run_chat_schema_attempt
from mobility_bench.models.mock_client import MockClient
from mobility_bench.runtime.input_schema import build_input_schema
from mobility_bench.runtime.workspace import Workspace
from mobility_bench.scoring.base import CheckerResult
from mobility_bench.tasks.base import TaskSpec


def _checker(root: Path) -> CheckerResult:
    output = root / "outputs" / "answer.txt"
    if output.exists() and output.read_text(encoding="utf-8").strip() == "ok":
        return CheckerResult(success=True, partial_score=1.0)
    return CheckerResult(success=False, partial_score=0.0, failure_category="logic_error")


def _task(input_files: list[str], expected_outputs: list[str] | None = None) -> TaskSpec:
    return TaskSpec(
        task_id="TX",
        title="Toy task",
        family="test",
        input_files=input_files,
        expected_outputs=expected_outputs or ["outputs/answer.txt"],
        prompt_template="Write the expected output.",
        checker=_checker,
        timeout_seconds=60,
    )


def _seed_inputs(root: Path) -> None:
    inputs = root / "inputs"
    inputs.mkdir(parents=True)
    pd.DataFrame({"indicator_id": ["a", "b"], "value": [1.5, 2.5]}).to_csv(
        inputs / "table.csv", index=False
    )
    (inputs / "config.json").write_text(json.dumps({"city_qid": "Q1", "years": [2020]}), encoding="utf-8")
    with zipfile.ZipFile(inputs / "feed.zip", "w") as archive:
        frame = pd.DataFrame({"stop_id": ["s1"], "stop_name": ["A"]})
        archive.writestr("stops.txt", frame.to_csv(index=False))


def test_build_input_schema_describes_each_format(tmp_path: Path) -> None:
    _seed_inputs(tmp_path)
    task = _task(["inputs/table.csv", "inputs/config.json", "inputs/feed.zip"])
    schema = build_input_schema(tmp_path, task)

    assert "inputs/table.csv" in schema
    assert "indicator_id:" in schema
    assert "2 rows x 2 columns" in schema
    assert "inputs/config.json" in schema
    assert "city_qid" in schema
    assert "inputs/feed.zip" in schema
    assert "stops.txt: 1 rows x 2 columns" in schema


def test_build_input_schema_is_deterministic_and_bounded(tmp_path: Path) -> None:
    _seed_inputs(tmp_path)
    task = _task(["inputs/table.csv", "inputs/config.json", "inputs/feed.zip"])
    first = build_input_schema(tmp_path, task)
    second = build_input_schema(tmp_path, task)
    assert first == second

    bounded = build_input_schema(tmp_path, task, max_chars=80)
    assert len(bounded) <= 200
    assert "remaining input files omitted" in bounded


def test_build_input_schema_ignores_undeclared_files(tmp_path: Path) -> None:
    _seed_inputs(tmp_path)
    (tmp_path / "inputs" / "answer_key.json").write_text('{"query": "SELECT ?item"}', encoding="utf-8")
    task = _task(["inputs/config.json"])
    schema = build_input_schema(tmp_path, task)

    assert "inputs/config.json" in schema
    assert "answer_key" not in schema
    assert "inputs/table.csv" not in schema


def test_build_input_schema_handles_missing_inputs(tmp_path: Path) -> None:
    assert build_input_schema(tmp_path, _task(["inputs/absent.csv"])) == ""


def test_build_input_schema_covers_previous_outputs(tmp_path: Path) -> None:
    outputs = tmp_path / "outputs"
    outputs.mkdir()
    pd.DataFrame({"qid": ["Q1"], "distance_m": [12.0]}).to_csv(outputs / "upstream.csv", index=False)
    pd.DataFrame({"qid": ["Q1"]}).to_csv(outputs / "answer.txt", index=False)
    task = _task(["all previous outputs"], expected_outputs=["outputs/answer.txt"])
    schema = build_input_schema(tmp_path, task)

    assert "outputs/upstream.csv" in schema
    assert "distance_m:" in schema
    assert "outputs/answer.txt" not in schema


def test_chat_schema_includes_the_schema_and_reports_its_mode(tmp_path: Path) -> None:
    _seed_inputs(tmp_path)
    task = _task(["inputs/table.csv"])
    model = MockClient(
        [
            """<solution_file path="src/solution.py">
from pathlib import Path
Path("outputs").mkdir(exist_ok=True)
Path("outputs/answer.txt").write_text("ok")
</solution_file>"""
        ]
    )
    summary = run_chat_schema_attempt(
        task=task,
        model=model,
        workspace=Workspace(tmp_path),
        run_id="run-schema",
        model_id="mock",
        repetition=1,
    )

    prompt = model.calls[0][1]["content"]
    assert "Input data structure" in prompt
    assert "indicator_id:" in prompt
    assert summary.mode == "chat_schema"
    assert summary.success is True
    assert summary.prompt_chars == len(prompt) + len(model.calls[0][0]["content"])


def test_chat_schema_prompt_has_no_schema_section_without_inputs(tmp_path: Path) -> None:
    model = MockClient(["<solution_file path=\"src/solution.py\">\npass\n</solution_file>"])
    run_chat_schema_attempt(
        task=_task([]),
        model=model,
        workspace=Workspace(tmp_path),
        run_id="run-schema-empty",
        model_id="mock",
        repetition=1,
    )
    assert "Input data structure" not in model.calls[0][1]["content"]
