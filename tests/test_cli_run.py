import json
from pathlib import Path
from typing import Any

from mobility_bench.cli import main
from mobility_bench.models.base import ModelResponse


def test_cli_run_mock_writes_result_jsonl(tmp_path: Path) -> None:
    benchmark = tmp_path / "benchmark.yaml"
    benchmark.write_text(
        f"""
runtime:
  temperature: 0.0
  max_iterations: 1
  max_tool_calls: 1
  timeout_seconds_per_task: 60
  bash_timeout_seconds: 5
  repetitions: 1
  network_enabled: false
  clean_workspace_per_attempt: true
paths:
  data_dir: data
  runs_dir: {tmp_path / "runs"}
  reports_dir: {tmp_path / "reports"}
execution:
  backend: local
  docker_image: mobility-bench-runner:latest
  docker_network: none
  expose_sensitive_env: false
""",
        encoding="utf-8",
    )
    code = main(
        [
            "run",
            "--models",
            "configs/models.yaml",
            "--benchmark",
            str(benchmark),
            "--tasks",
            "T01",
            "--model-id",
            "mock",
            "--mode",
            "chat_only",
            "--repetitions",
            "1",
        ]
    )
    assert code == 0
    result_files = list((tmp_path / "runs" / "results_jsonl").glob("*.jsonl"))
    assert len(result_files) == 1
    record = json.loads(result_files[0].read_text(encoding="utf-8"))
    assert record["task_id"] == "T01"
    assert record["failure_category"] == "format_error"


def test_cli_run_seeds_declared_reference_inputs(tmp_path: Path) -> None:
    benchmark = tmp_path / "benchmark.yaml"
    benchmark.write_text(
        f"""
runtime:
  temperature: 0.0
  max_iterations: 1
  max_tool_calls: 1
  timeout_seconds_per_task: 60
  bash_timeout_seconds: 5
  repetitions: 1
  network_enabled: false
  clean_workspace_per_attempt: true
paths:
  data_dir: data
  runs_dir: {tmp_path / "runs"}
  reports_dir: {tmp_path / "reports"}
execution:
  backend: local
  docker_image: mobility-bench-runner:latest
  docker_network: none
  expose_sensitive_env: false
""",
        encoding="utf-8",
    )
    code = main(
        [
            "run",
            "--models",
            "configs/models.yaml",
            "--benchmark",
            str(benchmark),
            "--tasks",
            "T02",
            "--model-id",
            "mock",
            "--mode",
            "chat_only",
            "--repetitions",
            "1",
        ]
    )
    assert code == 0
    workspace = tmp_path / "runs" / "workspaces" / "mock__chat_only__T02__rep01"
    assert (workspace / "outputs" / "t01_selected_indicators.csv").exists()
    assert not (workspace / "outputs" / "t02_eurostat_raw.csv").exists()


def test_cli_smoke_model_uses_configured_client(monkeypatch: Any, capsys: Any) -> None:
    observed: dict[str, Any] = {}

    class FakeClient:
        def generate(self, messages: list[dict[str, str]], **kwargs: Any) -> ModelResponse:
            observed["messages"] = messages
            observed["kwargs"] = kwargs
            return ModelResponse(
                text="ok",
                input_tokens=4,
                output_tokens=1,
                total_tokens=5,
                raw_response={"fake": True},
            )

    def fake_create_model_client(config: Any) -> FakeClient:
        observed["model_name"] = config.model_name
        return FakeClient()

    monkeypatch.setattr("mobility_bench.cli.create_model_client", fake_create_model_client)

    code = main(
        [
            "smoke-model",
            "--models",
            "configs/models.yaml",
            "--model-id",
            "qwen36_27b_ud_q4kxl",
            "--prompt",
            "ping",
            "--temperature",
            "0.1",
            "--max-tokens",
            "7",
        ]
    )

    assert code == 0
    assert observed["model_name"] == "qwen36-27b-ud-q4kxl"
    assert observed["messages"] == [{"role": "user", "content": "ping"}]
    assert observed["kwargs"] == {"temperature": 0.1, "max_tokens": 7}
    output = json.loads(capsys.readouterr().out)
    assert output["model_id"] == "qwen36_27b_ud_q4kxl"
    assert output["text"] == "ok"
    assert output["total_tokens"] == 5


def test_cli_create_sweep_manifest(tmp_path: Path) -> None:
    output = tmp_path / "manifest.json"

    code = main(
        [
            "create-sweep-manifest",
            "--models",
            "configs/models.yaml",
            "--benchmark",
            "configs/benchmark.yaml",
            "--output",
            str(output),
            "--notes",
            "cli test",
        ]
    )

    assert code == 0
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["notes"] == "cli test"
    assert data["planned_attempts"] == 810
