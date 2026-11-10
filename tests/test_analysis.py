import json
from pathlib import Path

import pandas as pd

from mobility_bench.analysis.aggregate_results import aggregate_results
from mobility_bench.analysis.paper_tables import make_paper_tables
from mobility_bench.analysis.plots import make_paper_figures
from mobility_bench.analysis.summary import make_benchmark_summary
from mobility_bench.analysis.sweep_manifest import create_sweep_manifest


def test_aggregate_results_writes_required_outputs(tmp_path: Path) -> None:
    results = tmp_path / "results"
    output = tmp_path / "aggregated"
    results.mkdir()
    rows = [
        {
            "run_id": "r1",
            "task_id": "T01",
            "model_id": "mock",
            "mode": "chat_only",
            "repetition": 1,
            "success": False,
            "partial_score": 0.0,
            "failure_category": "logic_error",
            "time_to_submission_seconds": 10,
            "total_tokens": 100,
            "infrastructure_failure": False,
        },
        {
            "run_id": "r2",
            "task_id": "T01",
            "model_id": "mock",
            "mode": "agent_tools",
            "repetition": 1,
            "success": True,
            "partial_score": 1.0,
            "failure_category": None,
            "time_to_submission_seconds": 5,
            "total_tokens": 200,
            "infrastructure_failure": False,
        },
    ]
    with (results / "attempts.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    outputs = aggregate_results(results, output)
    assert outputs["attempts"].exists()
    attempts = pd.read_csv(outputs["attempts"])
    assert "task_family" in attempts.columns
    assert "source_repo" in attempts.columns
    gain = pd.read_csv(outputs["agentic_gain"])
    assert gain.loc[0, "agentic_gain"] == 1.0
    categories = pd.read_csv(outputs["failure_categories"])
    assert "success" in set(categories["failure_category"])
    assert "none" not in set(categories["failure_category"])


def test_create_sweep_manifest_writes_reproducibility_metadata(tmp_path: Path) -> None:
    output = tmp_path / "manifest.json"

    manifest = create_sweep_manifest(
        benchmark_config="configs/benchmark.yaml",
        models_config="configs/models.yaml",
        output_path=output,
        notes="test manifest",
    )

    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["notes"] == "test manifest"
    assert data["modes"] == ["chat_only", "chat_schema", "agent_tools"]
    assert data["planned_attempts"] == 9 * 3 * 15 * 3
    assert data["models"]["qwen36_27b_ud_q4kxl"]["source_repo"] == "Qwen/Qwen3.6-27B"
    assert "nvidia_smi" in data["commands"]


def test_report_artifacts_are_created(tmp_path: Path) -> None:
    aggregated = tmp_path / "aggregated"
    reports = tmp_path / "reports"
    paper = tmp_path / "paper"
    aggregated.mkdir()
    pd.DataFrame({"model_id": ["mock"], "mode": ["agent_tools"], "pass_rate": [1.0]}).to_csv(
        aggregated / "pass_rate_by_model_mode.csv",
        index=False,
    )
    pd.DataFrame({"model_id": ["mock"], "task_id": ["T01"], "agentic_gain": [1.0]}).to_csv(
        aggregated / "agentic_gain.csv",
        index=False,
    )
    pd.DataFrame({"model_id": ["mock"], "mode": ["agent_tools"], "time_efficiency": [0.2]}).to_csv(
        aggregated / "efficiency_metrics.csv",
        index=False,
    )
    pd.DataFrame({"failure_category": ["none"], "count": [1]}).to_csv(
        aggregated / "failure_categories.csv",
        index=False,
    )

    tables = make_paper_tables(aggregated_dir=aggregated, reports_dir=reports, paper_draft_dir=paper)
    figures = make_paper_figures(aggregated_dir=aggregated, reports_dir=reports, paper_draft_dir=paper)

    assert tables["main_results"].exists()
    assert figures["agentic_gain"].exists()
    assert (paper / "tables" / "table_main_results.tex").exists()
    assert (paper / "figures" / "fig_agentic_gain.pdf").exists()


def test_benchmark_summary_is_created(tmp_path: Path) -> None:
    aggregated = tmp_path / "aggregated"
    reports = tmp_path / "reports"
    aggregated.mkdir()
    pd.DataFrame(
        {
            "run_id": ["r1"],
            "task_id": ["T01"],
            "model_id": ["mock"],
            "mode": ["agent_tools"],
            "success": [True],
        }
    ).to_csv(aggregated / "attempts.csv", index=False)
    pd.DataFrame({"model_id": ["mock"], "mode": ["agent_tools"], "pass_rate": [1.0]}).to_csv(
        aggregated / "pass_rate_by_model_mode.csv",
        index=False,
    )
    pd.DataFrame({"task_id": ["T01"], "pass_rate": [1.0]}).to_csv(
        aggregated / "pass_rate_by_task.csv",
        index=False,
    )
    pd.DataFrame({"model_id": ["mock"], "task_id": ["T01"], "agentic_gain": [1.0]}).to_csv(
        aggregated / "agentic_gain.csv",
        index=False,
    )
    pd.DataFrame({"failure_category": ["none"], "count": [1]}).to_csv(
        aggregated / "failure_categories.csv",
        index=False,
    )
    (aggregated / "sweet_spot_summary.json").write_text(
        json.dumps({"sweet_spot_model": "mock", "note": "A configured model satisfied the sweet-spot thresholds."}),
        encoding="utf-8",
    )

    summary = make_benchmark_summary(aggregated_dir=aggregated, reports_dir=reports)

    assert summary.exists()
    text = summary.read_text(encoding="utf-8")
    assert "# Benchmark Summary" in text
    assert "- Attempts: 1" in text
    assert "| mock | agent_tools | 1.0 |" in text


def test_empty_result_tables_keep_placeholders(tmp_path: Path) -> None:
    aggregated = tmp_path / "aggregated"
    reports = tmp_path / "reports"
    paper = tmp_path / "paper"
    aggregated.mkdir()
    pd.DataFrame(columns=["model_id", "mode", "attempts", "successful_attempts", "pass_rate"]).to_csv(
        aggregated / "pass_rate_by_model_mode.csv",
        index=False,
    )
    pd.DataFrame(columns=["model_id", "task_id", "agentic_gain"]).to_csv(
        aggregated / "agentic_gain.csv",
        index=False,
    )
    pd.DataFrame(columns=["failure_category", "count"]).to_csv(
        aggregated / "failure_categories.csv",
        index=False,
    )

    tables = make_paper_tables(aggregated_dir=aggregated, reports_dir=reports, paper_draft_dir=paper)

    main_results = tables["main_results"].read_text(encoding="utf-8")
    model_table = tables["models"].read_text(encoding="utf-8")
    assert "Pending benchmark results" in main_results
    assert "% RESULTS_PLACEHOLDER" in main_results
    assert "mock" not in model_table
