"""Result aggregation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from mobility_bench.scoring.metrics import agentic_gain, efficiency_metrics, pass_rate_by


ATTEMPT_COLUMNS = [
    "run_id",
    "task_id",
    "task_family",
    "model_id",
    "model_family",
    "model_size_b",
    "architecture",
    "quantization",
    "context_length",
    "source_repo",
    "artifact_repo",
    "mode",
    "repetition",
    "success",
    "partial_score",
    "failure_category",
    "checker_errors",
    "time_to_submission_seconds",
    "timeout_reached",
    "stop_reason",
    "cap_exhausted",
    "n_model_calls",
    "n_tool_calls_total",
    "n_read_calls",
    "n_write_calls",
    "n_edit_calls",
    "n_bash_calls",
    "n_submit_calls",
    "n_failed_tool_calls",
    "n_code_executions",
    "n_error_messages_observed",
    "input_tokens_total",
    "output_tokens_total",
    "total_tokens",
    "token_counts_estimated",
    "first_silent_checker_success",
    "final_checker_success",
    "infrastructure_failure",
    "notes",
]


def _normalise_record(record: dict[str, Any]) -> dict[str, Any]:
    if record.get("event_type") == "attempt_finished" and isinstance(record.get("summary"), dict):
        record = record["summary"]
    normalised = {column: record.get(column) for column in ATTEMPT_COLUMNS}
    normalised["success"] = bool(normalised["success"])
    return normalised


def load_attempts(results_dir: str | Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(Path(results_dir).glob("*.jsonl")):
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(_normalise_record(json.loads(line)))
    return pd.DataFrame(rows, columns=ATTEMPT_COLUMNS)


def write_sweet_spot_summary(df: pd.DataFrame, output_path: Path) -> None:
    summary: dict[str, Any] = {
        "threshold_overall_pass_rate": 0.85,
        "threshold_time_multiplier": 2.0,
        "sweet_spot_model": None,
        "note": "No model satisfies the threshold or no attempts are available.",
    }
    if not df.empty:
        by_model = pass_rate_by(df, ["model_id"])
        infra = df.groupby("model_id", dropna=False)["infrastructure_failure"].mean().reset_index()
        times = df.groupby("model_id", dropna=False)["time_to_submission_seconds"].median().reset_index()
        merged = by_model.merge(infra, on="model_id", how="left").merge(times, on="model_id", how="left")
        best_time = merged["time_to_submission_seconds"].min()
        candidates = merged[
            (merged["pass_rate"] >= 0.85)
            & (merged["time_to_submission_seconds"] <= 2 * best_time)
            & (merged["infrastructure_failure"].fillna(0) == 0)
        ]
        if not candidates.empty:
            candidates = candidates.sort_values(["pass_rate", "time_to_submission_seconds"], ascending=[False, True])
            summary["sweet_spot_model"] = candidates.iloc[0]["model_id"]
            summary["note"] = "A configured model satisfied the sweet-spot thresholds."
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=True), encoding="utf-8")


def aggregate_results(results_dir: str | Path, output_dir: str | Path) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    attempts = load_attempts(results_dir)

    outputs = {
        "attempts": output / "attempts.csv",
        "pass_rate_by_model_mode": output / "pass_rate_by_model_mode.csv",
        "pass_rate_by_task": output / "pass_rate_by_task.csv",
        "agentic_gain": output / "agentic_gain.csv",
        "efficiency_metrics": output / "efficiency_metrics.csv",
        "failure_categories": output / "failure_categories.csv",
        "sweet_spot_summary": output / "sweet_spot_summary.json",
    }

    attempts.to_csv(outputs["attempts"], index=False)
    pass_rate_by(attempts, ["model_id", "mode"]).to_csv(outputs["pass_rate_by_model_mode"], index=False)
    pass_rate_by(attempts, ["task_id"]).to_csv(outputs["pass_rate_by_task"], index=False)
    agentic_gain(attempts).to_csv(outputs["agentic_gain"], index=False)
    efficiency_metrics(attempts).to_csv(outputs["efficiency_metrics"], index=False)
    if attempts.empty:
        failure_categories = pd.DataFrame(columns=["failure_category", "count"])
    else:
        failure_categories = attempts["failure_category"].fillna("success").value_counts().reset_index()
        failure_categories.columns = ["failure_category", "count"]
    failure_categories.to_csv(outputs["failure_categories"], index=False)
    write_sweet_spot_summary(attempts, outputs["sweet_spot_summary"])
    return outputs


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="runs/results_jsonl")
    parser.add_argument("--output-dir", default="runs/aggregated")
    args = parser.parse_args()
    aggregate_results(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
