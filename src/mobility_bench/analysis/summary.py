"""Markdown benchmark summary generation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _markdown_table(df: pd.DataFrame, columns: list[str], max_rows: int = 8) -> list[str]:
    if df.empty:
        return ["Pending benchmark results."]
    available = [column for column in columns if column in df.columns]
    if not available:
        return ["Pending benchmark results."]
    view = df.loc[:, available].head(max_rows).copy()
    lines = [
        "| " + " | ".join(available) + " |",
        "| " + " | ".join("---" for _ in available) + " |",
    ]
    for _, row in view.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in available) + " |")
    return lines


def make_benchmark_summary(
    *,
    aggregated_dir: str | Path,
    reports_dir: str | Path,
) -> Path:
    aggregated = Path(aggregated_dir)
    reports = Path(reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    output = reports / "benchmark_summary.md"

    attempts = _read_csv(aggregated / "attempts.csv")
    pass_rates = _read_csv(aggregated / "pass_rate_by_model_mode.csv")
    task_rates = _read_csv(aggregated / "pass_rate_by_task.csv")
    agentic_gain = _read_csv(aggregated / "agentic_gain.csv")
    failures = _read_csv(aggregated / "failure_categories.csv")
    sweet_spot = _read_json(aggregated / "sweet_spot_summary.json")

    if attempts.empty:
        n_attempts = 0
        n_models = 0
        n_tasks = 0
        modes = "none"
    else:
        n_attempts = len(attempts)
        n_models = attempts["model_id"].nunique(dropna=True)
        n_tasks = attempts["task_id"].nunique(dropna=True)
        modes = ", ".join(sorted(str(mode) for mode in attempts["mode"].dropna().unique()))

    lines = [
        "# Benchmark Summary",
        "",
        "## Coverage",
        "",
        f"- Attempts: {n_attempts}",
        f"- Models: {n_models}",
        f"- Tasks: {n_tasks}",
        f"- Modes: {modes}",
        "",
        "## Model And Mode Pass Rates",
        "",
        *_markdown_table(pass_rates, ["model_id", "mode", "pass_rate", "n_attempts"]),
        "",
        "## Task Pass Rates",
        "",
        *_markdown_table(task_rates, ["task_id", "pass_rate", "n_attempts"]),
        "",
        "## Agentic Gain",
        "",
        *_markdown_table(agentic_gain, ["model_id", "task_id", "agentic_gain"]),
        "",
        "## Failure Categories",
        "",
        *_markdown_table(failures, ["failure_category", "count"]),
        "",
        "## Sweet Spot",
        "",
        f"- Model: {sweet_spot.get('sweet_spot_model') or 'none'}",
        f"- Note: {sweet_spot.get('note') or 'Pending benchmark results.'}",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8")
    return output
