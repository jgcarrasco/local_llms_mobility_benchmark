"""Paper-ready plot generation."""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _placeholder_figure(path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.text(0.5, 0.5, "Pending benchmark results", ha="center", va="center")
    ax.set_axis_off()
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _bar_plot(df: pd.DataFrame, x: str, y: str, path: Path, title: str) -> None:
    if df.empty or x not in df or y not in df:
        _placeholder_figure(path, title)
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(df[x].astype(str), df[y])
    ax.set_title(title)
    ax.set_ylabel(y)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def make_paper_figures(
    *,
    aggregated_dir: str | Path,
    reports_dir: str | Path,
    paper_draft_dir: str | Path | None = None,
) -> dict[str, Path]:
    aggregated = Path(aggregated_dir)
    figure_dir = Path(reports_dir) / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "pass_rate_by_model": figure_dir / "fig_pass_rate_by_model.pdf",
        "agentic_gain": figure_dir / "fig_agentic_gain.pdf",
        "time_efficiency": figure_dir / "fig_time_efficiency.pdf",
        "failure_categories": figure_dir / "fig_failure_categories.pdf",
    }

    pass_rate = pd.read_csv(aggregated / "pass_rate_by_model_mode.csv") if (aggregated / "pass_rate_by_model_mode.csv").exists() else pd.DataFrame()
    gain = pd.read_csv(aggregated / "agentic_gain.csv") if (aggregated / "agentic_gain.csv").exists() else pd.DataFrame()
    efficiency = pd.read_csv(aggregated / "efficiency_metrics.csv") if (aggregated / "efficiency_metrics.csv").exists() else pd.DataFrame()
    failures = pd.read_csv(aggregated / "failure_categories.csv") if (aggregated / "failure_categories.csv").exists() else pd.DataFrame()

    _bar_plot(pass_rate, "model_id", "pass_rate", outputs["pass_rate_by_model"], "Pass rate by model")
    _bar_plot(gain, "model_id", "agentic_gain", outputs["agentic_gain"], "Agentic gain")
    _bar_plot(efficiency, "model_id", "time_efficiency", outputs["time_efficiency"], "Time efficiency")
    _bar_plot(failures, "failure_category", "count", outputs["failure_categories"], "Failure categories")

    if paper_draft_dir is not None:
        paper_figures = Path(paper_draft_dir) / "figures"
        paper_figures.mkdir(parents=True, exist_ok=True)
        for path in outputs.values():
            shutil.copy2(path, paper_figures / path.name)
    return outputs
