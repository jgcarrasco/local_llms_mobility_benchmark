#!/usr/bin/env python3
"""Generate paper figures from full-sweep aggregated results."""

from __future__ import annotations

import sys
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd

# Matplotlib style for paper
plt.rcParams.update({
    "figure.dpi": 300,
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 7,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def _model_label(model_id: str) -> str:
    """Short human-readable model label."""
    mapping = {
        "qwen35_2b_q4": "Q3.5 2B Q4",
        "qwen35_2b_bf16": "Q3.5 2B BF16",
        "qwen35_4b_q4": "Q3.5 4B Q4",
        "qwen35_4b_bf16": "Q3.5 4B BF16",
        "qwen35_9b_q4": "Q3.5 9B Q4",
        "qwen35_9b_bf16": "Q3.5 9B BF16",
        "qwen35_27b_q4": "Q3.5 27B Q4",
        "qwen35_35b_a3b_q4": "Q3.5 35B MoE",
        "qwen36_27b_q4": "Q3.6 27B Q4",
        "qwen36_35b_a3b_q4": "Q3.6 35B MoE",
    }
    return mapping.get(model_id, model_id)


def _model_sort_key(model_id: str) -> tuple:
    """Sort key: family, size, quantization."""
    family = 0 if "qwen35" in model_id else 1
    size = 0
    if "2b" in model_id:
        size = 2
    elif "4b" in model_id:
        size = 4
    elif "9b" in model_id:
        size = 9
    elif "27b" in model_id:
        size = 27
    elif "35b" in model_id:
        size = 35
    quant = 0 if "q4" in model_id else 1
    return (family, size, quant)


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson confidence interval for a binomial proportion."""
    if total == 0:
        return (float("nan"), float("nan"))
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half_width = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return center - half_width, center + half_width


def generate_heatmaps(attempts: pd.DataFrame, out_dir: Path) -> Path:
    """Heatmap of pass rate by selected model/task condition, one per mode."""
    task_order = [f"T{i:02d}" for i in range(1, 16)]
    attempts = attempts[
        (attempts["quantization"] == "UD-Q4_K_XL")
        | (attempts["model_id"] == "qwen35_4b_bf16")
    ]
    model_order = sorted(attempts["model_id"].unique(), key=_model_sort_key)

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.8), sharex=True, constrained_layout=True)

    for ax, mode in zip(axes, ["chat_only", "agent_tools"]):
        df = attempts[attempts["mode"] == mode]
        grouped = (
            df.groupby(["model_id", "task_id"])["success"]
            .mean()
            .unstack(fill_value=0)
            .reindex(index=model_order, columns=task_order, fill_value=0)
        )

        im = ax.imshow(grouped.values, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_xticks(range(len(task_order)))
        ax.set_xticklabels(task_order, rotation=45, ha="right")
        ax.set_yticks(range(len(model_order)))
        ax.set_yticklabels([_model_label(m) for m in model_order])
        for label in ax.get_yticklabels():
            label.set_ha("left")
            label.set_x(-0.14)
        ax.set_title("Chat-only" if mode == "chat_only" else "Agent mode")
        ax.set_xlabel("Task")

        # Add text annotations
        for i in range(len(model_order)):
            for j in range(len(task_order)):
                val = grouped.values[i, j]
                text = f"{val:.0%}" if val > 0 else "0"
                color = "white" if val < 0.35 or val > 0.85 else "black"
                ax.text(j, i, text, ha="center", va="center", color=color, fontsize=6)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation="vertical", fraction=0.025, pad=0.01)
    cbar.set_label("Pass rate")
    path = out_dir / "fig_heatmaps.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def generate_efficiency_scatter(attempts: pd.DataFrame, out_dir: Path) -> Path:
    """Scatter plot: pass rate vs runtime for agent-mode runs."""
    attempts = attempts[
        (attempts["mode"] == "agent_tools")
        & (
            (attempts["quantization"] == "UD-Q4_K_XL")
            | (attempts["model_id"] == "qwen35_4b_bf16")
        )
    ]
    # Compute per-model metrics.
    metrics = []
    for model_id, g in attempts.groupby("model_id"):
        pass_rate = g["success"].mean()
        med_time = g["time_to_submission_seconds"].median()
        arch = g["architecture"].iloc[0]
        size_b = g["model_size_b"].iloc[0]
        metrics.append({
            "model_id": model_id,
            "pass_rate": pass_rate,
            "med_time": med_time,
            "size_b": size_b,
            "architecture": arch,
        })
    df = pd.DataFrame(metrics)

    fig, ax = plt.subplots(figsize=(5.8, 3.6))

    scatter_for_colorbar = None
    min_size = df["size_b"].min()
    max_size = df["size_b"].max()
    marker_sizes = 45 + 140 * (df["size_b"] - min_size) / (max_size - min_size)
    df = df.assign(marker_size=marker_sizes)
    for arch, marker, label in [("dense", "o", "Dense"), ("moe", "D", "MoE")]:
        s = df[df["architecture"] == arch]
        scatter_for_colorbar = ax.scatter(
            s["med_time"],
            s["pass_rate"] * 100,
            c=s["size_b"],
            s=s["marker_size"],
            marker=marker,
            cmap="viridis",
            vmin=min_size,
            vmax=max_size,
            alpha=0.85,
            edgecolor="black",
            linewidth=0.35,
            label=label,
        )
        for _, row in s.iterrows():
            ax.annotate(
                _model_label(row["model_id"]),
                (row["med_time"], row["pass_rate"] * 100),
                textcoords="offset points",
                xytext=(4, 4),
                ha="left",
                fontsize=6,
                alpha=0.75,
                clip_on=True,
            )

    cbar = fig.colorbar(scatter_for_colorbar, ax=ax, fraction=0.045, pad=0.025)
    cbar.set_label("Model size (B parameters)")
    ax.set_xlabel("Median time to submission (s)")
    ax.set_ylabel("Pass rate (%)")
    ax.set_title("Agent mode accuracy-latency trade-off")
    ax.legend(loc="lower right", ncol=2, frameon=True, framealpha=0.92, borderpad=0.35)
    ax.set_xlim(df["med_time"].min() - 1.0, df["med_time"].max() + 8.0)
    ax.set_ylim(max(0, df["pass_rate"].min() * 100 - 8.0), min(100, df["pass_rate"].max() * 100 + 8.0))
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    fig.tight_layout()
    path = out_dir / "fig_efficiency_scatter.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def generate_size_pass_rate(attempts: pd.DataFrame, out_dir: Path) -> Path:
    """Line plot of quantized agent-mode pass rate by model size."""
    agent = attempts[
        (attempts["mode"] == "agent_tools")
        & (attempts["quantization"] == "UD-Q4_K_XL")
    ].copy()

    grouped = (
        agent.groupby(["model_family", "model_size_b", "architecture"])["success"]
        .agg(["sum", "count", "mean"])
        .reset_index()
        .sort_values(["model_family", "model_size_b"])
    )
    intervals = grouped.apply(
        lambda row: _wilson_interval(int(row["sum"]), int(row["count"])),
        axis=1,
        result_type="expand",
    )
    grouped["ci_low"] = intervals[0]
    grouped["ci_high"] = intervals[1]

    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    styles = {
        "qwen3.5": {"label": "Qwen 3.5", "marker": "o", "color": "#1f77b4"},
        "qwen3.6": {"label": "Qwen 3.6", "marker": "s", "color": "#d62728"},
    }
    for family, family_df in grouped.groupby("model_family"):
        style = styles.get(family, {"label": family, "marker": "o", "color": None})
        y = family_df["mean"] * 100
        yerr = np.vstack([
            y - family_df["ci_low"] * 100,
            family_df["ci_high"] * 100 - y,
        ])
        ax.errorbar(
            family_df["model_size_b"],
            y,
            yerr=yerr,
            marker=style["marker"],
            linewidth=1.6,
            markersize=5,
            capsize=2.5,
            elinewidth=0.8,
            capthick=0.8,
            color=style["color"],
            label=style["label"],
        )
        moe = family_df[family_df["architecture"] == "moe"]
        if not moe.empty:
            ax.scatter(
                moe["model_size_b"],
                moe["mean"] * 100,
                marker="D",
                s=32,
                color=style["color"],
                edgecolor="black",
                linewidth=0.4,
                zorder=3,
            )

    ax.set_xlabel("Model size (B parameters)")
    ax.set_ylabel("Pass rate (%)")
    ax.set_ylim(0, 95)
    ax.set_xticks([2, 4, 9, 27, 35])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = out_dir / "fig_size_pass_rate.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def generate_failure_bar(attempts: pd.DataFrame, out_dir: Path) -> Path:
    """Stacked horizontal bar of failure categories by quantized model."""
    agent = attempts[
        (attempts["mode"] == "agent_tools")
        & (attempts["quantization"] == "UD-Q4_K_XL")
    ]
    # Exclude successes
    failures = agent[agent["success"] == False]  # noqa: E712
    if failures.empty:
        return None

    cat_order = (
        failures["failure_category"]
        .value_counts()
        .index.tolist()
    )

    model_order = sorted(agent["model_id"].unique(), key=_model_sort_key)

    pivot = (
        failures.groupby(["model_id", "failure_category"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=model_order, columns=cat_order, fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(6, 3.0))
    left = np.zeros(len(model_order))
    colors = plt.cm.tab10(np.linspace(0, 0.9, len(cat_order)))
    for cat, color in zip(cat_order, colors):
        ax.barh(range(len(model_order)), pivot[cat], left=left, color=color, label=cat.replace("_", " "))
        left += pivot[cat].values

    ax.set_yticks(range(len(model_order)))
    ax.set_yticklabels([_model_label(m) for m in model_order])
    ax.set_xlabel("Failure count")
    ax.set_title("Failure categories in quantized agent mode")
    ax.legend(loc="upper right", fontsize=6)
    fig.tight_layout()
    path = out_dir / "fig_failure_bar.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def main() -> int:
    aggregated_dir = Path("runs/full_sweep/aggregated")
    attempts_path = aggregated_dir / "attempts.csv"
    out_dir = Path("paper_draft/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not attempts_path.exists():
        print(f"ERROR: {attempts_path} not found", file=sys.stderr)
        return 1

    attempts = pd.read_csv(attempts_path)

    print("Generating heatmaps...")
    generate_heatmaps(attempts, out_dir)

    print("Generating efficiency scatter...")
    generate_efficiency_scatter(attempts, out_dir)

    print("Generating size/pass-rate line plot...")
    generate_size_pass_rate(attempts, out_dir)

    print("Generating failure bar chart...")
    generate_failure_bar(attempts, out_dir)

    print(f"Figures written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
