#!/usr/bin/env python3
"""Generate the paper figures from aggregated sweep results.

The figures read the aggregates produced by ``mobility-bench aggregate`` and the
clustered intervals produced by ``mobility-bench significance``, so a figure
cannot show a different number from the table it accompanies.
"""

from __future__ import annotations

import argparse
import sys
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
        "qwen38_27b_q4": "Q3.8 27B Q4",
    }
    return mapping.get(model_id, model_id)


def _compact_label(model_id: str) -> str:
    """Short label for point annotations, where horizontal space is limited."""
    if model_id == BF16_REFERENCE:
        return "4B BF16"
    match = None
    for size in ("2b", "4b", "9b", "27b", "35b"):
        if size in model_id:
            match = size
            break
    if match is None:
        return model_id
    family = model_id.split("_")[0].replace("qwen", "")
    generation = f"{family[0]}.{family[1:]}"
    suffix = " MoE" if "a3b" in model_id else ""
    return f"{generation} {match[:-1]}B{suffix}"


def _model_sort_key(model_id: str) -> tuple:
    """Sort key: family, size, quantization."""
    family = 1
    for rank, tag in enumerate(("qwen35", "qwen36", "qwen38")):
        if tag in model_id:
            family = rank
            break
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


def _is_four_bit(attempts: pd.DataFrame) -> pd.Series:
    """Whether an attempt used 4-bit weights (BF16 is the only 16-bit format)."""
    return attempts["quantization"].astype(str).str.upper() != "BF16"


QWEN_MODELS = (
    "qwen35_2b_q4",
    "qwen35_4b_q4",
    "qwen35_9b_q4",
    "qwen35_27b_q4",
    "qwen35_35b_a3b_q4",
    "qwen36_27b_q4",
    "qwen36_35b_a3b_q4",
    "qwen38_27b_q4",
)
BF16_REFERENCE = "qwen35_4b_bf16"


def _qwen_core(attempts: pd.DataFrame) -> pd.DataFrame:
    """The Qwen 4-bit configurations that Sections 5.1--5.3 analyse."""
    return attempts[attempts["model_id"].isin(QWEN_MODELS)]


def _load_clustered_rates(aggregated_dir: Path) -> pd.DataFrame:
    """Task-clustered pass-rate intervals, keyed by model and condition."""
    path = aggregated_dir / "significance" / "rates_with_ci.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run `mobility-bench significance` before generating figures"
        )
    return pd.read_csv(path).set_index(["model_id", "mode"])


def generate_heatmaps(attempts: pd.DataFrame, out_dir: Path) -> Path:
    """Heatmap of pass rate by selected model/task condition, one per mode."""
    task_order = [f"T{i:02d}" for i in range(1, 16)]
    attempts = _qwen_core(attempts)
    model_order = sorted(attempts["model_id"].unique(), key=_model_sort_key)

    fig, axes = plt.subplots(2, 1, figsize=(7.2, 5.8), sharex=True, constrained_layout=True)

    for ax, mode in zip(axes, ["chat_only", "agent_tools"]):
        df = attempts[attempts["mode"] == mode]
        counts = (
            df.groupby(["model_id", "task_id"])["success"]
            .agg(successes="sum", attempts="count")
            .reset_index()
        )

        def _cell(column: str) -> np.ndarray:
            return (
                counts.pivot(index="model_id", columns="task_id", values=column)
                .reindex(index=model_order, columns=task_order)
                .fillna(0)
                .to_numpy(dtype=float)
            )

        successes = _cell("successes")
        attempts_per_cell = _cell("attempts")
        with np.errstate(invalid="ignore", divide="ignore"):
            grouped = np.where(
                attempts_per_cell > 0, successes / attempts_per_cell, np.nan
            )

        im = ax.imshow(grouped, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
        ax.set_xticks(range(len(task_order)))
        ax.set_xticklabels(task_order, rotation=45, ha="right")
        ax.set_yticks(range(len(model_order)))
        ax.set_yticklabels([_model_label(m) for m in model_order])
        for label in ax.get_yticklabels():
            label.set_ha("left")
            label.set_x(-0.14)
        ax.set_title("Chat-only" if mode == "chat_only" else "Agent mode")
        ax.set_xlabel("Task")

        # Annotate each cell with successes out of repetitions so that the
        # reliability of a cell is visible rather than implied by a shade.
        for i in range(len(model_order)):
            for j in range(len(task_order)):
                total = attempts_per_cell[i, j]
                if total == 0:
                    continue
                successes_in_cell = int(successes[i, j])
                text = f"{successes_in_cell}/{int(total)}"
                rate = grouped[i, j]
                color = "white" if rate < 0.35 or rate > 0.85 else "black"
                ax.text(j, i, text, ha="center", va="center", color=color, fontsize=6.5)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), orientation="vertical", fraction=0.025, pad=0.01)
    cbar.set_label("Pass rate")
    path = out_dir / "fig_heatmaps.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def generate_efficiency_scatter(attempts: pd.DataFrame, out_dir: Path) -> Path:
    """Scatter plot: pass rate vs runtime for agent-mode runs."""
    attempts = _qwen_core(attempts)
    attempts = attempts[
        (attempts["mode"] == "agent_tools")
        | (attempts["model_id"] == BF16_REFERENCE) & (attempts["mode"] == "agent_tools")
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
    marker_sizes = 38 + 110 * (df["size_b"] - min_size) / (max_size - min_size)
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
        # Labels alternate above and below the marker when two configurations
        # sit close in latency, so that no label can overwrite its neighbour.
        side = 1
        previous_x = None
        for _, row in s.sort_values("med_time").iterrows():
            x, y = float(row["med_time"]), float(row["pass_rate"] * 100)
            if previous_x is not None and x - previous_x < 8.0:
                side = -side
            else:
                side = 1
            previous_x = x
            to_left = x > df["med_time"].quantile(0.75)
            ax.annotate(
                _compact_label(row["model_id"]),
                (x, y),
                textcoords="offset points",
                xytext=(-8 if to_left else 9, 10 * side),
                ha="right" if to_left else "left",
                va="bottom" if side > 0 else "top",
                fontsize=6,
                alpha=0.8,
                clip_on=False,
            )

    cbar = fig.colorbar(scatter_for_colorbar, ax=ax, fraction=0.045, pad=0.025)
    cbar.set_label("Model size (B parameters)")
    ax.set_xlabel("Median time to submission (s)")
    ax.set_ylabel("Pass rate (%)")
    ax.set_title("Agent mode accuracy-latency trade-off")
    ax.legend(loc="lower right", ncol=2, frameon=True, framealpha=0.92, borderpad=0.35)
    ax.set_xlim(df["med_time"].min() - 4.0, df["med_time"].max() + 10.0)
    ax.set_ylim(max(0, df["pass_rate"].min() * 100 - 10.0), 112)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))

    fig.tight_layout()
    path = out_dir / "fig_efficiency_scatter.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def generate_size_pass_rate(
    attempts: pd.DataFrame, out_dir: Path, rates: pd.DataFrame
) -> Path:
    """Line plot of 4-bit agent-mode pass rate by model size."""
    agent = _qwen_core(attempts)
    agent = agent[agent["mode"] == "agent_tools"].copy()

    grouped = (
        agent.groupby(["model_family", "model_size_b", "architecture"])
        .agg(
            successes=("success", "sum"),
            total=("success", "count"),
            mean=("success", "mean"),
            model_id=("model_id", "first"),
        )
        .reset_index()
        .sort_values(["model_family", "model_size_b"])
    )
    # Intervals come from the clustered bootstrap so that a figure can never
    # report different uncertainty from the text for the same cell.
    grouped["ci_low"] = grouped["model_id"].map(
        lambda m: rates.loc[(m, "agent_tools"), "ci_low"]
    )
    grouped["ci_high"] = grouped["model_id"].map(
        lambda m: rates.loc[(m, "agent_tools"), "ci_high"]
    )
    grouped = grouped.dropna(subset=["ci_low", "ci_high"])

    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    styles = {
        "qwen3.5": {"label": "Qwen 3.5", "marker": "o", "color": "#1f77b4"},
        "qwen3.6": {"label": "Qwen 3.6", "marker": "s", "color": "#d62728"},
        "qwen3.8": {"label": "Qwen 3.8", "marker": "^", "color": "#2ca02c"},
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
    ax.set_ylim(0, 105)
    ax.set_xticks([2, 4, 9, 27, 35])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = out_dir / "fig_size_pass_rate.pdf"
    fig.savefig(path)
    plt.close(fig)
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--aggregated-dir", default="runs/sweep_v2/aggregated")
    parser.add_argument("--out-dir", default="manuscripts/first_revision/paper_draft/figures")
    args = parser.parse_args(argv)

    aggregated_dir = Path(args.aggregated_dir)
    attempts_path = aggregated_dir / "attempts.csv"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not attempts_path.exists():
        print(f"ERROR: {attempts_path} not found", file=sys.stderr)
        return 1

    attempts = pd.read_csv(attempts_path)
    rates = _load_clustered_rates(aggregated_dir)

    print("Generating heatmaps...")
    generate_heatmaps(attempts, out_dir)

    print("Generating efficiency scatter...")
    generate_efficiency_scatter(attempts, out_dir)

    print("Generating size/pass-rate line plot...")
    generate_size_pass_rate(attempts, out_dir, rates)

    print(f"Figures written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
