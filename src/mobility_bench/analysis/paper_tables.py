"""Paper-ready LaTeX table generation."""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from mobility_bench.config import load_models_config


TASK_ROWS = [
    ("A", "T01--T03", "Dataset discovery and connector setup"),
    ("B", "T04--T06", "GTFS processing"),
    ("C", "T07--T09", "Semantic enrichment with Wikidata-style data"),
    ("D", "T10--T13", "Feature engineering and analytics"),
    ("E", "T14--T15", "Validation, visualization, and reporting"),
]


def _display_family(value: str) -> str:
    if value.startswith("qwen"):
        return value.replace("qwen", "Qwen ", 1)
    return value


def _display_architecture(value: str) -> str:
    if value == "moe":
        return "MoE"
    return value.title()


def _display_size(value: float) -> str:
    return f"{value:g}B"


def _escape_latex(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _write_latex_table(df: pd.DataFrame, path: Path, caption: str, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    column_format = "l" * max(1, len(df.columns))
    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{{_escape_latex(caption)}}}",
        rf"\label{{{label}}}",
        rf"\begin{{tabular}}{{{column_format}}}",
        r"\toprule",
        " & ".join(_escape_latex(column) for column in df.columns) + r" \\",
        r"\midrule",
    ]
    placeholder_table = list(df.columns) == ["placeholder"]
    for _, row in df.iterrows():
        if placeholder_table:
            lines.append(r"% RESULTS_PLACEHOLDER: Populate after running mobility-bench aggregate.")
        lines.append(" & ".join(_escape_latex(value) for value in row.tolist()) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def make_paper_tables(
    *,
    aggregated_dir: str | Path,
    reports_dir: str | Path,
    models_config: str | Path = "configs/models.yaml",
    paper_draft_dir: str | Path | None = None,
) -> dict[str, Path]:
    aggregated = Path(aggregated_dir)
    table_dir = Path(reports_dir) / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    models = load_models_config(models_config)
    model_rows = []
    for model_id, config in models.models.items():
        if config.backend == "mock":
            continue
        model_rows.append(
            {
                "Model ID": model_id,
                "Family": _display_family(config.family),
                "Size": _display_size(config.size_b),
                "Architecture": _display_architecture(config.architecture),
                "Quantization": config.quantization,
            }
        )
    outputs = {
        "models": table_dir / "table_models.tex",
        "tasks": table_dir / "table_tasks.tex",
        "main_results": table_dir / "table_main_results.tex",
        "agentic_gain": table_dir / "table_agentic_gain.tex",
        "failure_taxonomy": table_dir / "table_failure_taxonomy.tex",
    }
    _write_latex_table(pd.DataFrame(model_rows), outputs["models"], "Configured model metadata.", "tab:models")
    _write_latex_table(
        pd.DataFrame(TASK_ROWS, columns=["Family", "Task IDs", "Scope"]),
        outputs["tasks"],
        "Benchmark task families.",
        "tab:tasks",
    )

    table_inputs = {
        "main_results": aggregated / "pass_rate_by_model_mode.csv",
        "agentic_gain": aggregated / "agentic_gain.csv",
        "failure_taxonomy": aggregated / "failure_categories.csv",
    }
    captions = {
        "main_results": "Main benchmark results.",
        "agentic_gain": "Gain of the schema and tool conditions over the no-information baseline.",
        "failure_taxonomy": "Failure taxonomy counts.",
    }
    labels = {
        "main_results": "tab:main-results",
        "agentic_gain": "tab:agentic-gain",
        "failure_taxonomy": "tab:failure-taxonomy",
    }
    gain_columns = ["model_id", "task_id", "schema_gain", "agentic_gain", "agent_over_schema"]
    for key, csv_path in table_inputs.items():
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            if df.empty:
                df = pd.DataFrame({"placeholder": ["Pending benchmark results"]})
        else:
            df = pd.DataFrame({"placeholder": ["Pending benchmark results"]})
        if key == "agentic_gain" and set(gain_columns) <= set(df.columns):
            df = df[gain_columns]
        _write_latex_table(df, outputs[key], captions[key], labels[key])

    if paper_draft_dir is not None:
        paper_tables = Path(paper_draft_dir) / "tables"
        paper_tables.mkdir(parents=True, exist_ok=True)
        for path in outputs.values():
            shutil.copy2(path, paper_tables / path.name)
    return outputs
