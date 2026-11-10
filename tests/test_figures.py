"""Tests for the paper figure pipeline.

The pipeline once pointed at a directory that no longer existed and silently
produced figures from a pilot run, so these tests pin the data source and the
model selection rather than the rendered pixels.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "generate_paper_figures.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("generate_paper_figures", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


figures = _load_module()


def _attempt(model_id: str, *, quantization: str = "UD-Q4_K_XL", family: str = "qwen3.5"):
    return {
        "model_id": model_id,
        "mode": "agent_tools",
        "task_id": "T01",
        "success": True,
        "quantization": quantization,
        "model_family": family,
        "model_size_b": 4.0,
        "architecture": "dense",
        "time_to_submission_seconds": 10.0,
    }


def test_qwen_core_drops_the_other_families() -> None:
    frame = pd.DataFrame(
        [
            _attempt("qwen35_4b_q4"),
            _attempt("qwen38_27b_q4", quantization="UD-Q4_K_M", family="qwen3.8"),
            _attempt("gemma4_12b_q4", quantization="Q4_K_M", family="gemma4"),
            _attempt("phi4_14b_q4", quantization="Q4_K_M", family="phi4"),
        ]
    )

    kept = figures._qwen_core(frame)["model_id"].tolist()

    assert kept == ["qwen35_4b_q4", "qwen38_27b_q4"]


def test_missing_significance_output_fails_loudly(tmp_path: Path) -> None:
    (tmp_path / "attempts.csv").write_text("model_id\n", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        figures._load_clustered_rates(tmp_path)


def test_compact_label_covers_every_qwen_configuration() -> None:
    labels = [figures._compact_label(model) for model in figures.QWEN_MODELS]

    assert labels == [
        "3.5 2B",
        "3.5 4B",
        "3.5 9B",
        "3.5 27B",
        "3.5 35B MoE",
        "3.6 27B",
        "3.6 35B MoE",
        "3.8 27B",
    ]
    assert figures._compact_label(figures.BF16_REFERENCE) == "4B BF16"
