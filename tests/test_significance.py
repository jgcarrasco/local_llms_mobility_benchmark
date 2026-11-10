from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mobility_bench.analysis.significance import (
    cluster_bootstrap_rate,
    holm_adjust,
    load_attempts,
    mode_effect_regression,
    paired_gain,
    pairwise_wilcoxon,
    pass_at_k_table,
    pass_at_k_value,
    run_significance,
)

MODES = ("chat_only", "chat_schema", "agent_tools")


def make_attempts(probabilities: dict[tuple[str, str], float], tasks: int = 15, seed: int = 0) -> pd.DataFrame:
    """Build attempts whose success probability depends only on model and mode."""
    rng = np.random.default_rng(seed)
    rows = []
    for (model_id, mode), probability in probabilities.items():
        for index in range(tasks):
            for repetition in range(1, 6):
                rows.append(
                    {
                        "model_id": model_id,
                        "mode": mode,
                        "task_id": f"T{index:02d}",
                        "repetition": repetition,
                        "success": bool(rng.random() < probability),
                        "output_tokens_total": 1000 + int(rng.random() * 500),
                    }
                )
    return pd.DataFrame(rows)


def make_clustered_attempts() -> pd.DataFrame:
    """Tasks that either always pass or always fail: maximum task-level clustering."""
    rows = []
    for index in range(15):
        outcome = index < 8
        for repetition in range(1, 6):
            rows.append(
                {
                    "model_id": "m1",
                    "mode": "chat_only",
                    "task_id": f"T{index:02d}",
                    "repetition": repetition,
                    "success": outcome,
                    "output_tokens_total": 1000,
                }
            )
    return pd.DataFrame(rows)


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return centre - half, centre + half


def test_pass_at_k_matches_pass_rate_when_k_is_one() -> None:
    attempts = make_attempts({("m1", "chat_only"): 0.5, ("m1", "agent_tools"): 0.8})
    table = pass_at_k_table(attempts, ks=[1])
    for _, row in table.iterrows():
        subset = attempts[
            (attempts["model_id"] == row["model_id"]) & (attempts["mode"] == row["mode"])
        ]
        assert row["value"] == pytest.approx(subset["success"].mean())


def test_pass_at_k_decreases_with_k_and_hits_bounds() -> None:
    always = pass_at_k_value([5, 5], [5, 5], 5)
    never = pass_at_k_value([0, 0], [5, 5], 5)
    assert always == pytest.approx(1.0)
    assert never == pytest.approx(0.0)
    counts = [(5, 5), (3, 5), (0, 5)]
    values = [pass_at_k_value([s for s, _ in counts], [n for _, n in counts], k) for k in range(1, 6)]
    assert values == sorted(values, reverse=True)


def test_cluster_bootstrap_is_wider_than_wilson_for_clustered_data() -> None:
    attempts = make_clustered_attempts()
    interval = cluster_bootstrap_rate(attempts, "m1", "chat_only", iters=4000, seed=1)
    successes = int(attempts["success"].sum())
    low, high = wilson_interval(successes, len(attempts))
    assert (interval.high - interval.low) > (high - low)
    assert interval.low <= interval.point <= interval.high


def test_paired_gain_recovers_the_difference_between_conditions() -> None:
    attempts = make_attempts(
        {("m1", "chat_only"): 0.3, ("m1", "agent_tools"): 0.7}, tasks=15, seed=2
    )
    interval = paired_gain(attempts, "m1", "agent_tools", iters=2000, seed=3)
    baseline = attempts[attempts["mode"] == "chat_only"]["success"].mean()
    treatment = attempts[attempts["mode"] == "agent_tools"]["success"].mean()
    assert interval.point == pytest.approx(treatment - baseline)
    assert interval.low <= interval.point <= interval.high


def test_holm_adjust_is_at_least_as_large_as_raw_and_bounded() -> None:
    raw = [0.001, 0.02, 0.03, 0.2]
    adjusted = holm_adjust(raw)
    assert np.all(adjusted >= np.array(raw) - 1e-12)
    assert np.all(adjusted <= 1.0)
    assert adjusted[0] == pytest.approx(0.004)


def test_mode_effect_regression_finds_a_positive_workspace_effect() -> None:
    attempts = make_attempts(
        {
            ("m1", "chat_only"): 0.2,
            ("m1", "agent_tools"): 0.6,
            ("m2", "chat_only"): 0.4,
            ("m2", "agent_tools"): 0.8,
        },
        tasks=15,
        seed=4,
    )
    results = mode_effect_regression(attempts)
    logistic = results[
        (results["family"] == "binomial")
        & (~results["token_control"])
        & (results["term"] == "mode_agent_tools")
    ]
    assert not logistic.empty
    row = logistic.iloc[0]
    assert row["coef"] > 0
    assert row["ci_low"] > 0  # the workspace effect is separated from zero


def test_pairwise_wilcoxon_separates_a_clear_gap() -> None:
    attempts = make_attempts(
        {("m1", "agent_tools"): 0.05, ("m2", "agent_tools"): 0.95}, tasks=15, seed=5
    )
    results = pairwise_wilcoxon(attempts, "agent_tools", ["m1", "m2"])
    assert len(results) == 1
    assert results.loc[0, "p_holm"] < 0.05
    assert results.loc[0, "median_diff"] < 0


def test_run_significance_writes_every_artefact(tmp_path: Path) -> None:
    attempts = make_attempts(
        {
            ("m1", "chat_only"): 0.3,
            ("m1", "chat_schema"): 0.4,
            ("m1", "agent_tools"): 0.7,
            ("m2", "chat_only"): 0.6,
            ("m2", "chat_schema"): 0.7,
            ("m2", "agent_tools"): 0.9,
        },
        tasks=15,
        seed=6,
    )
    source = tmp_path / "attempts.csv"
    attempts.to_csv(source, index=False)
    loaded = load_attempts(tmp_path)
    assert len(loaded) == len(attempts)

    outputs = run_significance(source, tmp_path / "out", iters=500, seed=7)
    for name, path in outputs.items():
        assert path.exists(), name
    gains = pd.read_csv(outputs["gains"])
    assert set(gains["contrast"]) == {"input_schema", "interaction", "workspace"}
    # The agentic gain decomposes into the input-schema and interaction gains.
    point = gains.groupby(["model_id", "contrast"])["gain"].first().unstack()
    assert np.allclose(
        point["workspace"], point["input_schema"] + point["interaction"], atol=1e-9
    )
    reliability = pd.read_csv(outputs["reliability"])
    assert set(reliability["k"]) == {1, 2, 3, 4, 5}


def test_quantisation_contrasts_pair_only_precision_differences() -> None:
    from mobility_bench.analysis.significance import quantisation_contrasts

    tasks = [f"T{index:02d}" for index in range(1, 4)]
    rows = []
    for model_id, quantization in (("m4_q4", "UD-Q4_K_XL"), ("m4_bf16", "BF16")):
        for task in tasks:
            for repetition in range(2):
                # The BF16 model fails one task that the Q4 model solves.
                success = not (quantization == "BF16" and task == "T02")
                rows.append(
                    {
                        "model_id": model_id,
                        "mode": "agent_tools",
                        "task_id": task,
                        "repetition": repetition,
                        "success": success,
                        "model_family": "qwen3.5",
                        "model_size_b": 4.0,
                        "architecture": "dense",
                        "quantization": quantization,
                    }
                )
    attempts = pd.DataFrame(rows)

    contrasts = quantisation_contrasts(attempts, iters=200, seed=3)

    assert list(contrasts.columns) == ["model_a", "model_b", "mode", "gain", "ci_low", "ci_high"]
    assert len(contrasts) == 1
    row = contrasts.iloc[0]
    assert row["model_a"] == "m4_q4"
    assert row["model_b"] == "m4_bf16"
    assert row["gain"] == pytest.approx(-1 / 3)
    assert row["ci_low"] <= row["gain"] <= row["ci_high"]
