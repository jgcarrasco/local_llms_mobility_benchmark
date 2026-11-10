"""Clustered significance analysis.

The unit of independence is the task, not the attempt. Five repetitions of the
same task share its difficulty, prompt, and fixtures, so treating them as
independent Bernoulli trials understates the variance and manufactures
significance. Everything here resamples or clusters by task:

- pass^k with the combinatorial estimator (pass^1 equals the pooled pass rate);
- cluster bootstrap intervals for rates and for paired differences;
- logistic regression with task and model fixed effects and cluster-robust
  standard errors, with and without a token-volume control;
- Wilcoxon signed-rank comparisons on task-level rates, Holm-corrected.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy import stats

DEFAULT_ITERS = 10_000
DEFAULT_SEED = 2026_11_04
MODES = ("chat_only", "chat_schema", "agent_tools")
BASELINE_MODE = "chat_only"


@dataclass(frozen=True)
class Interval:
    """A point estimate with a resampled interval."""

    point: float
    low: float
    high: float


def load_attempts(path: Path | str) -> pd.DataFrame:
    """Read attempt rows from an aggregated CSV or a results directory."""
    path = Path(path)
    if path.is_dir():
        for candidate in ("attempts.csv", "attempts.jsonl"):
            if (path / candidate).exists():
                path = path / candidate
                break
        else:
            raise FileNotFoundError(f"no attempts.csv or attempts.jsonl in {path}")
    if path.suffix == ".jsonl":
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        return pd.DataFrame(rows)
    return pd.read_csv(path)


def _task_counts(frame: pd.DataFrame) -> pd.DataFrame:
    """Successes and attempts per task, indexed by task id."""
    return frame.groupby("task_id")["success"].agg(["sum", "count"])


def _draws(rng: np.random.Generator, n_tasks: int, iters: int) -> np.ndarray:
    """Bootstrap indices that resample tasks, keeping each task's attempts."""
    return rng.integers(0, n_tasks, size=(iters, n_tasks))


def _interval(point: float, samples: np.ndarray) -> Interval:
    low, high = np.percentile(samples, [2.5, 97.5])
    return Interval(point=float(point), low=float(low), high=float(high))


def pass_at_k_value(successes: Sequence[int], attempts: Sequence[int], k: int) -> float:
    """Combinatorial pass^k: mean over tasks of C(s, k) / C(n, k).

    Averaged over tasks, this is the probability that k independent attempts on
    the same task all succeed. At k = 1 it reduces to the pooled pass rate, and
    at k = n it is the fraction of tasks solved in every repetition.
    """
    values = []
    for s, n in zip(successes, attempts):
        s, n = int(s), int(n)
        if n >= k:
            values.append(math.comb(s, k) / math.comb(n, k))
    if not values:
        return float("nan")
    return float(np.mean(values))


def pass_at_k_table(
    attempts: pd.DataFrame, ks: Iterable[int] = range(1, 6)
) -> pd.DataFrame:
    """pass^k for every model and condition."""
    rows = []
    for (model_id, mode), frame in attempts.groupby(["model_id", "mode"], sort=True):
        counts = _task_counts(frame)
        for k in ks:
            rows.append(
                {
                    "model_id": model_id,
                    "mode": mode,
                    "k": k,
                    "value": pass_at_k_value(counts["sum"], counts["count"], k),
                }
            )
    return pd.DataFrame(rows)


def cluster_bootstrap_rate(
    attempts: pd.DataFrame,
    model_id: str,
    mode: str,
    iters: int = DEFAULT_ITERS,
    seed: int = DEFAULT_SEED,
) -> Interval:
    """Pass rate with an interval from resampling tasks."""
    frame = attempts[(attempts["model_id"] == model_id) & (attempts["mode"] == mode)]
    if frame.empty:
        raise ValueError(f"no attempts for {model_id} in {mode}")
    counts = _task_counts(frame)
    successes = counts["sum"].to_numpy(dtype=int)
    totals = counts["count"].to_numpy(dtype=int)
    rng = np.random.default_rng(seed)
    index = _draws(rng, len(counts), iters)
    draw_rates = successes[index].sum(axis=1) / totals[index].sum(axis=1)
    point = successes.sum() / totals.sum()
    return _interval(point, draw_rates)


def _paired_difference(
    reference: pd.DataFrame, comparison: pd.DataFrame, iters: int, seed: int
) -> Interval:
    """Bootstrap ``comparison - reference`` by resampling tasks."""
    tasks = reference.index.union(comparison.index)
    base = reference.reindex(tasks)
    treat = comparison.reindex(tasks)
    base_rate = (base["sum"] / base["count"]).to_numpy(dtype=float)
    treat_rate = (treat["sum"] / treat["count"]).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    index = _draws(rng, len(tasks), iters)
    draws = treat_rate[index].mean(axis=1) - base_rate[index].mean(axis=1)
    point = float(treat_rate.mean() - base_rate.mean())
    return _interval(point, draws)


def paired_gain(
    attempts: pd.DataFrame,
    model_id: str,
    treatment: str,
    baseline: str = BASELINE_MODE,
    iters: int = DEFAULT_ITERS,
    seed: int = DEFAULT_SEED,
) -> Interval:
    """Difference in pass rate between two conditions, resampled paired by task."""
    frames = {
        mode: _task_counts(
            attempts[(attempts["model_id"] == model_id) & (attempts["mode"] == mode)]
        )
        for mode in (baseline, treatment)
    }
    if frames[baseline].empty or frames[treatment].empty:
        raise ValueError(f"missing {baseline} or {treatment} attempts for {model_id}")
    return _paired_difference(frames[baseline], frames[treatment], iters, seed)


def paired_model_gain(
    attempts: pd.DataFrame,
    reference: str,
    comparison: str,
    mode: str = "agent_tools",
    iters: int = DEFAULT_ITERS,
    seed: int = DEFAULT_SEED,
) -> Interval:
    """Difference in pass rate between two models in one condition, paired by task."""
    frames = {
        model: _task_counts(
            attempts[(attempts["model_id"] == model) & (attempts["mode"] == mode)]
        )
        for model in (reference, comparison)
    }
    if frames[reference].empty or frames[comparison].empty:
        raise ValueError(f"missing {mode} attempts for {reference} or {comparison}")
    return _paired_difference(frames[reference], frames[comparison], iters, seed)


def _quantisation_bits(label: object) -> int:
    """Weight precision in bits, for ordering a quantisation contrast."""
    text = str(label).upper()
    return 16 if text.startswith("BF") else 4


def quantisation_contrasts(
    attempts: pd.DataFrame,
    mode: str = "agent_tools",
    iters: int = DEFAULT_ITERS,
    seed: int = DEFAULT_SEED,
) -> pd.DataFrame:
    """Paired precision contrasts between models that differ only in quantisation.

    Pairs are models of the same family, size, and architecture that were
    evaluated at two weight precisions. The lower-precision model is the
    reference, so each gain reads as ``higher precision - lower precision``.
    """
    columns = ["model_a", "model_b", "mode", "gain", "ci_low", "ci_high"]
    required = {"model_id", "mode", "model_family", "model_size_b", "architecture", "quantization"}
    if not required.issubset(attempts.columns):
        return pd.DataFrame(columns=columns)
    frame = attempts[attempts["mode"] == mode]
    if frame.empty:
        return pd.DataFrame(columns=columns)
    meta = frame.drop_duplicates("model_id").set_index("model_id")
    groups: dict[tuple, list[str]] = {}
    for model_id, row in meta.iterrows():
        key = (row.get("model_family"), row.get("model_size_b"), row.get("architecture"))
        groups.setdefault(key, []).append(model_id)

    rows = []
    for key in sorted(groups, key=str):
        members = sorted(groups[key], key=lambda m: (_quantisation_bits(meta.loc[m, "quantization"]), m))
        if len(members) != 2:
            continue
        reference, comparison = members
        interval = paired_model_gain(
            attempts, reference, comparison, mode=mode, iters=iters, seed=seed
        )
        rows.append(
            {
                "model_a": reference,
                "model_b": comparison,
                "mode": mode,
                "gain": interval.point,
                "ci_low": interval.low,
                "ci_high": interval.high,
            }
        )
    return pd.DataFrame(rows, columns=columns)



def _design_matrix(attempts: pd.DataFrame, control_tokens: bool) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Mode, model, and task fixed effects, optionally with a token-volume control."""
    frame = attempts.copy()
    frame["success"] = frame["success"].astype(float)
    mode_dummies = pd.get_dummies(frame["mode"], prefix="mode")
    if f"mode_{BASELINE_MODE}" not in mode_dummies:
        raise ValueError(f"attempts do not contain the baseline mode {BASELINE_MODE}")
    column_order = [
        f"mode_{mode}"
        for mode in MODES
        if mode != BASELINE_MODE and f"mode_{mode}" in mode_dummies
    ]
    model_dummies = pd.get_dummies(frame["model_id"], prefix="model", drop_first=True)
    task_dummies = pd.get_dummies(frame["task_id"], prefix="task", drop_first=True)
    exog = pd.concat([mode_dummies[column_order], model_dummies, task_dummies], axis=1)
    if control_tokens:
        tokens = np.log1p(frame["output_tokens_total"].fillna(0).astype(float))
        exog["log_tokens"] = (tokens - tokens.mean()) / tokens.std(ddof=0)
    exog = exog.astype(float)
    exog.insert(0, "const", 1.0)
    return exog, frame["success"], frame["task_id"]


def _fit_glm(attempts: pd.DataFrame, control_tokens: bool, family: str) -> pd.DataFrame:
    import statsmodels.api as sm  # imported lazily: optional `analysis` extra

    exog, endog, groups = _design_matrix(attempts, control_tokens)
    if family == "binomial":
        model = sm.GLM(endog, exog, family=sm.families.Binomial())
    else:
        model = sm.OLS(endog, exog)
    result = model.fit(cov_type="cluster", cov_kwds={"groups": groups})
    conf = result.conf_int()
    frame = pd.DataFrame(
        {
            "term": exog.columns,
            "coef": result.params.to_numpy(),
            "se": result.bse.to_numpy(),
            "p_value": result.pvalues.to_numpy(),
            "ci_low": conf[0].to_numpy(),
            "ci_high": conf[1].to_numpy(),
        }
    )
    frame["family"] = family
    frame["token_control"] = control_tokens
    frame["n_obs"] = len(endog)
    return frame


def mode_effect_regression(attempts: pd.DataFrame) -> pd.DataFrame:
    """Mode effects with and without the token control, logistic and linear.

    The logistic fits answer the reviewer's request for a model that accounts
    for tasks; the linear probability fits are a robustness check that cannot
    suffer from separation.
    """
    frames = []
    for family in ("binomial", "linear"):
        for control_tokens in (False, True):
            frames.append(_fit_glm(attempts, control_tokens, family))
    return pd.concat(frames, ignore_index=True)


def holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Holm step-down adjustment."""
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        candidate = (total - rank) * values[index]
        running = max(running, candidate)
        adjusted[index] = min(1.0, running)
    return adjusted


def _task_rates(attempts: pd.DataFrame, model_id: str, mode: str) -> pd.Series:
    counts = _task_counts(
        attempts[(attempts["model_id"] == model_id) & (attempts["mode"] == mode)]
    )
    return counts["sum"] / counts["count"]


def pairwise_wilcoxon(
    attempts: pd.DataFrame,
    mode: str,
    models: Sequence[str] | None = None,
    holm: bool = True,
) -> pd.DataFrame:
    """Paired comparisons on task-level rates for one condition."""
    models = sorted(models or attempts["model_id"].unique())
    rates = {model: _task_rates(attempts, model, mode) for model in models}
    rows = []
    for i, model_a in enumerate(models):
        for model_b in models[i + 1 :]:
            paired = pd.concat([rates[model_a], rates[model_b]], axis=1).dropna()
            differences = paired.iloc[:, 0] - paired.iloc[:, 1]
            if (differences == 0).all() or len(paired) < 2:
                statistic, p_value = float("nan"), 1.0
            else:
                statistic, p_value = stats.wilcoxon(
                    paired.iloc[:, 0], paired.iloc[:, 1], zero_method="wilcox"
                )
            rows.append(
                {
                    "mode": mode,
                    "model_a": model_a,
                    "model_b": model_b,
                    "n_tasks": len(paired),
                    "median_diff": float(differences.median()),
                    "statistic": float(statistic),
                    "p_raw": float(p_value),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["p_holm"] = (
        holm_adjust(frame["p_raw"]) if holm else frame["p_raw"]
    )
    return frame


def mode_pairwise_wilcoxon(attempts: pd.DataFrame) -> pd.DataFrame:
    """Per-model test of the workspace condition against the one-shot baseline."""
    rows = []
    for model_id in sorted(attempts["model_id"].unique()):
        baseline = _task_rates(attempts, model_id, BASELINE_MODE)
        for treatment in MODES:
            if treatment == BASELINE_MODE:
                continue
            paired = pd.concat([baseline, _task_rates(attempts, model_id, treatment)], axis=1).dropna()
            differences = paired.iloc[:, 1] - paired.iloc[:, 0]
            if (differences == 0).all() or len(paired) < 2:
                statistic, p_value = float("nan"), 1.0
            else:
                statistic, p_value = stats.wilcoxon(
                    paired.iloc[:, 1], paired.iloc[:, 0], zero_method="wilcox"
                )
            rows.append(
                {
                    "model_id": model_id,
                    "contrast": f"{treatment} - {BASELINE_MODE}",
                    "n_tasks": len(paired),
                    "median_diff": float(differences.median()),
                    "statistic": float(statistic),
                    "p_raw": float(p_value),
                }
            )
    frame = pd.DataFrame(rows)
    frame["p_holm"] = holm_adjust(frame["p_raw"])
    return frame


def top_models(attempts: pd.DataFrame, mode: str, count: int = 4) -> list[str]:
    """The `count` highest pass-rate models in a condition, for the ranking tests."""
    rates = (
        attempts[attempts["mode"] == mode]
        .groupby("model_id")["success"]
        .mean()
        .sort_values(ascending=False)
    )
    return list(rates.index[:count])


def run_significance(
    attempts_path: Path | str,
    output_dir: Path | str,
    iters: int = DEFAULT_ITERS,
    seed: int = DEFAULT_SEED,
    top: int = 4,
) -> dict[str, Path]:
    """Write every significance artefact and return the paths."""
    attempts = load_attempts(attempts_path)
    outputs = Path(output_dir)
    outputs.mkdir(parents=True, exist_ok=True)

    reliability = pass_at_k_table(attempts)

    rate_rows = []
    for (model_id, mode) in sorted({(row["model_id"], row["mode"]) for row in attempts.to_dict("records")}):
        interval = cluster_bootstrap_rate(attempts, model_id, mode, iters=iters, seed=seed)
        rate_rows.append(
            {
                "model_id": model_id,
                "mode": mode,
                "pass_rate": interval.point,
                "ci_low": interval.low,
                "ci_high": interval.high,
            }
        )
    rates = pd.DataFrame(rate_rows)

    gain_rows = []
    for model_id in sorted(attempts["model_id"].unique()):
        contrasts = (
            ("chat_schema", BASELINE_MODE, "input_schema"),
            ("agent_tools", "chat_schema", "interaction"),
            ("agent_tools", BASELINE_MODE, "workspace"),
        )
        for treatment, baseline, label in contrasts:
            interval = paired_gain(
                attempts, model_id, treatment, baseline=baseline, iters=iters, seed=seed
            )
            gain_rows.append(
                {
                    "model_id": model_id,
                    "contrast": label,
                    "gain": interval.point,
                    "ci_low": interval.low,
                    "ci_high": interval.high,
                }
            )
    gains = pd.DataFrame(gain_rows)

    regression = mode_effect_regression(attempts)

    ranking = pairwise_wilcoxon(
        attempts, "agent_tools", top_models(attempts, "agent_tools", top)
    )
    ranking["family"] = "top_models_agent_tools"

    mode_tests = mode_pairwise_wilcoxon(attempts)
    mode_tests["family"] = "mode_contrast_per_model"

    pairwise = pd.concat([ranking, mode_tests], ignore_index=True)

    precision = quantisation_contrasts(attempts, iters=iters, seed=seed)

    paths = {
        "reliability": outputs / "reliability.csv",
        "rates": outputs / "rates_with_ci.csv",
        "gains": outputs / "gains_with_ci.csv",
        "regression": outputs / "regression.csv",
        "pairwise": outputs / "pairwise.csv",
        "precision": outputs / "precision.csv",
        "metadata": outputs / "significance.json",
    }
    reliability.to_csv(paths["reliability"], index=False)
    rates.to_csv(paths["rates"], index=False)
    gains.to_csv(paths["gains"], index=False)
    regression.to_csv(paths["regression"], index=False)
    pairwise.to_csv(paths["pairwise"], index=False)
    precision.to_csv(paths["precision"], index=False)
    paths["metadata"].write_text(
        json.dumps(
            {
                "method": "cluster bootstrap over tasks; pass^k combinatorial",
                "bootstrap_iters": iters,
                "seed": seed,
                "randomisation_unit": "task",
                "correction": "holm",
                "n_attempts": int(len(attempts)),
                "n_tasks": int(attempts["task_id"].nunique()),
                "n_models": int(attempts["model_id"].nunique()),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return paths
