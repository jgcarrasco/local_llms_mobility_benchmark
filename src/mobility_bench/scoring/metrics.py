"""Metric computations."""

from __future__ import annotations

import pandas as pd


def pass_rate_by(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[*group_cols, "attempts", "successful_attempts", "pass_rate"])
    grouped = df.groupby(group_cols, dropna=False)["success"]
    result = grouped.agg(attempts="count", successful_attempts="sum").reset_index()
    result["pass_rate"] = result["successful_attempts"] / result["attempts"]
    return result


def agentic_gain(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["model_id", "task_id", "agent_tools", "chat_only", "agentic_gain"])
    rates = pass_rate_by(df, ["model_id", "task_id", "mode"])
    pivot = rates.pivot_table(
        index=["model_id", "task_id"],
        columns="mode",
        values="pass_rate",
        fill_value=0.0,
    ).reset_index()
    if "agent_tools" not in pivot.columns:
        pivot["agent_tools"] = 0.0
    if "chat_only" not in pivot.columns:
        pivot["chat_only"] = 0.0
    pivot["agentic_gain"] = pivot["agent_tools"] - pivot["chat_only"]
    return pivot[["model_id", "task_id", "agent_tools", "chat_only", "agentic_gain"]]


def efficiency_metrics(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "model_id",
        "mode",
        "pass_rate",
        "median_time_to_submission_seconds",
        "median_total_tokens",
        "time_efficiency",
        "token_efficiency",
    ]
    if df.empty:
        return pd.DataFrame(columns=columns)
    rates = pass_rate_by(df, ["model_id", "mode"])
    medians = (
        df.groupby(["model_id", "mode"], dropna=False)
        .agg(
            median_time_to_submission_seconds=("time_to_submission_seconds", "median"),
            median_total_tokens=("total_tokens", "median"),
        )
        .reset_index()
    )
    result = rates.merge(medians, on=["model_id", "mode"], how="left")
    result["time_efficiency"] = result["pass_rate"] / result["median_time_to_submission_seconds"]
    result["token_efficiency"] = result["pass_rate"] / result["median_total_tokens"].replace(0, pd.NA)
    return result[columns]
