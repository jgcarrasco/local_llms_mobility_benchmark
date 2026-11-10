"""Structural summaries of the task inputs for the schema-aware condition."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd

from mobility_bench.tasks.base import TaskSpec


MAX_SCHEMA_CHARS = 6000
MAX_SAMPLE_ROWS = 3
MAX_ZIP_MEMBERS = 8
MAX_JSON_SAMPLE_CHARS = 400


def _format_csv(path: Path) -> str:
    frame = pd.read_csv(path)
    columns = ", ".join(f"{name}:{dtype}" for name, dtype in zip(frame.columns, frame.dtypes))
    sample = frame.head(MAX_SAMPLE_ROWS).to_csv(index=False).rstrip("\n")
    indented = "\n".join(f"    {line}" for line in sample.splitlines())
    return f"{len(frame)} rows x {len(frame.columns)} columns\n  columns: {columns}\n  first rows:\n{indented}"


def _describe_json_value(value: Any) -> str:
    if isinstance(value, dict):
        return f"object with keys {sorted(value)[:12]}"
    if isinstance(value, list):
        if not value:
            return "empty list"
        first = value[0]
        if isinstance(first, dict):
            return f"list of {len(value)} objects with keys {sorted(first)[:12]}"
        return f"list of {len(value)} values, first: {str(first)[:80]}"
    return f"{type(value).__name__}"


def _format_json(path: Path) -> str:
    payload = json.loads(path.read_text(encoding="utf-8"))
    lines = [_describe_json_value(payload)]
    sample = json.dumps(payload, ensure_ascii=True, indent=2)
    if len(sample) > MAX_JSON_SAMPLE_CHARS:
        sample = sample[:MAX_JSON_SAMPLE_CHARS] + "\n  ..."
    lines.append("  sample:\n" + "\n".join(f"    {line}" for line in sample.splitlines()))
    return "\n".join(lines)


def _format_zip(path: Path) -> str:
    with zipfile.ZipFile(path) as archive:
        members = sorted(archive.namelist())
        lines = [f"archive with {len(members)} members: {members[:MAX_ZIP_MEMBERS]}"]
        for member in members[:MAX_ZIP_MEMBERS]:
            # GTFS members are CSV content with a .txt extension.
            if not member.lower().endswith((".csv", ".txt")):
                continue
            with archive.open(member) as handle:
                frame = pd.read_csv(handle)
            columns = ", ".join(f"{name}:{dtype}" for name, dtype in zip(frame.columns, frame.dtypes))
            lines.append(f"  {member}: {len(frame)} rows x {len(frame.columns)} columns; columns: {columns}")
    return "\n".join(lines)


def _format_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8", errors="replace")
    first_lines = "\n".join(text.splitlines()[:MAX_SAMPLE_ROWS])
    return f"{len(text)} characters\n  first lines:\n" + "\n".join(
        f"    {line}" for line in first_lines.splitlines()
    )


def _format_entry(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _format_csv(path)
    if suffix == ".json":
        return _format_json(path)
    if suffix == ".zip":
        return _format_zip(path)
    return _format_text(path)


def declared_input_paths(workspace_root: Path, task: TaskSpec) -> list[tuple[str, Path]]:
    """Return (label, path) pairs for the files the task declares as inputs.

    Only declared inputs are described. Summarising the whole ``inputs/``
    directory would show files that belong to other tasks, including fixtures
    that contain reference answers.
    """
    if "all previous outputs" in task.input_files:
        root = workspace_root / "outputs"
        if not root.is_dir():
            return []
        excluded = set(task.expected_outputs)
        pairs: list[tuple[str, Path]] = []
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            label = path.relative_to(workspace_root).as_posix()
            if label in excluded:
                continue
            pairs.append((label, path))
        return pairs
    pairs = []
    for relative in task.input_files:
        path = workspace_root / relative
        if path.is_file():
            pairs.append((relative, path))
    return pairs


def build_input_schema(
    workspace_root: Path,
    task: TaskSpec,
    max_chars: int = MAX_SCHEMA_CHARS,
) -> str:
    """Return a deterministic, bounded description of the declared input files.

    The summary describes structure and a few sample rows only. It is generated
    from the seeded inputs so that every attempt of a task receives identical
    text, and it is logged verbatim with the attempt.
    """
    sections: list[str] = []
    used = 0
    truncated = False
    for label, path in declared_input_paths(workspace_root, task):
        try:
            body = _format_entry(path)
        except Exception as exc:  # noqa: BLE001 - a broken fixture must not abort the attempt
            body = f"<unreadable: {type(exc).__name__}: {exc}>"
        section = f"{label}:\n  {body}"
        if used + len(section) > max_chars:
            truncated = True
            break
        sections.append(section)
        used += len(section)
    if truncated:
        sections.append("... remaining input files omitted to keep the summary bounded")
    return "\n\n".join(sections)
