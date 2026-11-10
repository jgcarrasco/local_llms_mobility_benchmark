"""Restore the reference inputs that are seeded into an attempt workspace."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mobility_bench.tasks.base import TaskSpec


SEEDED_PREFIXES = ("outputs/", "src/", "reports/", "figures/")


def _restore_file(source: Path, target: Path) -> bool:
    """Copy source over target when the two differ; return whether a copy happened."""
    if target.exists() and target.read_bytes() == source.read_bytes():
        return False
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return True


def restore_seeded_inputs(task: TaskSpec, data_dir: Path, workspace_root: Path) -> list[str]:
    """Rewrite the task's seeded inputs from the reference copies.

    An attempt workspace is created with the reference upstream artifacts, but a
    model can overwrite them while working. Restoring them before scoring means
    the submitted output is always judged against the inputs the task specifies.
    The task's own expected outputs are left untouched.
    """
    restored: list[str] = []
    if "all previous outputs" in task.input_files:
        reference_root = data_dir / "reference_outputs"
        expected = set(task.expected_outputs)
        for child in ("outputs", "src"):
            source_root = reference_root / child
            if not source_root.exists():
                continue
            for source in sorted(source_root.rglob("*")):
                if not source.is_file():
                    continue
                relative = source.relative_to(reference_root).as_posix()
                if relative in expected:
                    continue
                if _restore_file(source, workspace_root / relative):
                    restored.append(relative)
        return restored
    for relative in task.input_files:
        source = (
            data_dir / "reference_outputs" / relative
            if relative.startswith(SEEDED_PREFIXES)
            else data_dir / relative
        )
        if source.exists() and _restore_file(source, workspace_root / relative):
            restored.append(relative)
    return restored
