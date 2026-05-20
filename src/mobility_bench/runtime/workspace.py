"""Attempt workspace management."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path


class WorkspacePathError(ValueError):
    """Raised when an agent attempts to access a path outside the workspace."""


@dataclass(frozen=True)
class Workspace:
    root: Path

    def __post_init__(self) -> None:
        object.__setattr__(self, "root", self.root.resolve())

    def ensure_base_dirs(self) -> None:
        for relative in ("inputs", "src", "outputs", "reports", "figures"):
            (self.root / relative).mkdir(parents=True, exist_ok=True)

    def resolve_path(self, relative_path: str | Path, *, must_exist: bool = False) -> Path:
        path = Path(relative_path)
        if path.is_absolute():
            raise WorkspacePathError(f"absolute paths are not allowed: {relative_path}")
        if not path.parts:
            raise WorkspacePathError("empty paths are not allowed")
        resolved = (self.root / path).resolve()
        if self.root != resolved and self.root not in resolved.parents:
            raise WorkspacePathError(f"path escapes workspace: {relative_path}")
        if must_exist and not resolved.exists():
            raise WorkspacePathError(f"path does not exist: {relative_path}")
        return resolved


def create_attempt_workspace(
    run_id: str,
    runs_dir: str | Path = "runs",
    *,
    clean: bool = True,
) -> Workspace:
    root = Path(runs_dir) / "workspaces" / run_id
    if clean and root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    workspace = Workspace(root)
    workspace.ensure_base_dirs()
    return workspace
