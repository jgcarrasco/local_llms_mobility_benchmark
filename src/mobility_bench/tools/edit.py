"""Edit tool implementation."""

from __future__ import annotations

from mobility_bench.runtime.workspace import Workspace, WorkspacePathError
from mobility_bench.tools.base import ToolResult


def edit_file(
    workspace: Workspace,
    path: str,
    old: str,
    new: str,
    *,
    replace_all: bool = True,
) -> ToolResult:
    if old == "":
        return ToolResult(ok=False, stderr="'old' must not be empty", metadata={"path": path})
    try:
        resolved = workspace.resolve_path(path, must_exist=True)
        content = resolved.read_text(encoding="utf-8")
        if old not in content:
            return ToolResult(ok=False, stderr="'old' text not found", metadata={"path": path})
        count = content.count(old) if replace_all else 1
        updated = content.replace(old, new, -1 if replace_all else 1)
        resolved.write_text(updated, encoding="utf-8")
    except (OSError, UnicodeDecodeError, WorkspacePathError) as exc:
        return ToolResult(ok=False, stderr=str(exc), metadata={"path": path})
    return ToolResult(ok=True, metadata={"path": path, "replacements": count})
