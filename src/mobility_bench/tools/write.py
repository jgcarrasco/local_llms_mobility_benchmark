"""Write tool implementation."""

from __future__ import annotations

from mobility_bench.runtime.workspace import Workspace, WorkspacePathError
from mobility_bench.tools.base import ToolResult


def write_file(workspace: Workspace, path: str, content: str) -> ToolResult:
    try:
        resolved = workspace.resolve_path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
    except (OSError, WorkspacePathError) as exc:
        return ToolResult(ok=False, stderr=str(exc), metadata={"path": path})
    return ToolResult(ok=True, metadata={"path": path, "n_chars": len(content)})
