"""Read tool implementation."""

from __future__ import annotations

from mobility_bench.runtime.workspace import Workspace, WorkspacePathError


def read_file(workspace: Workspace, path: str, *, max_chars: int = 6_000) -> dict[str, object]:
    try:
        resolved = workspace.resolve_path(path, must_exist=True)
        content = resolved.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError, WorkspacePathError) as exc:
        return {
            "ok": False,
            "content": "",
            "truncated": False,
            "n_chars": 0,
            "stderr": str(exc),
            "metadata": {},
        }

    total_chars = len(content)
    truncated = total_chars > max_chars
    if truncated:
        content = content[:max_chars]
    return {
        "ok": True,
        "content": content,
        "truncated": truncated,
        "n_chars": len(content),
        "metadata": {
            "path": path,
            "total_chars": total_chars,
            "max_chars": max_chars,
            "hint": "Use bash with commands like head, cut, grep, or python for targeted inspection of large files."
            if truncated
            else "",
        },
    }
