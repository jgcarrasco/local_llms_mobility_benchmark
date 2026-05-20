"""Submit tool implementation."""

from __future__ import annotations

from mobility_bench.tools.base import ToolResult


def submit(message: str = "") -> ToolResult:
    return ToolResult(ok=True, stdout=message, metadata={"submitted": True})
