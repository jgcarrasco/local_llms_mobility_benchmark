"""Workspace tools exposed to agent mode."""

from __future__ import annotations

from typing import Any

from mobility_bench.runtime.workspace import Workspace
from mobility_bench.tools.bash import run_bash
from mobility_bench.tools.base import ToolResult
from mobility_bench.tools.edit import edit_file
from mobility_bench.tools.read import read_file
from mobility_bench.tools.submit import submit
from mobility_bench.tools.write import write_file

ALLOWED_TOOLS = {"read", "write", "edit", "bash", "submit"}
TOOL_ALIASES = {
    "read_file": "read",
    "write_file": "write",
    "run": "bash",
    "execute": "bash",
    "shell": "bash",
}


def _missing_argument(tool: str, argument: str) -> dict[str, Any]:
    return ToolResult(ok=False, stderr=f"{tool} requires args.{argument}").to_dict()


def normalize_tool_call(tool: str, args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    normalised_tool = TOOL_ALIASES.get(tool.strip().lstrip(":").strip(), tool.strip().lstrip(":").strip())
    normalised_args = dict(args)
    if normalised_tool in {"read", "write", "edit"} and "path" not in normalised_args and "file" in normalised_args:
        normalised_args["path"] = normalised_args["file"]
    if normalised_tool == "bash" and "command" not in normalised_args and "cmd" in normalised_args:
        normalised_args["command"] = normalised_args["cmd"]
    return normalised_tool, normalised_args


def execute_tool(
    workspace: Workspace,
    tool: str,
    args: dict[str, Any],
    *,
    bash_timeout_seconds: int = 60,
    execution_backend: str = "local",
    docker_image: str = "mobility-bench-runner:latest",
    docker_network: str = "none",
) -> dict[str, Any]:
    tool, args = normalize_tool_call(tool, args)
    if tool == "read":
        if "path" not in args:
            return _missing_argument("read", "path")
        max_chars = int(args.get("max_chars", 6_000))
        return read_file(workspace, path=str(args["path"]), max_chars=max_chars)
    if tool == "write":
        if "path" not in args:
            return _missing_argument("write", "path")
        if "content" not in args:
            return _missing_argument("write", "content")
        result = write_file(
            workspace,
            path=str(args["path"]),
            content=str(args["content"]),
        )
        return result.to_dict()
    if tool == "edit":
        if "path" not in args:
            return _missing_argument("edit", "path")
        if "old" not in args:
            return _missing_argument("edit", "old")
        if "new" not in args:
            return _missing_argument("edit", "new")
        result = edit_file(
            workspace,
            path=str(args["path"]),
            old=str(args["old"]),
            new=str(args["new"]),
            replace_all=bool(args.get("replace_all", True)),
        )
        return result.to_dict()
    if tool == "bash":
        if "command" not in args:
            return _missing_argument("bash", "command")
        result = run_bash(
            workspace,
            command=str(args["command"]),
            timeout_seconds=bash_timeout_seconds,
            backend=execution_backend,
            docker_image=docker_image,
            docker_network=docker_network,
        )
        return result.to_dict()
    if tool == "submit":
        result = submit(message=str(args.get("message", "")))
        return result.to_dict()
    return ToolResult(ok=False, stderr=f"unknown tool: {tool}").to_dict()
