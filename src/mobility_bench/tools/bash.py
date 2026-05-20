"""Bash tool implementation."""

from __future__ import annotations

import re
import shlex
import sys
from pathlib import Path

from mobility_bench.runtime.sandbox import run_command, run_command_in_docker
from mobility_bench.runtime.workspace import Workspace
from mobility_bench.tools.base import ToolResult


ABSOLUTE_PATH_RE = re.compile(r"(?<![\w.-])/(?:[^\s\"'`;|&<>$\\]*)")
BANNED_COMMAND_PATTERNS = (
    re.compile(r"\b(?:python\d*(?:\.\d+)?\s+-m\s+)?pip\s+install\b"),
    re.compile(r"\b(?:source|\.)\s+\S*activate\b"),
    re.compile(r"\bconda\s+activate\b"),
)
ALLOWED_ABSOLUTE_PREFIXES = (
    Path("/bin"),
    Path("/lib"),
    Path("/lib64"),
    Path("/usr"),
)
ALLOWED_ABSOLUTE_PATHS = {Path(sys.executable).resolve()}


def _command_policy_error(workspace: Workspace, command: str) -> str | None:
    for pattern in BANNED_COMMAND_PATTERNS:
        if pattern.search(command):
            return "bash command attempts to modify or activate the host Python environment"

    workspace_root = workspace.root.resolve()
    for match in ABSOLUTE_PATH_RE.finditer(command):
        raw_path = match.group(0).rstrip("),.:]")
        if raw_path == "/":
            return "bash command references filesystem root outside the workspace"
        path = Path(raw_path)
        try:
            resolved = path.resolve(strict=False)
        except OSError:
            resolved = path
        if resolved == workspace_root or workspace_root in resolved.parents:
            continue
        if resolved in ALLOWED_ABSOLUTE_PATHS:
            continue
        if any(resolved == prefix or prefix in resolved.parents for prefix in ALLOWED_ABSOLUTE_PREFIXES):
            continue
        return f"bash command references absolute path outside workspace: {raw_path}"

    try:
        tokens = shlex.split(command)
    except ValueError:
        return None
    for token, next_token in zip(tokens, tokens[1:]):
        if token == "cd" and next_token.startswith("/"):
            target = Path(next_token).resolve(strict=False)
            if target != workspace_root and workspace_root not in target.parents:
                return f"bash command changes directory outside workspace: {next_token}"
    return None


def run_bash(
    workspace: Workspace,
    command: str,
    *,
    timeout_seconds: int = 60,
    backend: str = "local",
    docker_image: str = "mobility-bench-runner:latest",
    docker_network: str = "none",
) -> ToolResult:
    policy_error = _command_policy_error(workspace, command)
    if policy_error:
        return ToolResult(
            ok=False,
            stdout="",
            stderr=policy_error,
            metadata={"recoverable_policy_error": True},
        )
    if backend == "docker":
        result = run_command_in_docker(
            command=command,
            workspace=workspace.root,
            image=docker_image,
            timeout_seconds=timeout_seconds,
            network=docker_network,
        )
    else:
        result = run_command(command, cwd=workspace.root, timeout_seconds=timeout_seconds)
    return ToolResult(
        ok=result.exit_code == 0 and not result.timeout_reached,
        stdout=result.stdout,
        stderr=result.stderr,
        metadata={
            "exit_code": result.exit_code,
            "duration_seconds": result.duration_seconds,
            "timeout_reached": result.timeout_reached,
        },
    )
