"""Execution helpers for generated code."""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path


SENSITIVE_ENV_MARKERS = (
    "API_KEY",
    "AUTH",
    "CREDENTIAL",
    "KEY",
    "OPENAI",
    "PASSWORD",
    "SECRET",
    "TOKEN",
)


@dataclass(frozen=True)
class CommandResult:
    stdout: str
    stderr: str
    exit_code: int
    duration_seconds: float
    timeout_reached: bool


def sanitized_environment() -> dict[str, str]:
    env: dict[str, str] = {}
    for key, value in os.environ.items():
        upper_key = key.upper()
        if any(marker in upper_key for marker in SENSITIVE_ENV_MARKERS):
            continue
        env[key] = value
    return env


def run_command(command: str, cwd: Path, timeout_seconds: int) -> CommandResult:
    start = time.monotonic()
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            shell=True,
            capture_output=True,
            timeout=timeout_seconds,
            env=sanitized_environment(),
            check=False,
        )
        return CommandResult(
            stdout=completed.stdout.decode("utf-8", errors="replace"),
            stderr=completed.stderr.decode("utf-8", errors="replace"),
            exit_code=completed.returncode,
            duration_seconds=time.monotonic() - start,
            timeout_reached=False,
        )
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
            exit_code=124,
            duration_seconds=time.monotonic() - start,
            timeout_reached=True,
        )


def build_docker_command(
    *,
    command: str,
    workspace: Path,
    image: str,
    network: str = "none",
) -> list[str]:
    return [
        "docker",
        "run",
        "--rm",
        "--network",
        network,
        "-v",
        f"{workspace.resolve()}:/workspace",
        "-w",
        "/workspace",
        image,
        "/bin/sh",
        "-lc",
        command,
    ]


def run_command_in_docker(
    *,
    command: str,
    workspace: Path,
    image: str,
    timeout_seconds: int,
    network: str = "none",
) -> CommandResult:
    start = time.monotonic()
    docker_command = build_docker_command(
        command=command,
        workspace=workspace,
        image=image,
        network=network,
    )
    try:
        completed = subprocess.run(
            docker_command,
            capture_output=True,
            timeout=timeout_seconds,
            env=sanitized_environment(),
            check=False,
        )
        return CommandResult(
            stdout=completed.stdout.decode("utf-8", errors="replace"),
            stderr=completed.stderr.decode("utf-8", errors="replace"),
            exit_code=completed.returncode,
            duration_seconds=time.monotonic() - start,
            timeout_reached=False,
        )
    except subprocess.TimeoutExpired as exc:
        return CommandResult(
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
            exit_code=124,
            duration_seconds=time.monotonic() - start,
            timeout_reached=True,
        )
