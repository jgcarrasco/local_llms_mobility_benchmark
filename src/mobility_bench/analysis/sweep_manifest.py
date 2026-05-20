"""Sweep manifest generation."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mobility_bench.config import load_benchmark_config, load_models_config
from mobility_bench.tasks.registry import get_all_tasks


def _run(command: list[str]) -> dict[str, Any]:
    executable = shutil.which(command[0])
    if executable is None:
        return {"available": False, "command": command}
    try:
        completed = subprocess.run(
            [executable, *command[1:]],
            text=True,
            capture_output=True,
            timeout=10,
            check=False,
        )
    except Exception as exc:  # pragma: no cover - defensive environment capture
        return {"available": True, "command": command, "error": str(exc)}
    return {
        "available": True,
        "command": command,
        "exit_code": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _file_size(path: Path) -> int | None:
    return path.stat().st_size if path.exists() else None


def _dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())


def _git_status() -> dict[str, Any]:
    return {
        "commit": _run(["git", "rev-parse", "HEAD"]).get("stdout"),
        "branch": _run(["git", "branch", "--show-current"]).get("stdout"),
        "status_short": _run(["git", "status", "--short"]).get("stdout", ""),
    }


def create_sweep_manifest(
    *,
    benchmark_config: str | Path,
    models_config: str | Path,
    output_path: str | Path,
    notes: str | None = None,
) -> Path:
    benchmark_path = Path(benchmark_config)
    models_path = Path(models_config)
    benchmark = load_benchmark_config(benchmark_path)
    models = load_models_config(models_path)
    tasks = get_all_tasks()
    non_mock_models = {model_id: model for model_id, model in models.models.items() if model.backend != "mock"}
    runs_dir = benchmark.paths.runs_dir

    manifest: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "notes": notes,
        "git": _git_status(),
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "platform": platform.platform(),
        },
        "commands": {
            "llama_server_version": _run(["llama-server", "--version"]),
            "nvidia_smi": _run(
                [
                    "nvidia-smi",
                    "--query-gpu=index,name,memory.total,memory.used,utilization.gpu,driver_version",
                    "--format=csv,noheader,nounits",
                ]
            ),
        },
        "configs": {
            "benchmark": {
                "path": str(benchmark_path),
                "size_bytes": _file_size(benchmark_path),
                "runtime": benchmark.runtime.model_dump(mode="json"),
                "execution": benchmark.execution.model_dump(mode="json"),
            },
            "models": {
                "path": str(models_path),
                "size_bytes": _file_size(models_path),
            },
        },
        "models": {
            model_id: {
                "family": model.family,
                "size_b": model.size_b,
                "architecture": model.architecture,
                "backend": model.backend,
                "base_url": model.base_url,
                "model_name": model.model_name,
                "quantization": model.quantization,
                "context_length": model.context_length,
                "source_repo": model.source_repo,
                "artifact_repo": model.artifact_repo,
                "generation": model.generation,
            }
            for model_id, model in non_mock_models.items()
        },
        "tasks": {
            task_id: {
                "title": task.title,
                "family": task.family,
                "input_files": task.input_files,
                "expected_outputs": task.expected_outputs,
            }
            for task_id, task in tasks.items()
        },
        "planned_attempts": len(non_mock_models) * 2 * len(tasks) * benchmark.runtime.repetitions,
        "artifact_footprint_bytes": {
            "runs": _dir_size(runs_dir),
            "raw_logs": _dir_size(runs_dir / "raw_logs"),
            "workspaces": _dir_size(runs_dir / "workspaces"),
            "results_jsonl": _dir_size(runs_dir / "results_jsonl"),
            "aggregated": _dir_size(runs_dir / "aggregated"),
        },
    }

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True, sort_keys=True) + "\n", encoding="utf-8")
    return path
