"""Base task data structures."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from mobility_bench.scoring.base import CheckerResult


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    title: str
    family: str
    input_files: list[str]
    expected_outputs: list[str]
    prompt_template: str
    checker: Callable[[Path], CheckerResult]
    timeout_seconds: int
    output_requirements: dict[str, list[str]] = field(default_factory=dict)

    def render_prompt(self) -> str:
        input_files = "\n".join(f"- {path}" for path in self.input_files) or "- None"
        expected_outputs = "\n".join(f"- {path}" for path in self.expected_outputs)
        requirements = []
        for path, items in self.output_requirements.items():
            requirements.append(f"- {path}:")
            requirements.extend(f"  - {item}" for item in items)
        requirements_text = "\n".join(requirements) if requirements else "- Follow the task description and required output paths."
        csv_outputs = [path for path in self.expected_outputs if path.lower().endswith(".csv")]
        if csv_outputs:
            requirements_text += (
                "\n- Every CSV output must be RFC 4180 compliant: quote any field that contains a comma, "
                "a double quote, or a newline, so that each row has exactly as many fields as the header."
            )
        return (
            f"Task ID: {self.task_id}\n"
            f"Title: {self.title}\n"
            f"Family: {self.family}\n\n"
            f"{self.prompt_template}\n\n"
            f"Available input files:\n{input_files}\n\n"
            f"Required output files:\n{expected_outputs}\n\n"
            f"Output requirements:\n{requirements_text}\n"
        )
