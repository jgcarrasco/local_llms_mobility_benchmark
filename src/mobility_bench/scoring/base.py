"""Common scoring structures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class CheckerResult:
    success: bool
    partial_score: float
    checker_errors: list[str] = field(default_factory=list)
    failure_category: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
