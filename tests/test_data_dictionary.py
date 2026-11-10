"""The released data dictionary must describe every column the package ships.

The dictionary lives in the public release tree, so this module is skipped in
the development tree and enforced in the release repository.
"""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DICTIONARY = ROOT / "DATA_DICTIONARY.md"

pytestmark = pytest.mark.skipif(
    not DICTIONARY.exists(), reason="data dictionary is only generated for the release tree"
)


def _aggregate_csvs() -> list[Path]:
    return sorted((ROOT / "runs").glob("*/aggregated/**/*.csv"))


def _aggregate_files() -> list[Path]:
    return sorted(
        path
        for path in (ROOT / "runs").glob("*/aggregated/**/*")
        if path.is_file()
    )


def test_release_ships_aggregates() -> None:
    assert _aggregate_csvs(), "no aggregate CSV files found under runs/"


def test_every_column_is_documented() -> None:
    text = DICTIONARY.read_text(encoding="utf-8")
    undocumented: dict[str, list[str]] = {}
    for path in _aggregate_csvs():
        with path.open(newline="", encoding="utf-8") as handle:
            columns = next(csv.reader(handle))
        missing = [name for name in columns if f"`{name}`" not in text]
        if missing:
            undocumented[str(path.relative_to(ROOT))] = missing
    assert not undocumented, f"columns missing from DATA_DICTIONARY.md: {undocumented}"


def test_every_aggregate_file_is_named() -> None:
    text = DICTIONARY.read_text(encoding="utf-8")
    missing = [
        str(path.relative_to(ROOT))
        for path in _aggregate_files()
        if path.name not in text
    ]
    assert not missing, f"files missing from DATA_DICTIONARY.md: {missing}"
