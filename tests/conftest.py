"""Shared pytest configuration."""

from __future__ import annotations

import matplotlib

# The test suite generates figures; use a non-interactive backend so it also
# runs in headless environments (CI, SSH, containers).
matplotlib.use("Agg")
