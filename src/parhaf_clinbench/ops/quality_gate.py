"""Blocking local quality gate (lint, typing, tests)."""

from __future__ import annotations

import subprocess


def run_local_quality_gate() -> None:
    """Run blocking local quality checks.

    Raises:
        RuntimeError: If one quality check fails.
    """

    commands = [
        ["uv", "run", "ruff", "check", "src", "tests"],
        ["uv", "run", "mypy"],
        ["uv", "run", "pytest"],
    ]
    for cmd in commands:
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"Gate locale échouée sur: {' '.join(cmd)}")
