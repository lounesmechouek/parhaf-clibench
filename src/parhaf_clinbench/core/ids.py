"""Run identifier helpers."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime


def make_run_id(prefix: str = "run") -> str:
    """Generate a timestamped run identifier.

    Args:
        prefix: Prefix added before timestamp and random suffix.

    Returns:
        Run identifier formatted as `<prefix>_<UTC timestamp>_<short uuid>`.

    Examples:
        >>> run_id = make_run_id("bench")
        >>> run_id.startswith("bench_")
        True
    """

    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    short = uuid.uuid4().hex[:8]
    return f"{prefix}_{stamp}_{short}"
