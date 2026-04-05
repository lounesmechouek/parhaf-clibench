"""Lightweight exports for external plotting tools."""

from __future__ import annotations

from pathlib import Path

from parhaf_clinbench.core.models import TrackReport
from parhaf_clinbench.reporting.tables import track_table_rows


def export_metrics_csv(path: Path, reports: list[TrackReport]) -> None:
    """Export a minimal CSV file without plotting dependencies.

    Args:
        path: Output CSV path.
        reports: Per-track report models.
    """

    lines = ["track,task,official_metric,precision,recall,f1,ci_low,ci_high"]
    for report in reports:
        for row in track_table_rows(report):
            lines.append(
                f"{row['track']},{row['task']},{row['official_metric']},"
                f"{row['precision']},{row['recall']},{row['f1']},"
                f"{row['ci_low']},{row['ci_high']}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
