"""Tabular projections for reporting outputs."""

from __future__ import annotations

from parhaf_clinbench.core.models import TrackReport


def track_table_rows(report: TrackReport) -> list[dict[str, float | str]]:
    """Build tabular rows for a single track report.

    Args:
        report: Track-level report.

    Returns:
        List of row dictionaries used by CSV/markdown exporters.
    """

    rows: list[dict[str, float | str]] = []
    for task, metrics in report.per_task.items():
        boot = report.per_task_bootstrap[task]
        rows.append(
            {
                "track": report.track.value,
                "task": task,
                "official_metric": metrics.official_name,
                "precision": metrics.official.precision,
                "recall": metrics.official.recall,
                "f1": metrics.official.f1,
                "ci_low": boot.ci_low,
                "ci_high": boot.ci_high,
            }
        )
    rows.append(
        {
            "track": report.track.value,
            "task": "GLOBAL",
            "official_metric": "mean_task_f1",
            "precision": 0.0,
            "recall": 0.0,
            "f1": report.global_score,
            "ci_low": report.global_bootstrap.ci_low,
            "ci_high": report.global_bootstrap.ci_high,
        }
    )
    return rows
