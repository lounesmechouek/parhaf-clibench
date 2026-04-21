"""Reporting export utilities and derived artifact writers."""

from __future__ import annotations

import json
from pathlib import Path

from parhaf_clinbench.core.models import TrackReport
from parhaf_clinbench.reporting.markdown import render_report_markdown
from parhaf_clinbench.reporting.plots import export_metrics_csv


def export_reports(run_dir: Path, run_id: str, reports: list[TrackReport]) -> None:
    """Write structured reporting outputs for one run.

    Args:
        run_dir: Run output directory.
        run_id: Run identifier.
        reports: Per-track report models.
    """

    serializable = [report.model_dump(mode="json") for report in reports]
    (run_dir / "metrics.json").write_text(
        json.dumps({"tracks": serializable}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "report.md").write_text(
        render_report_markdown(run_id=run_id, reports=reports),
        encoding="utf-8",
    )
    export_metrics_csv(run_dir / "metrics.csv", reports)
