"""Markdown report generation."""

from __future__ import annotations

from parhaf_clinbench.core.models import TrackReport


def render_report_markdown(run_id: str, reports: list[TrackReport]) -> str:
    """Render a human-readable Markdown report.

    Args:
        run_id: Run identifier.
        reports: Per-track report models.

    Returns:
        Markdown report content.
    """

    lines: list[str] = [f"# PARHAF-CLINBENCH Report - {run_id}", ""]
    for report in reports:
        lines.append(f"## Track: `{report.track.value}`")
        lines.append("")
        lines.append("| Task | Metric | Precision | Recall | F1 | CI95 |")
        lines.append("|---|---|---:|---:|---:|---|")
        for task, metrics in report.per_task.items():
            boot = report.per_task_bootstrap[task]
            lines.append(
                "| "
                f"{task} | {metrics.official_name} | {metrics.official.precision:.4f} | "
                f"{metrics.official.recall:.4f} | {metrics.official.f1:.4f} | "
                f"[{boot.ci_low:.4f}, {boot.ci_high:.4f}] |"
            )
        lines.append(
            "| GLOBAL | mean_task_f1 | - | - | "
            f"{report.global_score:.4f} | [{report.global_bootstrap.ci_low:.4f}, "
            f"{report.global_bootstrap.ci_high:.4f}] |"
        )
        lines.append("")
    return "\n".join(lines)
