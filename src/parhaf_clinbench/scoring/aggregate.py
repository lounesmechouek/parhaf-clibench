"""Cross-task score aggregation helpers."""

from __future__ import annotations

from statistics import median

from parhaf_clinbench.core.models import TaskMetrics


def mean_task_f1(metrics: list[TaskMetrics]) -> float:
    """Return the arithmetic mean of official task-level F1 scores.

    Args:
        metrics: Task metric list.

    Returns:
        Mean official F1 value.
    """

    if not metrics:
        return 0.0
    return sum(item.official.f1 for item in metrics) / float(len(metrics))



def median_task_f1(metrics: list[TaskMetrics]) -> float:
    """Return the median of official task-level F1 scores.

    Args:
        metrics: Task metric list.

    Returns:
        Median official F1 value.
    """

    if not metrics:
        return 0.0
    return float(median(item.official.f1 for item in metrics))
