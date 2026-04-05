from __future__ import annotations

from parhaf_clinbench.core.enums import TaskId
from parhaf_clinbench.core.models import ScoreTriplet, TaskMetrics
from parhaf_clinbench.scoring.aggregate import mean_task_f1, median_task_f1


def _metric(task: TaskId, f1: float) -> TaskMetrics:
    triplet = ScoreTriplet(precision=f1, recall=f1, f1=f1)
    return TaskMetrics(
        task=task,
        official=triplet,
        official_name="test_metric",
        secondary={},
        robustness={},
    )


def test_mean_task_f1_computes_arithmetic_mean() -> None:
    metrics = [
        _metric(TaskId.PSEUDO, 1.0),
        _metric(TaskId.INFECTIO, 0.5),
        _metric(TaskId.RESPONSE, 0.0),
    ]
    assert mean_task_f1(metrics) == 0.5


def test_median_task_f1_computes_median() -> None:
    metrics = [
        _metric(TaskId.PSEUDO, 1.0),
        _metric(TaskId.INFECTIO, 0.5),
        _metric(TaskId.RESPONSE, 0.0),
    ]
    assert median_task_f1(metrics) == 0.5
