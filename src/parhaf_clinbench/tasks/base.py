"""Task definitions used by the benchmark pipeline."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from parhaf_clinbench.core.enums import TaskId


class TaskDefinition(BaseModel):
    """Task-level metadata such as official evaluation metric."""

    model_config = ConfigDict(extra="forbid")

    task_id: TaskId
    official_metric: str


TASK_DEFINITIONS: dict[TaskId, TaskDefinition] = {
    TaskId.PSEUDO: TaskDefinition(task_id=TaskId.PSEUDO, official_metric="span_micro_f1"),
    TaskId.INFECTIO: TaskDefinition(
        task_id=TaskId.INFECTIO,
        official_metric="text_label_negation_micro_f1",
    ),
    TaskId.RESPONSE: TaskDefinition(task_id=TaskId.RESPONSE, official_metric="text_label_micro_f1"),
    TaskId.SCENARIO: TaskDefinition(task_id=TaskId.SCENARIO, official_metric="text_label_micro_f1"),
}
