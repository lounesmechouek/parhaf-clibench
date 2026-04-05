from __future__ import annotations

from parhaf_clinbench.core.enums import TaskId
from parhaf_clinbench.tasks.base import TASK_DEFINITIONS
from parhaf_clinbench.tasks.infectio import OFFICIAL_METRIC as INFECTIO_OFFICIAL_METRIC
from parhaf_clinbench.tasks.pseudo import OFFICIAL_METRIC as PSEUDO_OFFICIAL_METRIC
from parhaf_clinbench.tasks.response import OFFICIAL_METRIC as RESPONSE_OFFICIAL_METRIC
from parhaf_clinbench.tasks.scenario import OFFICIAL_METRIC as SCENARIO_OFFICIAL_METRIC


def test_task_definitions_and_module_constants_are_consistent() -> None:
    assert TASK_DEFINITIONS[TaskId.PSEUDO].official_metric == PSEUDO_OFFICIAL_METRIC
    assert TASK_DEFINITIONS[TaskId.INFECTIO].official_metric == INFECTIO_OFFICIAL_METRIC
    assert TASK_DEFINITIONS[TaskId.RESPONSE].official_metric == RESPONSE_OFFICIAL_METRIC
    assert TASK_DEFINITIONS[TaskId.SCENARIO].official_metric == SCENARIO_OFFICIAL_METRIC
