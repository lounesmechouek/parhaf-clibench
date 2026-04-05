from __future__ import annotations

import pytest

from parhaf_clinbench.core.enums import TaskId
from parhaf_clinbench.orchestration.runner import _empty_prediction


def test_empty_prediction_requires_speciality_for_scenario() -> None:
    with pytest.raises(ValueError, match="sans `speciality`"):
        _empty_prediction("doc-1", TaskId.SCENARIO, None)


def test_empty_prediction_scenario_with_speciality_is_valid() -> None:
    document = _empty_prediction("doc-1", TaskId.SCENARIO, "CARDIOLOGIE")
    assert document.speciality == "CARDIOLOGIE"
    assert document.records == []
