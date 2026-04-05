from __future__ import annotations

import pytest

from parhaf_clinbench.core.enums import RuntimeName, TaskId, TrackId
from parhaf_clinbench.orchestration.experiment_plan import SuiteConfig
from parhaf_clinbench.orchestration.runner import _order_models_for_execution, _tracks_for_runtime


def _suite_for_tests() -> SuiteConfig:
    return SuiteConfig(
        suite_id="suite-order-test",
        benchmark_version="v1",
        tracks=[TrackId.ZEROSHOT, TrackId.FEWSHOT],
        tasks=[TaskId.PSEUDO],
        models=["qwen25_7b", "gliner2_multi", "aya_8b"],
        runtime_default=RuntimeName.VLLM,
        runtime_overrides={"gliner2_multi": RuntimeName.GLINER},
    )


def test_order_models_prioritizes_gliner_for_all_selection() -> None:
    ordered = _order_models_for_execution(
        selected_models=["qwen25_7b", "gliner2_multi", "aya_8b"],
        suite=_suite_for_tests(),
        model_selection="all",
    )
    assert ordered == ["gliner2_multi", "qwen25_7b", "aya_8b"]


def test_order_models_keeps_explicit_selection_order() -> None:
    ordered = _order_models_for_execution(
        selected_models=["aya_8b", "gliner2_multi"],
        suite=_suite_for_tests(),
        model_selection="aya_8b",
    )
    assert ordered == ["aya_8b", "gliner2_multi"]


def test_tracks_for_gliner_keep_only_zero_shot_on_all() -> None:
    resolved = _tracks_for_runtime(
        runtime_name=RuntimeName.GLINER,
        selected_tracks=[TrackId.ZEROSHOT, TrackId.FEWSHOT],
        track_selection="all",
    )
    assert resolved == [TrackId.ZEROSHOT]


def test_tracks_for_gliner_rejects_fewshot_only() -> None:
    with pytest.raises(ValueError):
        _tracks_for_runtime(
            runtime_name=RuntimeName.GLINER,
            selected_tracks=[TrackId.FEWSHOT],
            track_selection="fewshot",
        )
