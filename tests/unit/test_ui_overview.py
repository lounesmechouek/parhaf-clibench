from __future__ import annotations

import math

import pandas as pd

from ui.views import overview as overview_view


def _sample_scores() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"model": "m1", "track": "zero-shot", "task": "pseudo", "metric_kind": "official", "f1": 0.4, "ci_low": 0.3, "ci_high": 0.5},
            {"model": "m1", "track": "zero-shot", "task": "infectio", "metric_kind": "official", "f1": 0.2, "ci_low": 0.1, "ci_high": 0.3},
            {"model": "m1", "track": "zero-shot", "task": "response", "metric_kind": "official", "f1": 0.1, "ci_low": 0.05, "ci_high": 0.15},
            {"model": "m1", "track": "zero-shot", "task": "scenario", "metric_kind": "official", "f1": 0.45, "ci_low": 0.4, "ci_high": 0.5},
            {"model": "m2", "track": "few-shot", "task": "pseudo", "metric_kind": "official", "f1": 0.41, "ci_low": 0.35, "ci_high": 0.48},
            {"model": "m2", "track": "few-shot", "task": "infectio", "metric_kind": "official", "f1": 0.25, "ci_low": 0.2, "ci_high": 0.3},
            {"model": "m2", "track": "few-shot", "task": "response", "metric_kind": "official", "f1": 0.11, "ci_low": 0.08, "ci_high": 0.2},
            {"model": "m2", "track": "few-shot", "task": "scenario", "metric_kind": "official", "f1": 0.47, "ci_low": 0.45, "ci_high": 0.52},
        ]
    )


def _sample_global_scores() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"model": "m1", "track": "zero-shot", "global_f1": 0.24, "ci_low": 0.2, "ci_high": 0.28},
            {"model": "m2", "track": "few-shot", "global_f1": 0.31, "ci_low": 0.29, "ci_high": 0.34},
        ]
    )


def test_readiness_buckets() -> None:
    assert overview_view._readiness(0.7) == ("green", "Production-ready")
    assert overview_view._readiness(0.55) == ("amber", "Pilot only")
    assert overview_view._readiness(0.05) == ("red", "Research only")


def test_best_per_task_and_cards() -> None:
    winners = overview_view._best_per_task(_sample_scores())
    assert {w.task for w in winners} == {"pseudo", "infectio", "response", "scenario"}
    card = overview_view._task_card_html(winners[0])
    assert "task-card" in card


def test_figures_build() -> None:
    winners = overview_view._best_per_task(_sample_scores())
    fig = overview_view._best_per_task_figure(winners)
    assert fig.data
    fig2 = overview_view._system_ranking_figure(_sample_global_scores())
    assert fig2.data


def test_recommendation_html() -> None:
    html = overview_view._recommendation_html(
        {"tone": "amber", "icon": "X", "title": "Test", "body": "Body"}
    )
    assert "callout" in html
    assert "Test" in html


def test_best_per_task_handles_missing_ci() -> None:
    scores = _sample_scores().copy()
    scores.loc[0, "ci_low"] = math.nan
    scores.loc[0, "ci_high"] = math.nan
    winners = overview_view._best_per_task(scores)
    assert winners
