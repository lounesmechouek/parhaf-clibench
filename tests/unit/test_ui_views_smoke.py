from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Literal

import pandas as pd
import pytest

import ui.app as app
from ui.views import (
    error_explorer,
    head_to_head,
    leaderboard,
    methodology,
    model_card,
    overview,
    robustness,
    subgroups,
    task_deep_dive,
)


@dataclass
class _DummyContext:
    def __enter__(self) -> _DummyContext:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Literal[False]:
        return False

    def __getattr__(self, name: str) -> Any:
        return _dummy


def _dummy(*_args: Any, **_kwargs: Any) -> Any:
    return None


def _patch_streamlit(monkeypatch: pytest.MonkeyPatch) -> None:
    import streamlit as st

    def _columns(n: Any, **_kw: Any) -> list[_DummyContext]:
        count = n if isinstance(n, int) else len(n)
        return [_DummyContext() for _ in range(count)]

    monkeypatch.setattr(st, "set_page_config", _dummy)
    monkeypatch.setattr(st, "title", _dummy)
    monkeypatch.setattr(st, "caption", _dummy)
    monkeypatch.setattr(st, "markdown", _dummy)
    monkeypatch.setattr(st, "warning", _dummy)
    monkeypatch.setattr(st, "info", _dummy)
    monkeypatch.setattr(st, "success", _dummy)
    monkeypatch.setattr(st, "metric", _dummy)
    monkeypatch.setattr(st, "progress", _dummy)
    monkeypatch.setattr(st, "plotly_chart", _dummy)
    monkeypatch.setattr(st, "dataframe", _dummy)
    monkeypatch.setattr(st, "json", _dummy)
    monkeypatch.setattr(st, "code", _dummy)
    monkeypatch.setattr(st, "latex", _dummy)
    monkeypatch.setattr(st, "selectbox", lambda _label, options, **_kw: options[0] if options else None)
    monkeypatch.setattr(st, "toggle", lambda _label, value=False, **_kw: value)
    monkeypatch.setattr(st, "radio", lambda _label, options, **_kw: options[0] if options else None)
    monkeypatch.setattr(st, "columns", _columns)
    monkeypatch.setattr(st, "tabs", lambda labels, **_kw: [_DummyContext() for _ in labels])
    monkeypatch.setattr(st, "expander", lambda *_args, **_kw: _DummyContext())
    monkeypatch.setattr(st, "sidebar", SimpleNamespace(title=_dummy, radio=lambda _l, options, **_k: options[0], markdown=_dummy, caption=_dummy))


def _sample_scores() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"model": "m1", "track": "zero-shot", "task": "pseudo", "metric_kind": "official", "metric_name": "f1", "precision": 0.5, "recall": 0.5, "f1": 0.5, "ci_low": 0.4, "ci_high": 0.6},
            {"model": "m1", "track": "zero-shot", "task": "infectio", "metric_kind": "official", "metric_name": "f1", "precision": 0.2, "recall": 0.2, "f1": 0.2, "ci_low": 0.1, "ci_high": 0.3},
            {"model": "m1", "track": "zero-shot", "task": "response", "metric_kind": "official", "metric_name": "f1", "precision": 0.1, "recall": 0.1, "f1": 0.1, "ci_low": 0.05, "ci_high": 0.2},
            {"model": "m1", "track": "zero-shot", "task": "scenario", "metric_kind": "official", "metric_name": "f1", "precision": 0.45, "recall": 0.45, "f1": 0.45, "ci_low": 0.4, "ci_high": 0.5},
        ]
    )


def _sample_global_scores() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"model": "m1", "track": "zero-shot", "global_f1": 0.24, "ci_low": 0.2, "ci_high": 0.28},
        ]
    )


def _sample_manifest() -> dict[str, Any]:
    return {"n_models": 1, "n_gold_docs": {"pseudo": 10, "infectio": 3, "response": 2, "scenario": 12}}


def test_overview_and_methodology_render(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_streamlit(monkeypatch)
    monkeypatch.setattr(overview, "load_scores", _sample_scores)
    monkeypatch.setattr(overview, "load_global_scores", _sample_global_scores)
    monkeypatch.setattr(overview, "load_manifest", _sample_manifest)
    overview.render()

    monkeypatch.setattr(methodology, "load_manifest", _sample_manifest)
    monkeypatch.setattr(
        methodology,
        "load_audit",
        lambda: pd.DataFrame(
            [
                {
                    "model": "m1",
                    "track": "zero-shot",
                    "task": "pseudo",
                    "shipped_f1": 0.5,
                    "rescored_f1": 0.5,
                    "rescored_precision": 0.5,
                    "rescored_recall": 0.5,
                    "ci_low": 0.4,
                    "ci_high": 0.6,
                    "n_docs": 10,
                    "status": "ok",
                }
            ]
        ),
    )
    monkeypatch.setattr(
        methodology,
        "load_run_metadata",
        lambda: pd.DataFrame(
            [
                {
                    "model": "m1",
                    "model_hf_id": "hf",
                    "runtime_name": "rt",
                    "runtime_version": "1",
                    "gpu_name": "gpu",
                    "elapsed_seconds": 12,
                }
            ]
        ),
    )
    methodology.render()


def test_other_views_render(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_streamlit(monkeypatch)
    scores = _sample_scores()
    global_scores = _sample_global_scores()

    monkeypatch.setattr(leaderboard, "load_scores", lambda: scores)
    monkeypatch.setattr(leaderboard, "load_global_scores", lambda: global_scores)
    leaderboard.render()

    monkeypatch.setattr(task_deep_dive, "load_scores", lambda: scores)
    monkeypatch.setattr(
        task_deep_dive,
        "load_fewshot_vs_zeroshot",
        lambda: pd.DataFrame([{"task_a": "pseudo", "model_a": "m1", "delta": 0.1, "ci_low": 0.05, "ci_high": 0.15, "n_docs": 10, "status": "ok"}]),
    )
    task_deep_dive.render()

    monkeypatch.setattr(robustness, "load_robustness", lambda: pd.DataFrame([{"model": "m1", "track": "zero-shot", "task": "pseudo", "schema_conformity_rate": 0.9, "empty_output_rate": 0.1, "latency_median_ms": 100.0}]))
    monkeypatch.setattr(robustness, "load_timings", lambda: pd.DataFrame([{"model": "m1", "latency_ms": 120.0}]))
    monkeypatch.setattr(robustness, "load_global_scores", lambda: global_scores)
    robustness.render()

    monkeypatch.setattr(subgroups, "load_subgroups", lambda: pd.DataFrame([{"model": "m1", "track": "zero-shot", "task": "pseudo", "subgroup_kind": "length_quartile", "subgroup": "Q1", "precision": 0.5, "recall": 0.4, "f1": 0.45, "n_docs": 2}]))
    subgroups.render()

    monkeypatch.setattr(error_explorer, "load_error_taxonomy", lambda: pd.DataFrame([{"model": "m1", "track": "zero-shot", "task": "pseudo", "category": "invalid_json", "count": 2}]))
    monkeypatch.setattr(error_explorer, "load_errors", lambda: pd.DataFrame([{"model": "m1", "task": "pseudo", "track": "zero-shot", "error": "invalid json", "document_id": "d1"}]))
    error_explorer.render()

    monkeypatch.setattr(head_to_head, "load_paired_deltas", lambda: pd.DataFrame([{"model_a": "m1", "model_b": "m2", "task_a": "pseudo", "track_a": "zero-shot", "delta": 0.1, "ci_low": 0.05, "ci_high": 0.15}]))
    monkeypatch.setattr(head_to_head, "load_scores", lambda: scores)
    head_to_head.render()

    monkeypatch.setattr(model_card, "load_scores", lambda: scores)
    monkeypatch.setattr(model_card, "load_robustness", lambda: pd.DataFrame([{"model": "m1", "track": "zero-shot", "task": "pseudo", "raw_json_valid_rate": 0.9, "schema_conformity_rate": 0.8, "empty_output_rate": 0.1, "latency_median_ms": 100.0, "throughput_tokens_per_second": 12.0}]))
    monkeypatch.setattr(model_card, "load_timings", lambda: pd.DataFrame([{"model": "m1", "latency_ms": 110.0}]))
    monkeypatch.setattr(model_card, "load_run_metadata", lambda: pd.DataFrame([{"model": "m1", "model_hf_id": "hf", "runtime_name": "rt", "runtime_version": "1", "gpu_name": "gpu"}]))
    monkeypatch.setattr(model_card, "load_error_taxonomy", lambda: pd.DataFrame([{"model": "m1", "task": "pseudo", "track": "zero-shot", "category": "invalid_json", "count": 1}]))
    model_card.render()

    monkeypatch.setattr(app, "_pages", lambda: {"Dummy": lambda: None})
    app.main()
