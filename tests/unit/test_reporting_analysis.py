from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from parhaf_clinbench.core.enums import TaskId
from parhaf_clinbench.core.models import CanonicalDocument, DocumentExample, Record
from parhaf_clinbench.reporting import loader
from parhaf_clinbench.reporting.analysis import error_taxonomy, frames, rescoring, subgroups
from parhaf_clinbench.reporting.plots_extended import (
    error_taxonomy_stacked_bar,
    fewshot_slopegraph,
    forest_plot,
    global_leaderboard,
    leaderboard_bar,
    pareto_f1_vs_latency,
    robustness_heatmap,
    subgroup_small_multiples,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _sample_run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "m1_20240101T000000Z_abcd"
    run_dir.mkdir(parents=True)
    _write_json(
        run_dir / "metrics.json",
        {
            "tracks": [
                {
                    "track": "zero-shot",
                    "per_task": {
                        "pseudo": {
                            "official_name": "micro_f1",
                            "official": {"precision": 0.5, "recall": 0.5, "f1": 0.5},
                            "secondary": {},
                        }
                    },
                }
            ]
        },
    )
    (run_dir / "metrics.csv").write_text("task,track,f1,ci_low,ci_high\nGLOBAL,zero-shot,0.5,0.4,0.6\n", encoding="utf-8")
    (run_dir / "predictions.jsonl").write_text("", encoding="utf-8")
    (run_dir / "errors.jsonl").write_text("", encoding="utf-8")
    (run_dir / "timings.jsonl").write_text("", encoding="utf-8")
    _write_json(run_dir / "run_metadata.json", {"model_id": "m1"})
    (run_dir / "resolved_config.yaml").write_text("key: value\n", encoding="utf-8")
    (run_dir / "report.md").write_text("report", encoding="utf-8")
    return run_dir


def test_loader_and_frames(tmp_path: Path) -> None:
    run_dir = _sample_run_dir(tmp_path)
    run = loader.load_run(run_dir)
    suite = {"m1": run}

    scores = frames.build_scores_frame(suite)
    assert not scores.empty

    global_scores = frames.build_global_scores_frame(suite)
    assert not global_scores.empty

    robustness = frames.build_robustness_frame(suite)
    assert not robustness.empty

    predictions = frames.build_predictions_frame(suite)
    assert predictions.empty

    errors = frames.build_errors_frame(suite)
    assert list(errors.columns) == ["model", "document_id", "task", "track", "error"]


def test_error_taxonomy_and_plots() -> None:
    assert error_taxonomy.classify_error("invalid json") == "invalid_json"
    assert error_taxonomy.classify_error("end must be >= start") == "offset_drift"

    tax = pd.DataFrame([{"model": "m1", "task": "pseudo", "track": "zero-shot", "category": "invalid_json", "count": 2}])
    assert error_taxonomy_stacked_bar(tax).data

    scores = pd.DataFrame(
        [
            {"model": "m1", "track": "zero-shot", "task": "pseudo", "metric_kind": "official", "f1": 0.5, "ci_low": 0.4, "ci_high": 0.6},
            {"model": "m2", "track": "zero-shot", "task": "pseudo", "metric_kind": "official", "f1": 0.4, "ci_low": 0.3, "ci_high": 0.5},
        ]
    )
    assert leaderboard_bar(scores, track="zero-shot", task="pseudo").data
    assert forest_plot(scores, track="zero-shot").data

    global_scores = pd.DataFrame(
        [
            {"model": "m1", "track": "zero-shot", "global_f1": 0.3, "ci_low": 0.2, "ci_high": 0.4},
            {"model": "m2", "track": "zero-shot", "global_f1": 0.2, "ci_low": 0.1, "ci_high": 0.3},
        ]
    )
    assert global_leaderboard(global_scores).data
    robustness = pd.DataFrame(
        [{"model": "m1", "track": "zero-shot", "task": "pseudo", "schema_conformity_rate": 0.9, "empty_output_rate": 0.1, "latency_median_ms": 100.0}]
    )
    assert robustness_heatmap(robustness).data
    assert pareto_f1_vs_latency(global_scores, robustness).data

    sub_df = pd.DataFrame(
        [{"model": "m1", "track": "zero-shot", "task": "pseudo", "subgroup_kind": "length_quartile", "subgroup": "Q1", "f1": 0.4}]
    )
    assert subgroup_small_multiples(sub_df, task="pseudo", subgroup_kind="length_quartile").data

    scores_lift = pd.DataFrame(
        [
            {"model": "m1", "track": "zero-shot", "task": "pseudo", "metric_kind": "official", "f1": 0.2},
            {"model": "m1", "track": "few-shot", "task": "pseudo", "metric_kind": "official", "f1": 0.4},
        ]
    )
    assert fewshot_slopegraph(scores_lift, task="pseudo").data


def test_rescoring_helpers_and_subgroups() -> None:
    gold_doc = CanonicalDocument(
        document_id="doc-1",
        task=TaskId.PSEUDO,
        records=[Record(label="FIRST_NAME", text="Arun", start=0, end=4)],
    )
    example = DocumentExample(document_id="doc-1", task=TaskId.PSEUDO, speciality=None, text="Arun", gold=gold_doc)

    run = loader.RunArtifacts(
        run_dir=Path("."),
        model_id="m1",
        metrics={
            "tracks": [
                {
                    "track": "zero-shot",
                    "per_task": {
                        "pseudo": {
                            "official_name": "micro_f1",
                            "official": {"precision": 1.0, "recall": 1.0, "f1": 1.0},
                            "secondary": {},
                        }
                    },
                }
            ]
        },
        metrics_csv=pd.DataFrame(),
        predictions=pd.DataFrame(
            [
                {
                    "task": "pseudo",
                    "track": "zero-shot",
                    "document_id": "doc-1",
                    "parsed": {
                        "document_id": "doc-1",
                        "task": "pseudo",
                        "records": [{"label": "FIRST_NAME", "text": "Arun", "start": 0, "end": 4}],
                    },
                }
            ]
        ),
        errors=pd.DataFrame(),
        timings=pd.DataFrame(),
        run_metadata={},
        resolved_config={},
        report_md="",
    )

    parsed = rescoring._parsed_to_canonical(gold_doc.model_dump(), TaskId.PSEUDO, "doc-1")
    assert parsed.document_id == "doc-1"

    suite = {"m1": run}
    gold = {"pseudo": [example]}
    df = subgroups.score_by_length(suite, gold)
    assert not df.empty
