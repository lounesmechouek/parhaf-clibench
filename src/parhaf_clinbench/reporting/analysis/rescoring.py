"""Independently re-compute benchmark scores from raw run artefacts.

The runner persists `metrics.json` at inference time. For a staff-grade
analysis we should not blindly trust that file — we rebuild the canonical
predictions from `predictions.jsonl`, reload the gold from the cached HF
corpora and re-invoke the shipped ``scoring/*`` modules. The two sides of the
comparison are then joined on ``(model, track, task)`` in a single audit
frame.

This module also ships the spec-compliant scenario re-scoring that restricts
the official metric to *verbatim* gold fields — i.e. records whose
``text`` actually appears inside the source document — since the shipped
``compute_scenario_metrics`` does not enforce that constraint.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import ValidationError

from parhaf_clinbench.core.enums import TaskId
from parhaf_clinbench.core.models import CanonicalDocument, DocumentExample, Record
from parhaf_clinbench.reporting.analysis.local_gold import load_gold_examples_offline
from parhaf_clinbench.reporting.loader import RunArtifacts
from parhaf_clinbench.scoring.bootstrap import bootstrap_official_score, bootstrap_paired_delta
from parhaf_clinbench.scoring.common import DocCounts, ScoreComputation
from parhaf_clinbench.scoring.infectio import compute_infectio_metrics
from parhaf_clinbench.scoring.pseudo import compute_pseudo_metrics
from parhaf_clinbench.scoring.response import compute_response_metrics
from parhaf_clinbench.scoring.scenario import compute_scenario_metrics


@dataclass
class RescoredTask:
    """Re-computed metric bundle for one ``(model, track, task)`` cell."""

    model: str
    track: str
    task: str
    shipped_f1: float
    rescored_f1: float
    rescored_precision: float
    rescored_recall: float
    ci_low: float
    ci_high: float
    n_docs: int
    status: str  # "ok" | "mismatch" | "error"
    note: str = ""


def _empty_doc(task: TaskId, document_id: str) -> CanonicalDocument:
    """Empty prediction used when a row is missing or failed parsing."""

    kwargs: dict[str, Any] = {"document_id": document_id, "task": task, "records": []}
    if task == TaskId.SCENARIO:
        # Scenario task requires a speciality; pick a sentinel for empty preds.
        kwargs["speciality"] = "MEDECINE INTERNE"
    return CanonicalDocument(**kwargs)


def _parsed_to_canonical(parsed: Any, task: TaskId, doc_id: str) -> CanonicalDocument:
    """Best-effort conversion of a stored ``parsed`` payload to a CanonicalDocument.

    The runner already performed full schema validation at inference time;
    here we accept None (schema-invalid row) as an empty prediction rather
    than failing the audit on a per-document basis.
    """

    if parsed is None or not isinstance(parsed, dict):
        return _empty_doc(task, doc_id)
    payload = dict(parsed)
    payload.setdefault("document_id", doc_id)
    payload["task"] = task.value
    try:
        return CanonicalDocument.model_validate(payload)
    except ValidationError:
        return _empty_doc(task, doc_id)


def _predictions_to_canonical(
    run: RunArtifacts, track: str, task: str, references: list[DocumentExample]
) -> list[CanonicalDocument]:
    """Align stored predictions to the ordered reference documents."""

    task_id = TaskId(task)
    by_doc: dict[str, dict[str, Any]] = {}
    if not run.predictions.empty:
        df = run.predictions
        mask = (df["task"] == task) & (df["track"] == track)
        for _, row in df[mask].iterrows():
            by_doc[str(row["document_id"])] = row.get("parsed")

    canonical: list[CanonicalDocument] = []
    for ref in references:
        parsed = by_doc.get(ref.document_id)
        canonical.append(_parsed_to_canonical(parsed, task_id, ref.document_id))
    return canonical


def _compute_for_task(
    task: str,
    predictions: list[CanonicalDocument],
    references: list[CanonicalDocument],
) -> ScoreComputation:
    rob: dict[str, float] = {}
    if task == "pseudo":
        return compute_pseudo_metrics(predictions=predictions, references=references, robustness=rob)
    if task == "infectio":
        return compute_infectio_metrics(predictions=predictions, references=references, robustness=rob)
    if task == "response":
        return compute_response_metrics(predictions=predictions, references=references, robustness=rob)
    if task == "scenario":
        return compute_scenario_metrics(predictions=predictions, references=references, robustness=rob)
    raise ValueError(f"unknown task: {task}")


def load_gold_examples(cache_dir: Path) -> dict[str, list[DocumentExample]]:
    """Load gold reference examples strictly from local cache files.

    The analysis layer must remain network-independent for reproducibility.
    """

    return load_gold_examples_offline(cache_dir)


def rescore_suite(
    suite: dict[str, RunArtifacts],
    gold: dict[str, list[DocumentExample]],
    *,
    bootstrap_reps: int = 1000,
    tolerance: float = 1e-6,
) -> pd.DataFrame:
    """Re-score every ``(model, track, task)`` cell and audit against shipped metrics."""

    records: list[RescoredTask] = []
    for model, run in suite.items():
        for track_block in run.metrics.get("tracks", []):
            track = track_block["track"]
            for task, tb in track_block.get("per_task", {}).items():
                try:
                    references = [ex.gold for ex in gold[task]]
                    predictions = _predictions_to_canonical(run, track, task, gold[task])
                    computed = _compute_for_task(task, predictions, references)
                    bs = bootstrap_official_score(
                        doc_counts=computed.official_doc_counts,
                        repetitions=bootstrap_reps,
                    )
                    rescored_f1 = computed.metrics.official.f1
                    shipped_f1 = float(tb["official"]["f1"])
                    status = "ok" if abs(rescored_f1 - shipped_f1) <= tolerance else "mismatch"
                    note = ""
                    if status == "mismatch":
                        note = f"delta={rescored_f1 - shipped_f1:+.4g}"
                    records.append(
                        RescoredTask(
                            model=model,
                            track=track,
                            task=task,
                            shipped_f1=shipped_f1,
                            rescored_f1=rescored_f1,
                            rescored_precision=computed.metrics.official.precision,
                            rescored_recall=computed.metrics.official.recall,
                            ci_low=bs.ci_low,
                            ci_high=bs.ci_high,
                            n_docs=len(references),
                            status=status,
                            note=note,
                        )
                    )
                except Exception as exc:
                    records.append(
                        RescoredTask(
                            model=model,
                            track=track,
                            task=task,
                            shipped_f1=float(tb["official"]["f1"]),
                            rescored_f1=float("nan"),
                            rescored_precision=float("nan"),
                            rescored_recall=float("nan"),
                            ci_low=float("nan"),
                            ci_high=float("nan"),
                            n_docs=0,
                            status="error",
                            note=str(exc)[:200],
                        )
                    )
    return pd.DataFrame([rec.__dict__ for rec in records])


def collect_official_doc_counts(
    suite: dict[str, RunArtifacts],
    gold: dict[str, list[DocumentExample]],
) -> dict[tuple[str, str, str], list[DocCounts]]:
    """Collect per-document official TP/FP/FN counts for every score cell.

    Returns:
        Mapping keyed by ``(model, track, task)``.
    """

    out: dict[tuple[str, str, str], list[DocCounts]] = {}
    for model, run in suite.items():
        for track_block in run.metrics.get("tracks", []):
            track = str(track_block["track"])
            for task in track_block.get("per_task", {}):
                references = [ex.gold for ex in gold[task]]
                predictions = _predictions_to_canonical(run, track, task, gold[task])
                computed = _compute_for_task(task, predictions, references)
                out[(model, track, task)] = computed.official_doc_counts
    return out


def paired_bootstrap_deltas(
    doc_counts: dict[tuple[str, str, str], list[DocCounts]],
    *,
    comparisons: list[dict[str, str]],
    bootstrap_reps: int = 1000,
) -> pd.DataFrame:
    """Compute paired bootstrap deltas for arbitrary score-cell comparisons.

    Each item in ``comparisons`` must provide:
    ``model_a, track_a, task_a, model_b, track_b, task_b``.
    Optional field ``label`` is propagated to the output.
    """

    rows: list[dict[str, Any]] = []
    for cmp in comparisons:
        model_a = cmp["model_a"]
        track_a = cmp["track_a"]
        task_a = cmp["task_a"]
        model_b = cmp["model_b"]
        track_b = cmp["track_b"]
        task_b = cmp["task_b"]

        key_a = (model_a, track_a, task_a)
        key_b = (model_b, track_b, task_b)
        counts_a = doc_counts.get(key_a)
        counts_b = doc_counts.get(key_b)
        if counts_a is None or counts_b is None:
            rows.append(
                {
                    "label": cmp.get("label", ""),
                    "model_a": model_a,
                    "track_a": track_a,
                    "task_a": task_a,
                    "model_b": model_b,
                    "track_b": track_b,
                    "task_b": task_b,
                    "delta": float("nan"),
                    "ci_low": float("nan"),
                    "ci_high": float("nan"),
                    "n_docs": 0,
                    "status": "missing",
                }
            )
            continue
        if len(counts_a) != len(counts_b):
            rows.append(
                {
                    "label": cmp.get("label", ""),
                    "model_a": model_a,
                    "track_a": track_a,
                    "task_a": task_a,
                    "model_b": model_b,
                    "track_b": track_b,
                    "task_b": task_b,
                    "delta": float("nan"),
                    "ci_low": float("nan"),
                    "ci_high": float("nan"),
                    "n_docs": min(len(counts_a), len(counts_b)),
                    "status": "length_mismatch",
                }
            )
            continue
        bs = bootstrap_paired_delta(
            model_a=counts_a,
            model_b=counts_b,
            repetitions=bootstrap_reps,
        )
        rows.append(
            {
                "label": cmp.get("label", ""),
                "model_a": model_a,
                "track_a": track_a,
                "task_a": task_a,
                "model_b": model_b,
                "track_b": track_b,
                "task_b": task_b,
                "delta": bs.score_full,
                "ci_low": bs.ci_low,
                "ci_high": bs.ci_high,
                "n_docs": len(counts_a),
                "status": "ok",
            }
        )
    return pd.DataFrame(rows)


def build_fewshot_vs_zeroshot_deltas(
    doc_counts: dict[tuple[str, str, str], list[DocCounts]],
    *,
    bootstrap_reps: int = 1000,
) -> pd.DataFrame:
    """Compute paired deltas ``few-shot - zero-shot`` for each model/task."""

    models = sorted({model for model, _track, _task in doc_counts})
    tasks = sorted({task for _model, _track, task in doc_counts})
    comparisons: list[dict[str, str]] = []
    for model in models:
        for task in tasks:
            if (model, "few-shot", task) not in doc_counts:
                continue
            if (model, "zero-shot", task) not in doc_counts:
                continue
            comparisons.append(
                {
                    "label": "fewshot_minus_zeroshot",
                    "model_a": model,
                    "track_a": "few-shot",
                    "task_a": task,
                    "model_b": model,
                    "track_b": "zero-shot",
                    "task_b": task,
                }
            )
    return paired_bootstrap_deltas(
        doc_counts,
        comparisons=comparisons,
        bootstrap_reps=bootstrap_reps,
    )


def build_vs_baseline_deltas(
    doc_counts: dict[tuple[str, str, str], list[DocCounts]],
    *,
    baseline_model: str,
    baseline_track: str = "zero-shot",
    bootstrap_reps: int = 1000,
) -> pd.DataFrame:
    """Compute paired deltas ``model - baseline`` for each task on one track."""

    models = sorted({model for model, _track, _task in doc_counts})
    tasks = sorted({task for _model, _track, task in doc_counts})
    comparisons: list[dict[str, str]] = []
    for model in models:
        if model == baseline_model:
            continue
        for task in tasks:
            if (model, baseline_track, task) not in doc_counts:
                continue
            if (baseline_model, baseline_track, task) not in doc_counts:
                continue
            comparisons.append(
                {
                    "label": f"vs_{baseline_model}",
                    "model_a": model,
                    "track_a": baseline_track,
                    "task_a": task,
                    "model_b": baseline_model,
                    "track_b": baseline_track,
                    "task_b": task,
                }
            )
    return paired_bootstrap_deltas(
        doc_counts,
        comparisons=comparisons,
        bootstrap_reps=bootstrap_reps,
    )


# ---------------------------------------------------------------------------
# Scenario spec-compliant re-scoring (verbatim-only gold records)
# ---------------------------------------------------------------------------


def _nfc_lower(value: str) -> str:
    return unicodedata.normalize("NFC", value).casefold()


def _is_verbatim_record(record: Record, source_text: str) -> bool:
    """Return True iff ``record.text`` is a case-insensitive substring of source."""

    if record.text is None:
        return False
    return _nfc_lower(record.text.strip()) in _nfc_lower(source_text)


def filter_scenario_verbatim(
    gold: list[DocumentExample],
) -> list[DocumentExample]:
    """Return a copy of the gold examples where non-verbatim records are dropped.

    The `BENCHMARK PARTAGES` specification restricts the scenario official
    metric to fields whose gold text is a verbatim substring of the source
    document. The shipped scoring module does not apply this filter, so we
    materialise a filtered copy of the gold examples for the spec-compliant
    re-scoring pass.
    """

    filtered: list[DocumentExample] = []
    for ex in gold:
        kept_records = [rec for rec in ex.gold.records if _is_verbatim_record(rec, ex.text)]
        new_gold = CanonicalDocument(
            document_id=ex.gold.document_id,
            task=ex.gold.task,
            speciality=ex.gold.speciality,
            records=kept_records,
        )
        filtered.append(
            DocumentExample(
                document_id=ex.document_id,
                task=ex.task,
                speciality=ex.speciality,
                text=ex.text,
                gold=new_gold,
            )
        )
    return filtered


def rescore_scenario_verbatim(
    suite: dict[str, RunArtifacts],
    gold_scenario: list[DocumentExample],
    *,
    bootstrap_reps: int = 1000,
) -> pd.DataFrame:
    """Spec-compliant scenario re-scoring restricted to verbatim gold fields."""

    filtered = filter_scenario_verbatim(gold_scenario)
    references = [ex.gold for ex in filtered]
    rows: list[dict[str, Any]] = []
    for model, run in suite.items():
        for track_block in run.metrics.get("tracks", []):
            track = track_block["track"]
            if "scenario" not in track_block.get("per_task", {}):
                continue
            preds = _predictions_to_canonical(run, track, "scenario", filtered)
            computed = compute_scenario_metrics(
                predictions=preds, references=references, robustness={}
            )
            bs = bootstrap_official_score(
                doc_counts=computed.official_doc_counts, repetitions=bootstrap_reps
            )
            rows.append(
                {
                    "model": model,
                    "track": track,
                    "task": "scenario",
                    "rescored_f1_verbatim": computed.metrics.official.f1,
                    "precision": computed.metrics.official.precision,
                    "recall": computed.metrics.official.recall,
                    "ci_low": bs.ci_low,
                    "ci_high": bs.ci_high,
                    "n_docs": len(references),
                    "n_records_kept": sum(len(ex.gold.records) for ex in filtered),
                }
            )
    return pd.DataFrame(rows)
