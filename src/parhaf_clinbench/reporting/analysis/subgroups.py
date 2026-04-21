"""Subgroup scoring helpers.

Each helper slices documents by a meaningful axis (length quartile,
speciality, label, negation, response class, scenario field) and recomputes
the task's official F1 on each slice by reusing the shipped scoring modules
under ``scoring/``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from parhaf_clinbench.core.models import CanonicalDocument, DocumentExample, Record
from parhaf_clinbench.reporting.analysis.rescoring import _predictions_to_canonical
from parhaf_clinbench.reporting.loader import RunArtifacts
from parhaf_clinbench.scoring.infectio import compute_infectio_metrics
from parhaf_clinbench.scoring.pseudo import compute_pseudo_metrics
from parhaf_clinbench.scoring.response import compute_response_metrics
from parhaf_clinbench.scoring.scenario import compute_scenario_metrics

SCORERS: dict[str, Callable[..., Any]] = {
    "pseudo": compute_pseudo_metrics,
    "infectio": compute_infectio_metrics,
    "response": compute_response_metrics,
    "scenario": compute_scenario_metrics,
}


def _official_f1(task: str, preds: list[CanonicalDocument], refs: list[CanonicalDocument]) -> tuple[float, float, float, int]:
    if not refs:
        return (float("nan"), float("nan"), float("nan"), 0)
    comp = SCORERS[task](predictions=preds, references=refs, robustness={})
    off = comp.metrics.official
    return (off.f1, off.precision, off.recall, len(refs))


def _keep_label(record: Record, label: str) -> bool:
    """Return whether a record belongs to the requested label slice."""

    return record.label == label


def _keep_negation(record: Record, polarity: str) -> bool:
    """Return whether an infectiology record matches the requested polarity."""

    return str(record.attributes.get("negation", "null")) == polarity


def _make_label_filter(label: str) -> Callable[[Record], bool]:
    """Build a label predicate with a stable type for mypy."""

    return lambda record: _keep_label(record, label)


def _make_negation_filter(polarity: str) -> Callable[[Record], bool]:
    """Build a negation predicate with a stable type for mypy."""

    return lambda record: _keep_negation(record, polarity)


def _length_quartile(text: str, edges: list[float]) -> str:
    n = len(text)
    for idx, edge in enumerate(edges):
        if n <= edge:
            return f"Q{idx + 1}"
    return f"Q{len(edges) + 1}"


def score_by_length(
    suite: dict[str, RunArtifacts],
    gold: dict[str, list[DocumentExample]],
) -> pd.DataFrame:
    """F1 by document-length quartile, computed per (model, track, task)."""

    rows: list[dict[str, Any]] = []
    for task, examples in gold.items():
        lengths = np.array([len(ex.text) for ex in examples])
        if len(lengths) == 0:
            continue
        edges = list(np.quantile(lengths, [0.25, 0.5, 0.75]))
        buckets = [_length_quartile(ex.text, edges) for ex in examples]
        refs_full = [ex.gold for ex in examples]
        for model, run in suite.items():
            for track_block in run.metrics.get("tracks", []):
                track = track_block["track"]
                if task not in track_block.get("per_task", {}):
                    continue
                preds_full = _predictions_to_canonical(run, track, task, examples)
                for q in sorted(set(buckets)):
                    idx = [i for i, b in enumerate(buckets) if b == q]
                    preds = [preds_full[i] for i in idx]
                    refs = [refs_full[i] for i in idx]
                    f1, p, r, n = _official_f1(task, preds, refs)
                    rows.append(
                        {
                            "model": model,
                            "track": track,
                            "task": task,
                            "subgroup_kind": "length_quartile",
                            "subgroup": q,
                            "precision": p,
                            "recall": r,
                            "f1": f1,
                            "n_docs": n,
                        }
                    )
    return pd.DataFrame(rows)


def score_by_speciality(
    suite: dict[str, RunArtifacts],
    gold: dict[str, list[DocumentExample]],
) -> pd.DataFrame:
    """F1 stratified by medical speciality for the scenario task.

    Speciality is only consistently populated on the scenario corpus so we
    restrict this breakdown to that task.
    """

    rows: list[dict[str, Any]] = []
    task = "scenario"
    examples = gold.get(task) or []
    if not examples:
        return pd.DataFrame(rows)
    refs_full = [ex.gold for ex in examples]
    specs = [ex.gold.speciality or "UNKNOWN" for ex in examples]
    for model, run in suite.items():
        for track_block in run.metrics.get("tracks", []):
            track = track_block["track"]
            if task not in track_block.get("per_task", {}):
                continue
            preds_full = _predictions_to_canonical(run, track, task, examples)
            for spec in sorted(set(specs)):
                idx = [i for i, s in enumerate(specs) if s == spec]
                if not idx:
                    continue
                preds = [preds_full[i] for i in idx]
                refs = [refs_full[i] for i in idx]
                f1, p, r, n = _official_f1(task, preds, refs)
                rows.append(
                    {
                        "model": model,
                        "track": track,
                        "task": task,
                        "subgroup_kind": "speciality",
                        "subgroup": spec,
                        "precision": p,
                        "recall": r,
                        "f1": f1,
                        "n_docs": n,
                    }
                )
    return pd.DataFrame(rows)


def _record_filter_eval(
    task: str,
    preds_full: list[CanonicalDocument],
    refs_full: list[CanonicalDocument],
    keep: Callable[[Record], bool],
) -> tuple[float, float, float, int]:
    """Filter gold/pred records by a predicate and rescore the task.

    Applying the *same* predicate to predictions is the right thing to do only
    for the text/label-driven slices (by label / by negation / by class /
    by field). For slices where predictions should be matched against all
    preds regardless of their label (not the case here), ``keep`` should
    return True unconditionally on preds.
    """

    refs_filtered: list[CanonicalDocument] = []
    preds_filtered: list[CanonicalDocument] = []
    for pred, ref in zip(preds_full, refs_full, strict=True):
        refs_filtered.append(
            CanonicalDocument(
                document_id=ref.document_id,
                task=ref.task,
                speciality=ref.speciality,
                records=[r for r in ref.records if keep(r)],
            )
        )
        preds_filtered.append(
            CanonicalDocument(
                document_id=pred.document_id,
                task=pred.task,
                speciality=pred.speciality,
                records=[r for r in pred.records if keep(r)],
            )
        )
    return _official_f1(task, preds_filtered, refs_filtered)


def score_by_label(
    suite: dict[str, RunArtifacts],
    gold: dict[str, list[DocumentExample]],
) -> pd.DataFrame:
    """F1 per label for pseudo, infectio, response and scenario."""

    rows: list[dict[str, Any]] = []
    for task, examples in gold.items():
        refs_full = [ex.gold for ex in examples]
        labels = sorted({rec.label for ref in refs_full for rec in ref.records})
        for model, run in suite.items():
            for track_block in run.metrics.get("tracks", []):
                track = track_block["track"]
                if task not in track_block.get("per_task", {}):
                    continue
                preds_full = _predictions_to_canonical(run, track, task, examples)
                for label in labels:
                    f1, p, r, n = _record_filter_eval(
                        task,
                        preds_full,
                        refs_full,
                        _make_label_filter(label),
                    )
                    rows.append(
                        {
                            "model": model,
                            "track": track,
                            "task": task,
                            "subgroup_kind": "label",
                            "subgroup": label,
                            "precision": p,
                            "recall": r,
                            "f1": f1,
                            "n_docs": n,
                        }
                    )
    return pd.DataFrame(rows)


def score_by_negation(
    suite: dict[str, RunArtifacts],
    gold: dict[str, list[DocumentExample]],
) -> pd.DataFrame:
    """F1 per infectiology negation polarity."""

    rows: list[dict[str, Any]] = []
    task = "infectio"
    examples = gold.get(task) or []
    refs_full = [ex.gold for ex in examples]
    polarities = sorted(
        {
            str(rec.attributes.get("negation", "null"))
            for ref in refs_full
            for rec in ref.records
        }
    )
    for model, run in suite.items():
        for track_block in run.metrics.get("tracks", []):
            track = track_block["track"]
            if task not in track_block.get("per_task", {}):
                continue
            preds_full = _predictions_to_canonical(run, track, task, examples)
            for pol in polarities:
                f1, p, r, n = _record_filter_eval(
                    task,
                    preds_full,
                    refs_full,
                    _make_negation_filter(pol),
                )
                rows.append(
                    {
                        "model": model,
                        "track": track,
                        "task": task,
                        "subgroup_kind": "negation",
                        "subgroup": pol,
                        "precision": p,
                        "recall": r,
                        "f1": f1,
                        "n_docs": n,
                    }
                )
    return pd.DataFrame(rows)
