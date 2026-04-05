"""Scoring logic for structured-scenario extraction."""

from __future__ import annotations

from collections import Counter

from parhaf_clinbench.core.enums import TaskId
from parhaf_clinbench.core.models import CanonicalDocument, TaskMetrics
from parhaf_clinbench.parsing.normalize import normalize_text
from parhaf_clinbench.scoring.common import (
    DocCounts,
    ScoreComputation,
    aggregate_doc_counts,
    counter_tp_fp_fn,
    micro_from_counts,
)


def _text_label_counter(doc: CanonicalDocument) -> Counter[str]:
    """Count normalized `(text,label)` tuples."""

    counter: Counter[str] = Counter()
    for rec in doc.records:
        text = normalize_text(rec.text)
        if text:
            counter[f"{text}|{rec.label}"] += 1
    return counter


def _speciality_counter(doc: CanonicalDocument) -> Counter[str]:
    """Count document speciality as a single categorical token."""

    if doc.speciality is None:
        return Counter({"null": 1})
    return Counter({doc.speciality: 1})


def compute_scenario_metrics(
    *,
    predictions: list[CanonicalDocument],
    references: list[CanonicalDocument],
    robustness: dict[str, float],
) -> ScoreComputation:
    """Compute scenario metrics.

    The official metric is micro F1 over `(text, label)` tuples.

    Args:
        predictions: Predicted canonical documents.
        references: Gold canonical documents.
        robustness: Robustness metrics computed during inference.

    Returns:
        Full score computation for the scenario task.
    """

    official_counts: list[DocCounts] = []
    spec_counts: list[DocCounts] = []

    for pred, gold in zip(predictions, references, strict=True):
        official_counts.append(counter_tp_fp_fn(_text_label_counter(pred), _text_label_counter(gold)))
        spec_counts.append(counter_tp_fp_fn(_speciality_counter(pred), _speciality_counter(gold)))

    official_agg = aggregate_doc_counts(official_counts)
    spec_agg = aggregate_doc_counts(spec_counts)

    metrics = TaskMetrics(
        task=TaskId.SCENARIO,
        official=micro_from_counts(official_agg.tp, official_agg.fp, official_agg.fn),
        official_name="micro_f1_text_label",
        secondary={
            "micro_f1_speciality": micro_from_counts(spec_agg.tp, spec_agg.fp, spec_agg.fn),
        },
        robustness=robustness,
    )
    return ScoreComputation(metrics=metrics, official_doc_counts=official_counts)
