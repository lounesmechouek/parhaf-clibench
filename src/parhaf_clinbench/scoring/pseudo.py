"""Scoring logic for the pseudonymization task."""

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


def _texts_counter(doc: CanonicalDocument) -> Counter[str]:
    """Count normalized record texts for loose text-level matching."""

    counter: Counter[str] = Counter()
    for rec in doc.records:
        text = normalize_text(rec.text)
        if text:
            counter[text] += 1
    return counter


def _spans_counter(doc: CanonicalDocument) -> Counter[str]:
    """Count `(start,end)` spans for official pseudonymization scoring."""

    counter: Counter[str] = Counter()
    for rec in doc.records:
        if rec.start is None or rec.end is None:
            continue
        counter[f"{rec.start}:{rec.end}"] += 1
    return counter


def _span_label_counter(doc: CanonicalDocument) -> Counter[str]:
    """Count `(start,end,label)` tuples for stricter secondary scoring."""

    counter: Counter[str] = Counter()
    for rec in doc.records:
        if rec.start is None or rec.end is None:
            continue
        counter[f"{rec.start}:{rec.end}:{rec.label}"] += 1
    return counter


def compute_pseudo_metrics(
    *,
    predictions: list[CanonicalDocument],
    references: list[CanonicalDocument],
    robustness: dict[str, float],
) -> ScoreComputation:
    """Compute pseudonymization metrics.

    The official metric is micro F1 over span tuples `(start, end)`.

    Args:
        predictions: Predicted canonical documents.
        references: Gold canonical documents.
        robustness: Robustness metrics computed during inference.

    Returns:
        Full score computation for the pseudo task.
    """

    text_counts: list[DocCounts] = []
    span_counts: list[DocCounts] = []
    span_label_counts: list[DocCounts] = []

    for pred, gold in zip(predictions, references, strict=True):
        text_counts.append(counter_tp_fp_fn(_texts_counter(pred), _texts_counter(gold)))
        span_counts.append(counter_tp_fp_fn(_spans_counter(pred), _spans_counter(gold)))
        span_label_counts.append(counter_tp_fp_fn(_span_label_counter(pred), _span_label_counter(gold)))

    official_agg = aggregate_doc_counts(span_counts)
    official_score = micro_from_counts(official_agg.tp, official_agg.fp, official_agg.fn)

    text_agg = aggregate_doc_counts(text_counts)
    span_label_agg = aggregate_doc_counts(span_label_counts)

    metrics = TaskMetrics(
        task=TaskId.PSEUDO,
        official=official_score,
        official_name="micro_f1_span",
        secondary={
            "micro_f1_text": micro_from_counts(text_agg.tp, text_agg.fp, text_agg.fn),
            "micro_f1_span_label": micro_from_counts(
                span_label_agg.tp,
                span_label_agg.fp,
                span_label_agg.fn,
            ),
        },
        robustness=robustness,
    )
    return ScoreComputation(metrics=metrics, official_doc_counts=span_counts)
