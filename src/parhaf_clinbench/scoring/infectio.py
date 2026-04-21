"""Scoring logic for the infectiology task."""

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


def _text_counter(doc: CanonicalDocument) -> Counter[str]:
    """Count normalized record texts only."""

    counter: Counter[str] = Counter()
    for rec in doc.records:
        text = normalize_text(rec.text)
        if text:
            counter[text] += 1
    return counter


def _text_label_counter(doc: CanonicalDocument) -> Counter[str]:
    """Count normalized `(text,label)` pairs."""

    counter: Counter[str] = Counter()
    for rec in doc.records:
        text = normalize_text(rec.text)
        if text:
            counter[f"{text}|{rec.label}"] += 1
    return counter


def _text_label_neg_counter(doc: CanonicalDocument) -> Counter[str]:
    """Count normalized `(text,label,negation)` tuples."""

    counter: Counter[str] = Counter()
    for rec in doc.records:
        text = normalize_text(rec.text)
        if not text:
            continue
        neg = str(rec.attributes.get("negation", "null"))
        counter[f"{text}|{rec.label}|{neg}"] += 1
    return counter


def compute_infectio_metrics(
    *,
    predictions: list[CanonicalDocument],
    references: list[CanonicalDocument],
    robustness: dict[str, float],
) -> ScoreComputation:
    """Compute infectiology metrics.

    The official metric is micro F1 over `(text, label, negation)` tuples.

    Args:
        predictions: Predicted canonical documents.
        references: Gold canonical documents.
        robustness: Robustness metrics computed during inference.

    Returns:
        Full score computation for the infectio task.
    """

    text_counts: list[DocCounts] = []
    text_label_counts: list[DocCounts] = []
    official_counts: list[DocCounts] = []

    for pred, gold in zip(predictions, references, strict=True):
        text_counts.append(counter_tp_fp_fn(_text_counter(pred), _text_counter(gold)))
        text_label_counts.append(counter_tp_fp_fn(_text_label_counter(pred), _text_label_counter(gold)))
        official_counts.append(counter_tp_fp_fn(_text_label_neg_counter(pred), _text_label_neg_counter(gold)))

    official_agg = aggregate_doc_counts(official_counts)
    text_agg = aggregate_doc_counts(text_counts)
    text_label_agg = aggregate_doc_counts(text_label_counts)

    metrics = TaskMetrics(
        task=TaskId.INFECTIO,
        official=micro_from_counts(official_agg.tp, official_agg.fp, official_agg.fn),
        official_name="micro_f1_text_label_negation",
        secondary={
            "micro_f1_text": micro_from_counts(text_agg.tp, text_agg.fp, text_agg.fn),
            "micro_f1_text_label": micro_from_counts(
                text_label_agg.tp,
                text_label_agg.fp,
                text_label_agg.fn,
            ),
        },
        robustness=robustness,
    )
    return ScoreComputation(metrics=metrics, official_doc_counts=official_counts)
