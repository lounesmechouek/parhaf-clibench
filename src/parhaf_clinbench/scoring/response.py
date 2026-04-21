"""Scoring logic for treatment-response extraction."""

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


def _document_label_counter(doc: CanonicalDocument) -> Counter[str]:
    """Count one document-level label token used for secondary metrics."""

    labels = {record.label for record in doc.records}
    if not labels:
        return Counter({"null": 1})
    if len(labels) == 1:
        return Counter({next(iter(labels)): 1})
    return Counter({"__multiple_labels__": 1})


def compute_response_metrics(
    *,
    predictions: list[CanonicalDocument],
    references: list[CanonicalDocument],
    robustness: dict[str, float],
) -> ScoreComputation:
    """Compute response metrics.

    The official metric is micro F1 over `(text, label)` tuples.

    Args:
        predictions: Predicted canonical documents.
        references: Gold canonical documents.
        robustness: Robustness metrics computed during inference.

    Returns:
        Full score computation for the response task.
    """

    official_counts: list[DocCounts] = []
    label_counts: list[DocCounts] = []

    for pred, gold in zip(predictions, references, strict=True):
        official_counts.append(counter_tp_fp_fn(_text_label_counter(pred), _text_label_counter(gold)))
        label_counts.append(counter_tp_fp_fn(_document_label_counter(pred), _document_label_counter(gold)))

    official_agg = aggregate_doc_counts(official_counts)
    label_agg = aggregate_doc_counts(label_counts)

    metrics = TaskMetrics(
        task=TaskId.RESPONSE,
        official=micro_from_counts(official_agg.tp, official_agg.fp, official_agg.fn),
        official_name="micro_f1_text_label",
        secondary={
            "micro_f1_label": micro_from_counts(label_agg.tp, label_agg.fp, label_agg.fn),
        },
        robustness=robustness,
    )
    return ScoreComputation(metrics=metrics, official_doc_counts=official_counts)
