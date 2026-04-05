"""Utilities for micro-aggregated scoring."""

from __future__ import annotations

from collections import Counter

from pydantic import BaseModel, ConfigDict

from parhaf_clinbench.core.models import ScoreTriplet, TaskMetrics


class ScoreModel(BaseModel):
    """Local strict Pydantic base model for scoring structures."""

    model_config = ConfigDict(extra="forbid")


class DocCounts(ScoreModel):
    """Per-document TP/FP/FN counts."""

    tp: int
    fp: int
    fn: int


class ScoreComputation(ScoreModel):
    """Complete score computation bundle for a task."""

    metrics: TaskMetrics
    official_doc_counts: list[DocCounts]



def micro_from_counts(tp: int, fp: int, fn: int) -> ScoreTriplet:
    """Compute micro precision/recall/F1 from aggregate counts.

    Args:
        tp: True positives.
        fp: False positives.
        fn: False negatives.

    Returns:
        Micro-averaged precision/recall/F1 triplet.
    """

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return ScoreTriplet(precision=precision, recall=recall, f1=f1)



def counter_tp_fp_fn(pred: Counter[str], gold: Counter[str]) -> DocCounts:
    """Compute TP/FP/FN between two multisets represented as `Counter`.

    Args:
        pred: Predicted multiset.
        gold: Reference multiset.

    Returns:
        Per-document count structure.
    """

    tp = sum((pred & gold).values())
    fp = sum((pred - gold).values())
    fn = sum((gold - pred).values())
    return DocCounts(tp=tp, fp=fp, fn=fn)



def aggregate_doc_counts(values: list[DocCounts]) -> DocCounts:
    """Aggregate per-document counts at corpus level.

    Args:
        values: Per-document TP/FP/FN counts.

    Returns:
        Corpus-level TP/FP/FN totals.
    """

    return DocCounts(
        tp=sum(item.tp for item in values),
        fp=sum(item.fp for item in values),
        fn=sum(item.fn for item in values),
    )
