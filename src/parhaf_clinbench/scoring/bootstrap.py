"""Document-level non-parametric bootstrap utilities."""

from __future__ import annotations

import random
from statistics import mean

from parhaf_clinbench.core.models import BootstrapInterval
from parhaf_clinbench.scoring.common import DocCounts


def _f1(tp: int, fp: int, fn: int) -> float:
    """Compute micro F1 from TP/FP/FN counts."""

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0


def bootstrap_official_score(
    *,
    doc_counts: list[DocCounts],
    repetitions: int = 1000,
    seed: int = 42,
) -> BootstrapInterval:
    """Bootstrap an official score with percentile 95% confidence interval.

    Args:
        doc_counts: Per-document TP/FP/FN counts.
        repetitions: Number of bootstrap samples.
        seed: Random seed.

    Returns:
        Bootstrap interval with full-score baseline and confidence bounds.
    """

    if not doc_counts:
        return BootstrapInterval(score_full=0.0, ci_low=0.0, ci_high=0.0, repetitions=repetitions)

    tp_full = sum(item.tp for item in doc_counts)
    fp_full = sum(item.fp for item in doc_counts)
    fn_full = sum(item.fn for item in doc_counts)
    full = _f1(tp_full, fp_full, fn_full)

    rng = random.Random(seed)
    n_docs = len(doc_counts)
    samples: list[float] = []
    for _ in range(repetitions):
        tp = 0
        fp = 0
        fn = 0
        for _ in range(n_docs):
            pick = doc_counts[rng.randrange(0, n_docs)]
            tp += pick.tp
            fp += pick.fp
            fn += pick.fn
        samples.append(_f1(tp, fp, fn))

    samples.sort()
    low_index = round(0.025 * (repetitions - 1))
    high_index = round(0.975 * (repetitions - 1))
    return BootstrapInterval(
        score_full=full,
        ci_low=samples[low_index],
        ci_high=samples[high_index],
        repetitions=repetitions,
    )


def bootstrap_global_score(
    *,
    per_task_doc_counts: dict[str, list[DocCounts]],
    repetitions: int = 1000,
    seed: int = 42,
) -> BootstrapInterval:
    """Bootstrap global track score (mean of task bootstrap scores).

    Args:
        per_task_doc_counts: Mapping from task id to per-document counts.
        repetitions: Number of bootstrap samples.
        seed: Random seed.

    Returns:
        Bootstrap interval for global score.
    """

    if not per_task_doc_counts:
        return BootstrapInterval(score_full=0.0, ci_low=0.0, ci_high=0.0, repetitions=repetitions)

    task_names = sorted(per_task_doc_counts)
    full_scores = [
        _f1(
            sum(item.tp for item in per_task_doc_counts[task]),
            sum(item.fp for item in per_task_doc_counts[task]),
            sum(item.fn for item in per_task_doc_counts[task]),
        )
        for task in task_names
    ]
    score_full = mean(full_scores) if full_scores else 0.0

    rng = random.Random(seed)
    samples: list[float] = []
    for _ in range(repetitions):
        task_scores: list[float] = []
        for task in task_names:
            counts = per_task_doc_counts[task]
            if not counts:
                task_scores.append(0.0)
                continue
            n_docs = len(counts)
            tp = 0
            fp = 0
            fn = 0
            for _ in range(n_docs):
                pick = counts[rng.randrange(0, n_docs)]
                tp += pick.tp
                fp += pick.fp
                fn += pick.fn
            task_scores.append(_f1(tp, fp, fn))
        samples.append(mean(task_scores) if task_scores else 0.0)

    samples.sort()
    low_index = round(0.025 * (repetitions - 1))
    high_index = round(0.975 * (repetitions - 1))
    return BootstrapInterval(
        score_full=score_full,
        ci_low=samples[low_index],
        ci_high=samples[high_index],
        repetitions=repetitions,
    )


def bootstrap_paired_delta(
    *,
    model_a: list[DocCounts],
    model_b: list[DocCounts],
    repetitions: int = 1000,
    seed: int = 42,
) -> BootstrapInterval:
    """Bootstrap paired F1 delta between two models.

    Args:
        model_a: Per-document counts for model A.
        model_b: Per-document counts for model B.
        repetitions: Number of bootstrap samples.
        seed: Random seed.

    Returns:
        Bootstrap interval for `(score(model_a) - score(model_b))`.
    """

    if not model_a or not model_b or len(model_a) != len(model_b):
        return BootstrapInterval(score_full=0.0, ci_low=0.0, ci_high=0.0, repetitions=repetitions)

    def score(items: list[DocCounts]) -> float:
        return _f1(
            sum(x.tp for x in items),
            sum(x.fp for x in items),
            sum(x.fn for x in items),
        )

    full = score(model_a) - score(model_b)
    rng = random.Random(seed)
    n_docs = len(model_a)
    deltas: list[float] = []

    for _ in range(repetitions):
        sample_a: list[DocCounts] = []
        sample_b: list[DocCounts] = []
        for _ in range(n_docs):
            idx = rng.randrange(0, n_docs)
            sample_a.append(model_a[idx])
            sample_b.append(model_b[idx])
        deltas.append(score(sample_a) - score(sample_b))

    deltas.sort()
    low_index = round(0.025 * (repetitions - 1))
    high_index = round(0.975 * (repetitions - 1))
    return BootstrapInterval(
        score_full=full,
        ci_low=deltas[low_index],
        ci_high=deltas[high_index],
        repetitions=repetitions,
    )
