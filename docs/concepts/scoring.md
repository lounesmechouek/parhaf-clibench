# Scoring and bootstrap

Scoring is deterministic given a predictions file and a gold file.
The scorer reads canonical records, computes task-specific
true-positive, false-positive, and false-negative counts, and
produces micro-F1 per task plus a global equal-weight average across
tasks.

## Per-task metrics

| Task       | Elementary unit                                          | Source                                                                   |
|------------|----------------------------------------------------------|--------------------------------------------------------------------------|
| `pseudo`   | `(start, end)` byte pair                                 | [`parhaf_clinbench.scoring.pseudo`](../reference/parhaf_clinbench/scoring/pseudo.md)         |
| `infectio` | `(text, label, negation)` triple                         | [`parhaf_clinbench.scoring.infectio`](../reference/parhaf_clinbench/scoring/infectio.md)     |
| `response` | `(text, label)` pair                                     | [`parhaf_clinbench.scoring.response`](../reference/parhaf_clinbench/scoring/response.md)     |
| `scenario` | `(text, label)` pair                                     | [`parhaf_clinbench.scoring.scenario`](../reference/parhaf_clinbench/scoring/scenario.md)     |

The aggregator in
[`parhaf_clinbench.scoring.aggregate`](../reference/parhaf_clinbench/scoring/aggregate.md)
combines the per-task scores into a global score. The global score is
the arithmetic mean of the four task F1 values. This equal weighting
is deliberate: without it, the `scenario` task (one order of
magnitude more documents) would dominate the aggregate.

## Bootstrap confidence intervals

Source:
[`parhaf_clinbench.scoring.bootstrap`](../reference/parhaf_clinbench/scoring/bootstrap.md).

For every `(model, track, task)` cell, the package reports a 95%
percentile confidence interval computed by non-parametric
document-level bootstrap:

- `B = 1000` replications.
- Fixed seed (`seed = 42`), set once for reproducibility.
- Percentile interval using the 2.5th and 97.5th percentile of the
  replicate F1 distribution.

Three bootstrap utilities are exposed:

- `bootstrap_official_score`: intervals for a single cell.
- `bootstrap_global_score`: intervals for the equal-weight mean
  across tasks.
- `bootstrap_paired_delta`: intervals for `(model_a - model_b)`
  deltas, used for head-to-head comparisons in the UI and the study.

The paired version resamples the same set of documents for both
models at each replication, which cancels document-level noise and
is the right tool for significance claims.

## Offline scoring

You can score any pair of predictions and gold JSONL files without
re-running inference:

```bash
uv run parhaf-clinbench score \
  --predictions results/<run_id>/predictions.jsonl \
  --gold results/<run_id>/gold.jsonl \
  --task pseudo
```

The CLI prints a JSON summary with the official metric, the
bootstrap interval, and the count of parsed records. This command is
the backbone of the scoring audit described in the
[LLMOps section](../llmops/scoring-audit.md).

## Extending the scoring logic

If you need a new metric on an existing task, the cleanest path is to
read `predictions.jsonl` and `gold.jsonl` directly in a notebook and
compute your own number. The official metric is locked by design so
that two runs can always be compared apples to apples.

For a brand-new task, add a new scorer module and register it in
[`parhaf_clinbench.scoring.common`](../reference/parhaf_clinbench/scoring/common.md)
alongside the existing ones.
