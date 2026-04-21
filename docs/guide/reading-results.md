# Reading the results

A run directory is self-describing. This page names every file it
contains, in the order you are most likely to read them.

## `manifest.json`

The single source of truth for the run. Contains every hash, pin,
and parameter needed to reproduce the numbers later. See
[Run manifests](../llmops/manifests.md) for the schema.

## `predictions.jsonl`

One JSON object per document, with the canonical record shape defined
by
[`parhaf_clinbench.core.models.CanonicalDocument`](../reference/parhaf_clinbench/core/models.md).
This is the only file the scorer reads, which makes it easy to
rescore with a different metric or to diff two runs offline.

## `gold.jsonl`

The gold records for the same document set, in the same canonical
shape. Frozen copy of what the loader resolved against the pinned
dataset revision. Keep it alongside predictions if you ever plan to
rescore the cell from a machine without network access.

## `scores.json`

The official metric for the cell with its bootstrap interval. A
small example for the `pseudo` task:

```json
{
  "task": "pseudo",
  "official_metric": "span_micro_f1",
  "score": 0.468,
  "ci_low": 0.431,
  "ci_high": 0.504,
  "repetitions": 1000,
  "n_documents": 509
}
```

## `robustness.json`

Schema conformity, empty rate, JSON validity, and a breakdown of
document-level errors by category. See
[Robustness metrics](../concepts/robustness.md).

## `timing.json`

Per-document latency distribution and throughput. Populated by the
runner from the runtime timings.

## Aggregated parquet tables

At the suite level, the orchestrator aggregates the per-cell JSON
files into a set of parquet tables under `artifacts/`. These are the
tables the Streamlit UI loads:

| File                          | Description                                              |
|-------------------------------|----------------------------------------------------------|
| `scores.parquet`              | Per-cell F1 with bootstrap intervals.                    |
| `scores_global.parquet`       | Per-model global F1 (equal-weight mean across tasks).    |
| `robustness.parquet`          | Schema conformity, empty rate, JSON validity.            |
| `timings.parquet`             | Per-cell latency and throughput.                          |
| `subgroups.parquet`           | F1 by document length, negation polarity, specialty.     |
| `error_taxonomy.parquet`      | Counts per error category.                                |
| `vs_baseline.parquet`         | Paired deltas against a chosen baseline model.           |

## Figures

After a run completes, regenerate the figures from the parquet tables:

```bash
uv run python analysis/build_figures.py --run-dir results/<run_id>
```

Figures land in `<run_dir>/figures/` as PNG files and are mirrored
into the Streamlit UI asset path so the UI has a fallback when
Plotly rendering fails.

## The markdown report

A human-readable summary of the run is written to `report.md` by
[`parhaf_clinbench.reporting.markdown`](../reference/parhaf_clinbench/reporting/markdown.md).
It is a convenience file and not part of the canonical artifact set.
Treat it as a draft you can edit before sharing. Canonical numbers
live in `scores.json` and the parquet tables.
