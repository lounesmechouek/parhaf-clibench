# Artifacts and storage

A run directory is a self-contained unit. It can be moved between
machines, archived to object storage, or attached to a pull request.
Every downstream tool (Streamlit UI, monitoring replay, scoring
audit) reads it from disk without assuming any external state.

## Directory layout

```
results/<run_id>/
  manifest.json
  gold/
    pseudo.jsonl
    infectio.jsonl
    response.jsonl
    scenario.jsonl
  predictions.jsonl
  scores.json
  robustness.json
  timing.json
  <model>/<task>/<track>/
    predictions.jsonl
    scores.json
    robustness.json
    timing.json
    errors.jsonl
  artifacts/
    scores.parquet
    scores_global.parquet
    robustness.parquet
    timings.parquet
    subgroups.parquet
    error_taxonomy.parquet
    vs_baseline.parquet
  figures/
    *.png
  logs/
    runner.log
    events.jsonl
  report.md
```

## File roles

| Path                      | Produced by                                      | Consumed by                              |
|---------------------------|--------------------------------------------------|------------------------------------------|
| `manifest.json`           | Orchestrator                                     | Audit, diff, reproducibility claims.     |
| `gold/<task>.jsonl`       | Dataset loader                                   | Scorer, audit.                           |
| `<cell>/predictions.jsonl`| Runtime plus parser                              | Scorer, UI, audit.                       |
| `<cell>/scores.json`      | Scorer                                           | UI, report generator.                    |
| `<cell>/robustness.json`  | Parser plus scorer                               | UI, monitoring replay.                   |
| `<cell>/timing.json`      | Runner                                           | UI, latency figures.                     |
| `<cell>/errors.jsonl`     | Error-taxonomy pass                              | UI error explorer.                       |
| `artifacts/*.parquet`     | Aggregator                                       | UI, `analysis/build_figures.py`.         |
| `figures/*.png`           | `analysis/build_figures.py`                      | Report, Streamlit fallback.              |
| `logs/runner.log`         | Runner                                           | Debugging.                               |
| `logs/events.jsonl`       | Runner                                           | Monitoring replay.                       |
| `report.md`               | Reporting layer                                  | Human readers.                           |

## Size expectations

- One full run over the `v1_full` suite: roughly 2 GB, dominated by
  parquet tables and per-cell JSONL predictions.
- One smoke run: tens of megabytes.
- A long archive of runs: budget for ~2 GB per full run and keep
  the model cache separate (tens of GB per model).

## Retention

The artifact tree is additive. The orchestrator never overwrites a
previous run directory. Deletion is a manual decision. A reasonable
retention policy for a research team is:

- Keep every run referenced by a published report or blog post
  indefinitely.
- Keep runs from the current and previous development cycle on local
  disk.
- Move older runs to cold object storage.

## Moving a run directory

A run directory is safe to `tar`, move to another machine, and
extract. Two caveats:

- The dataset cache is outside the run directory. Replaying the
  scoring audit or rerunning the figure builder on the new machine
  requires the same cached dataset revisions.
- Absolute paths are never written inside the run. Everything under
  `results/<run_id>/` is relative.

## Cleaning up

There is no built-in "clean" command. Deletion is a `rm -rf` on the
run directory. The package treats run directories as immutable
artifacts, not as scratch space.
