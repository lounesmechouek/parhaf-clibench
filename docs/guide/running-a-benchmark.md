# Running a benchmark

The `parhaf-clinbench` CLI is a thin dispatcher over a small set of
subcommands. Every subcommand is safe to rerun and writes to a run
directory scoped by suite and timestamp.

Source: [`parhaf_clinbench.cli.main`](../reference/parhaf_clinbench/cli/main.md).

## Command overview

| Command             | Purpose                                                              |
|---------------------|----------------------------------------------------------------------|
| `run`               | Execute a full campaign for a suite.                                 |
| `smoke`             | Execute the small smoke suite as a health check.                     |
| `score`             | Recompute scores offline from a predictions and gold pair.           |
| `report`            | Print the path to the markdown report for a run.                     |
| `prefetch`          | Download a single model to the local cache.                          |
| `prefetch-suite`    | Download every model and dataset referenced by a suite.              |
| `audit-contracts`   | Check that every dataset matches the declared schema contract.       |

## `run`

```bash
uv run parhaf-clinbench run \
  --suite configs/suites/v1_full.yaml \
  --task all \
  --track all \
  --model all \
  --output-dir results/v1_full
```

| Flag            | Default                         | Purpose                                               |
|-----------------|---------------------------------|-------------------------------------------------------|
| `--suite`       | `parhaf_suite` setting          | Path to the suite YAML.                                |
| `--task`        | `all`                           | `pseudo`, `infectio`, `response`, `scenario`, or `all`.|
| `--track`       | `all`                           | `zeroshot`, `fewshot`, or `all`.                       |
| `--model`       | `all`                           | A model ID from the suite, or `all`.                   |
| `--output-dir`  | `parhaf_output_dir` setting     | Where to write run directories.                        |

The command prints the path of every run directory it created, one
per line, which makes it easy to pipe into a follow-up step.

## `smoke`

Runs `configs/suites/v1_smoke.yaml` by default. The smoke suite is
described on the [Smoke test page](../getting-started/smoke.md).

## `score`

Recomputes the official metric for a single `(task, predictions, gold)`
tuple without touching the runtime:

```bash
uv run parhaf-clinbench score \
  --predictions results/v1_full/<run_id>/predictions.jsonl \
  --gold results/v1_full/<run_id>/gold.jsonl \
  --task pseudo
```

The output is a JSON blob with the official metric, the bootstrap
interval, and the count of parsed records. This is the building
block of the [scoring audit](../llmops/scoring-audit.md).

## `report`

Prints the path to the markdown report in a run directory, failing if
the report is missing:

```bash
uv run parhaf-clinbench report --run-dir results/v1_full/<run_id>
```

## `prefetch` and `prefetch-suite`

`prefetch` downloads a single model (by model ID or by raw HF
identifier) and writes a JSON record with its local path, revision,
and size. `prefetch-suite` does the same for every model and every
dataset referenced by a suite.

```bash
uv run parhaf-clinbench prefetch --model qwen25_7b
uv run parhaf-clinbench prefetch-suite --suite configs/suites/v1_full.yaml
```

Run these once on a machine with network access. After that, the
benchmark can run offline against the cached content indefinitely.

## `audit-contracts`

Walks the suite, hits HuggingFace, and checks that the declared
label sets and schema fields in `configs/contracts/` match the
actual dataset at the pinned revision. Use it whenever you change a
`dataset_revision` to catch schema drift before a campaign burns GPU
hours.

```bash
uv run parhaf-clinbench audit-contracts \
  --suite configs/suites/v1_full.yaml \
  --output-json results/audit/contracts.json
```

Exit code is `0` when every dataset matches, `1` otherwise. Pass
`--allow-mismatch` to keep the exit code at `0` for reporting-only
invocations.

## Run directory layout

Each `(model, task, track)` cell gets its own directory inside the
run root:

```
results/<run_id>/
  manifest.json
  gold.jsonl
  predictions.jsonl
  scores.json
  robustness.json
  latency.json
  <model>/<task>/<track>/
    predictions.jsonl
    scores.json
    robustness.json
    timing.json
  artifacts/
    scores.parquet
    robustness.parquet
    timings.parquet
    subgroups.parquet
    error_taxonomy.parquet
    vs_baseline.parquet
  figures/
  report.md
```

The parquet artifacts under `artifacts/` are what the Streamlit UI
and the figure builder consume. They are derived from the per-cell
files and can be regenerated at any time from raw predictions.

See [Reading the results](reading-results.md) for a guided tour of
each file.
