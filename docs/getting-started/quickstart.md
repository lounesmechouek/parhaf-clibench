# Quickstart

This page walks through a first real benchmark run from a freshly
cloned checkout.

!!! note "Before you start"
    You need a CUDA GPU with enough memory for the model you intend
    to serve through vLLM (24 GB is comfortable for 7B to 9B models
    with bf16 weights). The Streamlit UI and the offline scoring
    commands run on any machine.

## 1. Install with vLLM support

```bash
uv sync --extra dev --extra vllm
```

The `vllm` extra pulls in `vllm`, which expects a recent CUDA driver
on the host.

## 2. Prefetch models and datasets

The benchmark is designed to run from a warm cache so that a network
failure cannot corrupt a run. Prefetch everything referenced by a
suite in one call:

```bash
export HF_TOKEN="hf_xxx"
uv run parhaf-clinbench prefetch-suite \
  --suite configs/suites/v1_full.yaml \
  --output-json results/v1_full/prefetch.json
```

The command resolves every model in the suite to its pinned revision
and every task to its pinned dataset revision, downloads them into
the configured caches, and writes a JSON record of the exact file
fingerprints it saw. That JSON is itself an audit artifact and should
be committed alongside a run if you want to replay it later.

## 3. Launch a campaign

```bash
uv run parhaf-clinbench run \
  --suite configs/suites/v1_full.yaml \
  --output-dir results/v1_full
```

A campaign materializes one directory per `(model, task, track)`
triple under the run root, each containing:

- `predictions.jsonl` with the canonical records returned by the
  model.
- `scores.json` with the official micro-F1 and the bootstrap
  confidence interval.
- `robustness.json` with schema conformity, empty rate, validity
  rate, and latency.
- `manifest.json` with every hash needed to reproduce the cell.

At the suite level you will also find the aggregated
`scores_global.parquet`, `robustness.parquet`, `timings.parquet`, and
the error taxonomy tables.

See [Running a benchmark](../guide/running-a-benchmark.md) for the
full reference of CLI flags and on-disk layout.

## 4. Build the figures

After a run completes, regenerate the report figures from the
artifacts:

```bash
uv run python analysis/build_figures.py --run-dir results/v1_full
```

Figures land in `results/v1_full/figures/` and are mirrored into the
Streamlit UI asset tree.

## 5. Explore the results

```bash
uv run --extra ui streamlit run ui/app.py
```

Point the app at your run directory from the sidebar. The UI loads
scores, robustness tables, and errors lazily from parquet, so large
runs stay responsive. See [Streamlit UI](../guide/ui.md).

## 6. Replay the scoring audit

The scoring pipeline can be re-run from raw predictions on any
machine, without re-inferencing anything. This is the default way
to audit a claim of "this run produced F1 = X on task Y".

```bash
uv run parhaf-clinbench score \
  --predictions results/v1_full/<run_id>/predictions.jsonl \
  --gold results/v1_full/<run_id>/gold.jsonl \
  --task pseudo
```

See [Scoring audit](../llmops/scoring-audit.md) for the full audit
protocol.
