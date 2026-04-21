# Smoke test

The smoke suite is a full end-to-end walkthrough of the benchmark
that never loads a large model. It uses a tiny local dataset under
`data/` and, by default, the `qwen25_7b` model configuration served
by vLLM. You can also swap it for the `mock` runtime to run the
pipeline without a GPU.

## Why it exists

- It verifies that the code path from config loading through scoring
  and report writing stays healthy. CI runs it on every pull request.
- It produces a real run directory you can inspect to understand the
  on-disk layout before launching a full campaign.
- It is the fastest way to confirm a new model YAML or a new prompt
  template parses correctly.

## Running the smoke suite

```bash
make smoke
```

This resolves to:

```bash
uv run parhaf-clinbench smoke \
  --suite configs/suites/v1_smoke.yaml \
  --output-dir results/smoke
```

## What to expect

- Duration: a few minutes on the mock runtime, a few more on a GPU.
- Output: a run directory under `results/smoke/` with the same
  structure as a full run.
- Exit code: `0` on success. CI asserts on this.

## Suite contents

```yaml title="configs/suites/v1_smoke.yaml"
suite_id: v1_smoke
benchmark_version: v1
tracks: [zero-shot]
tasks: [pseudo, infectio, response, scenario]
models: [qwen25_7b]
runtime_overrides:
  gliner2_multi: gliner
runtime_default: vllm
parameters:
  temperature: 0.0
  top_p: 1.0
  max_tokens: 256
  seed: 42
```

The smoke suite keeps one track and one model to keep its runtime
bounded, but exercises every task so that any schema change surfaces
immediately.

## Using the mock runtime

When you cannot or do not want to hit a GPU, point the runtime
override at `mock`:

```yaml title="configs/suites/v1_smoke_mock.yaml"
suite_id: v1_smoke_mock
benchmark_version: v1
tracks: [zero-shot]
tasks: [pseudo, infectio, response, scenario]
models: [qwen25_7b]
runtime_overrides:
  qwen25_7b: mock
runtime_default: mock
parameters:
  temperature: 0.0
  top_p: 1.0
  max_tokens: 256
  seed: 42
```

The mock runtime returns a fixed, schema-valid response for every
document. Numbers are meaningless, but the pipeline plumbing is
exercised exactly the same way. This is how CI runs the benchmark
without a GPU.
