# Determinism

Reproducibility in LLM benchmarks is a layered property. This page
documents which layers the package locks down, which ones it cannot
control, and how to tell whether two runs are supposed to match.

## What is deterministic

### The orchestrator

The orchestrator has no non-determinism of its own. It iterates over
cells in a stable order, reads pinned configs, and writes artifacts
with deterministic file names. Two runs with identical inputs
produce identical run directories modulo timestamps.

### Scoring and bootstrap

Scoring is a pure function of predictions and gold. Bootstrap
resampling uses a fixed seed (`seed = 42`) and a fixed number of
replications (`B = 1000`). See
[`parhaf_clinbench.scoring.bootstrap`](../reference/parhaf_clinbench/scoring/bootstrap.md).

### GLiNER2

GLiNER2 is deterministic by construction. There is no sampling, no
guided decoding, no randomness in the forward pass. Given the same
weights and the same inputs, it produces the same records.

### vLLM under greedy decoding

With `temperature = 0`, `top_p = 1`, and a fixed seed, vLLM is
deterministic within the following scope:

- same vLLM version,
- same CUDA and driver version,
- same GPU architecture,
- same xgrammar schema (tied to the prompt hash),
- same model weights (tied to the pinned revision).

Two runs that share all five of these can be compared byte for byte.

## What is not deterministic across

| Axis                        | Effect                                                          | How to control                                                    |
|-----------------------------|-----------------------------------------------------------------|-------------------------------------------------------------------|
| vLLM minor version          | Generated tokens can shift because of kernel or scheduler updates. | Pin the vLLM version in the Docker image.                         |
| CUDA driver                 | Floating-point reduction order can change.                      | Pin the driver on the host or use a fixed-tag Docker image.        |
| GPU architecture            | Different kernels are selected per architecture.                | Run on the same class of GPU.                                      |
| Multi-GPU batching          | Batch composition can affect sampling with non-greedy decoding. | Keep `temperature = 0` or run single-GPU for comparisons.          |
| HuggingFace re-uploads      | Same SHA, different content (rare but possible).                | Cache the dataset locally and trust the content fingerprint.       |

The general rule is that the Docker image is the cleanest
reproducibility anchor. Two runs executed with the same image tag on
the same GPU class are as reproducible as LLM inference gets today.

## What about sampling

Nothing in the package forbids sampling. If you set `temperature > 0`,
bootstrap intervals still cover document-level variance but no
longer cover sampling variance. A proper sensitivity study over
temperature or decoding strategies is the responsibility of the
suite author and is not wired into the built-in scoring.

## Checking determinism

A fast self-check:

1. Run the smoke suite twice with the mock runtime. Manifests should
   agree on everything except timestamps.
2. Run the smoke suite twice with vLLM under greedy decoding. The
   `predictions.jsonl` files should be byte-identical.

A slower, more rigorous check is the [scoring audit](scoring-audit.md),
which verifies that the shipped numbers in a run are reproducible to
six decimal places from the predictions alone.
