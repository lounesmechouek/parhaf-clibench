# Run manifests

Every run writes a `manifest.json` at the root of the run directory.
The manifest is the single source of truth for "what exactly ran"
and is the first file to diff when two runs produce different
numbers.

## What is inside

```json title="manifest.json"
{
  "run_id": "v1_full__20250410T120145Z",
  "suite_id": "v1_full",
  "benchmark_version": "v1",
  "started_at": "2025-04-10T12:01:45Z",
  "finished_at": "2025-04-10T18:44:12Z",
  "code": {
    "git_sha": "ca0a879",
    "git_dirty": false,
    "package_version": "0.1.0",
    "python_version": "3.11.10"
  },
  "hardware": {
    "gpu": "NVIDIA RTX A6000",
    "cuda": "12.4",
    "driver": "550.90.07"
  },
  "parameters": {
    "temperature": 0.0,
    "top_p": 1.0,
    "max_tokens": 512,
    "seed": 42
  },
  "runtimes": {
    "vllm": {
      "version": "0.8.5",
      "config_sha256": "8d2c..."
    },
    "gliner": {
      "version": "0.3.1",
      "config_sha256": "3f11..."
    }
  },
  "models": [
    {
      "model_id": "qwen25_7b",
      "hf_id": "Qwen/Qwen2.5-7B-Instruct",
      "revision": "a09a35458c702b33eeacc393d103063234e8bc28",
      "tokenizer_revision": "a09a35458c702b33eeacc393d103063234e8bc28",
      "runtime": "vllm"
    }
  ],
  "datasets": [
    {
      "task": "pseudo",
      "dataset": "HealthDataHub/PARHAF-pseudo-annotated",
      "revision": "4d866f075a4d91c4e5ce0058feedd6da1d8e879a",
      "fingerprint_sha256": "1a7d..."
    }
  ],
  "prompt_hashes": {
    "pseudo.zero-shot":   "f9a1...",
    "pseudo.few-shot":    "c1e4...",
    "infectio.zero-shot": "ab20..."
  },
  "bootstrap": {
    "repetitions": 1000,
    "seed": 42,
    "confidence": 0.95
  },
  "scoring_audit": {
    "enabled": true,
    "cells_total": 52,
    "cells_matching": 52,
    "tolerance": 1e-6
  }
}
```

## Field reference

| Field                  | Meaning                                                                       |
|------------------------|-------------------------------------------------------------------------------|
| `run_id`               | Unique identifier built from suite ID and UTC start timestamp.                |
| `suite_id`             | The `suite_id` from the suite YAML.                                           |
| `benchmark_version`    | The `benchmark_version` from the suite YAML.                                  |
| `code.git_sha`         | Short Git SHA of the working tree at run start.                               |
| `code.git_dirty`       | `true` if the working tree had uncommitted changes.                           |
| `code.package_version` | `pyproject.toml` version.                                                     |
| `hardware`             | GPU model, CUDA version, driver version.                                      |
| `parameters`           | Inference parameters from the suite YAML.                                     |
| `runtimes`             | Runtime versions and config hashes.                                           |
| `models`               | Resolved list with pinned revisions and runtime assignment.                   |
| `datasets`             | Resolved list with pinned revisions and content fingerprints.                 |
| `prompt_hashes`        | SHA-256 per `(task, track)` of the rendered template plus few-shot bank.      |
| `bootstrap`            | Bootstrap configuration used during scoring.                                  |
| `scoring_audit`        | Result of the optional in-run rescoring pass.                                 |

## Why this shape

The manifest is flat by design. Every field is trivial to diff with
`jq` or a notebook. Any field that is computed rather than copied
from a config is hashed, so a silent upstream change cannot hide
inside a nested blob.

## Using the manifest

- **Diff two runs** to find the first field that moved. Most
  mysterious score drifts are a prompt hash or a package version
  difference.
- **Replay a run** by checking out the code at `code.git_sha`,
  prefetching the exact model and dataset revisions, and pointing
  the CLI at the same suite.
- **Prove provenance** by keeping the manifest next to the figures
  in a report. A reviewer can trace every number back to a pin.

See [Versioning and hashing](versioning.md) for what each hash
covers and
[`parhaf_clinbench.orchestration.runner`](../reference/parhaf_clinbench/orchestration/runner.md)
for the code that writes the manifest.
