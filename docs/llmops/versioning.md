# Versioning and hashing

The package versions every object that can affect a benchmark
number. This page enumerates them, in the order a reader is likely
to ask about them.

## Summary

| Object                  | Where it is pinned                                     | How it is hashed                                               |
|-------------------------|--------------------------------------------------------|----------------------------------------------------------------|
| Model weights           | `configs/models/<id>.yaml` (`revision`)                | HuggingFace commit SHA is the pin; no separate hash needed.    |
| Tokenizer               | `configs/models/<id>.yaml` (`tokenizer_revision`)      | HuggingFace commit SHA.                                        |
| Dataset                 | `configs/tasks/<id>.yaml` (`dataset_revision`)         | HuggingFace commit SHA plus a local content fingerprint.       |
| Dataset content         | Computed at load time                                  | SHA-256 over name, example count, document IDs, text digests.  |
| Prompt template         | `prompts/<task>/<track>.jinja2`                        | SHA-256 over the rendered template with the canonical context. |
| Few-shot bank           | `assets/fewshot/<task>_examples.txt`                   | Hashed alongside the prompt template.                          |
| Runtime config          | `configs/runtimes/<id>.yaml`                           | SHA-256 over the YAML, included in the manifest.               |
| Inference parameters    | Suite `parameters` block                               | Serialized into the manifest.                                  |
| Code                    | Git repository                                         | Git short SHA plus dirty flag, captured at run time.           |
| Package version         | `pyproject.toml`                                       | Read at run time into the manifest.                            |
| Python runtime          | Host environment                                       | `sys.version` written into the manifest.                       |

The central hashing helpers live in
[`parhaf_clinbench.core.hashing`](../reference/parhaf_clinbench/core/hashing.md)
and are reusable from any extension.

## Model and tokenizer

Every model file pins two SHAs:

```yaml title="configs/models/qwen25_7b.yaml"
model_id: qwen25_7b
hf_id: Qwen/Qwen2.5-7B-Instruct
revision: a09a35458c702b33eeacc393d103063234e8bc28
tokenizer_revision: a09a35458c702b33eeacc393d103063234e8bc28
family: llm
max_context_tokens: 32768
```

The loader refuses to start without both revisions. This is
deliberate: HuggingFace repositories do not version tokenizers and
weights together, and a tokenizer upgrade can silently invalidate a
previous score.

## Dataset

Datasets are pinned the same way in `configs/tasks/<id>.yaml`
through `dataset_revision`. Source:
[`parhaf_clinbench.data.hf_loaders`](../reference/parhaf_clinbench/data/hf_loaders.md).

On top of the HuggingFace SHA, the package computes a content
fingerprint on the loaded subset:

- dataset name,
- number of examples,
- ordered document IDs,
- SHA-256 digest of each document text.

The fingerprint is written into the run manifest. Two runs with the
same fingerprint have the same inputs byte for byte. If HuggingFace
re-uploads the dataset under the same SHA (rare but possible), the
fingerprint changes and the run is marked as different.

The loader prefers a local cache:
`DownloadConfig(local_files_only=True)`. A warm cache produces a
bit-identical load whether the host is online or not. The prefetch
commands populate that cache.

## Prompts and few-shot bank

Every template under `prompts/<task>/<track>.jinja2` is rendered
once at campaign start with the canonical dynamic context produced
by
[`parhaf_clinbench.prompting.contracts`](../reference/parhaf_clinbench/prompting/contracts.md).
The rendered text is hashed with SHA-256.

The few-shot bank for a track goes through the same path. Both
hashes end up in the `prompt_hashes` map in the manifest, keyed by
`(task, track)`.

If you edit a template and the hash does not change, the change is
pure whitespace or a comment. If the hash changes, the run is
genuinely different from the previous one, and figures should not
be compared across the two.

## Runtime configuration

Runtime YAML files are serialized into the manifest with a hash. A
change to `guided_decoding_backend`, `max_num_seqs`, or
`gpu_memory_utilization` is visible in the diff.

## Code

The orchestrator captures:

- Git short SHA of the working tree, with a dirty flag if the tree
  has uncommitted changes.
- Package version from `pyproject.toml`.
- Python version from `sys.version`.

A run from a dirty working tree is not reproducible by definition,
and the manifest says so.

## Where to look

- Helpers: `src/parhaf_clinbench/core/hashing.py`.
- Manifest writer: `src/parhaf_clinbench/orchestration/runner.py`.
- Data fingerprint: `src/parhaf_clinbench/data/manifests.py`.

## A quick diagnostic

When two runs disagree on an F1 number, diff the two manifests. The
first field that differs is the one that moved. Nine times out of
ten this is a prompt hash, a dataset fingerprint, or a package
version.
