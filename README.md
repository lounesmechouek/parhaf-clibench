<h1 align="center">PARHAF-CLIBENCH</h1>

<p align="center">
  <b>A reproducible benchmark for clinical information extraction with language models on PARHAF.</b>
</p>

<p align="center">
  <a href="https://github.com/lounesmechouek/parhaf-clibench/actions/workflows/ci.yml"><img alt="CI" src="https://img.shields.io/github/actions/workflow/status/lounesmechouek/parhaf-clibench/ci.yml?branch=main&label=CI&logo=github"></a>
  <a href="https://github.com/lounesmechouek/parhaf-clibench/actions/workflows/docs.yml"><img alt="Docs" src="https://img.shields.io/github/actions/workflow/status/lounesmechouek/parhaf-clibench/docs.yml?branch=main&label=docs&logo=readthedocs&logoColor=white"></a>
  <a href="https://lounesmechouek.github.io/parhaf-clibench/"><img alt="Docs site" src="https://img.shields.io/badge/docs-online-3f51b5?logo=materialformkdocs&logoColor=white"></a>
  <a href="https://github.com/lounesmechouek/parhaf-clibench/blob/main/LICENSE"><img alt="License" src="https://img.shields.io/github/license/lounesmechouek/parhaf-clibench?color=3f51b5"></a>
  <img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white">
  <a href="https://docs.astral.sh/uv/"><img alt="uv" src="https://img.shields.io/badge/package%20manager-uv-DE5FE9?logo=astral&logoColor=white"></a>
  <a href="https://docs.astral.sh/ruff/"><img alt="Ruff" src="https://img.shields.io/badge/lint-ruff-FCC21B?logo=ruff"></a>
  <a href="https://mypy-lang.org/"><img alt="mypy strict" src="https://img.shields.io/badge/mypy-strict-2A6DB2"></a>
  <a href="https://hub.docker.com/"><img alt="Docker image" src="https://img.shields.io/badge/docker-vllm-2496ED?logo=docker&logoColor=white"></a>
  <a href="https://huggingface.co/datasets/HealthDataHub/PARHAF"><img alt="PARHAF on HuggingFace" src="https://img.shields.io/badge/dataset-PARHAF-FFD21E?logo=huggingface&logoColor=black"></a>
</p>

<p align="center">
  <a href="https://lounesmechouek.github.io/parhaf-clibench/">Documentation</a>
  &nbsp;·&nbsp;
  <a href="https://lounesmechouek.github.io/parhaf-clibench/getting-started/quickstart/">Quickstart</a>
  &nbsp;·&nbsp;
  <a href="https://lounesmechouek.github.io/parhaf-clibench/guide/adding-a-model/">Add a model</a>
  &nbsp;·&nbsp;
  <a href="https://lounesmechouek.github.io/parhaf-clibench/llmops/">LLMOps</a>
  &nbsp;·&nbsp;
  <a href="https://lounesmechouek.com/writing/slm_benchmark/">Pilot Study</a>
</p>

<p align="center">
  <img src="https://lounesmechouek.github.io/parhaf-clibench/assets/streamlit.png" alt="Streamlit results UI" width="860">
</p>

---

## Table of contents

- [Why parhaf-clibench](#why-parhaf-clibench)
- [Key features](#key-features)
- [Supported models and tasks](#supported-models-and-tasks)
- [Quickstart](#quickstart)
- [Documentation](#documentation)
- [Results of the pilot study](#study-v1-results)
- [UI and monitoring](#ui-and-monitoring)
- [LLMOps and reproducibility](#llmops-and-reproducibility)
- [Deployment](#deployment)
- [Project layout](#project-layout)
- [Contributing](#contributing)
- [Citation](#citation)
- [License and acknowledgements](#license-and-acknowledgements)

## Why parhaf-clibench

Clinical information extraction is a multi-task problem over free
text written by physicians. The same discharge letter carries
patient identifiers that must be pseudonymized, infection mentions
with their negation status, a treatment outcome, and a set of
structured fields that summarize the episode. Evaluating language
models on any one of these in isolation misses the point; evaluating
them across all four at once, reproducibly, is the problem this
package solves.

`parhaf-clibench` turns a handful of YAML files into a full
evaluation campaign over the four PARHAF clinical information
extraction tasks. It runs any vLLM-compatible HuggingFace model or a
GLiNER2 encoder baseline, enforces structured output at decoding
time, scores the results with bootstrap confidence intervals,
reports robustness metrics alongside accuracy, and writes an
artifact tree that the Streamlit UI and the terminal monitoring
dashboard read back. The package and the results of the
[study v1](https://lounesmechouek.com/writing/slm_benchmark/) are
distinct artifacts. The package is a general runner. The study is
one instance of what it produces.

## Key features

- **Four PARHAF tasks** wired end to end: pseudonymization,
  infectiology, response to treatment, structured scenario.
- **Two evaluation tracks**: zero-shot and few-shot with a fixed
  demonstration bank.
- **Any vLLM-compatible HuggingFace model** runs as a one-file YAML
  addition under `configs/models/`.
- **Encoder baseline via GLiNER2** exposed through the same runtime
  interface as the LLMs.
- **Structured-output decoding** via vLLM's xgrammar backend, with
  task-specific JSON schemas including label and attribute enums.
- **Micro-F1 scoring per task** plus an equal-weight global average,
  with document-level bootstrap confidence intervals (B = 1000) and
  paired deltas for head-to-head comparisons.
- **Robustness metrics** reported alongside accuracy: schema
  conformity, empty output rate, JSON validity, median and p95
  latency, throughput.
- **Deterministic runs**: greedy decoding, fixed seed, pinned
  HuggingFace revisions for weights and datasets, SHA-256 prompt
  hashes, full run manifest.
- **Streamlit UI** with nine pages and a **Rich-based terminal
  dashboard** for live monitoring.
- **CI gate** of `ruff`, `mypy` strict, and `pytest` with 70%
  coverage, plus a nightly smoke suite on the mock runtime.
- **Docker image** published on DockerHub and consumed by the
  RunPod ops helpers for cloud deployment.

## Supported models and tasks

### Tasks

| Task        | Extracts                                                  | Official metric                  |
|-------------|-----------------------------------------------------------|----------------------------------|
| `pseudo`    | Patient identifiers with exact character offsets          | `span_micro_f1`                  |
| `infectio`  | Bacteria, infections, anatomical sites with negation      | `text_label_negation_micro_f1`   |
| `response`  | Treatment-outcome classification with justification span  | `text_label_micro_f1`            |
| `scenario`  | Eight structured patient fields                           | `text_label_micro_f1`            |

### Models shipped as ready-to-use configs

| Model            | Runtime   | Family            | Source                                              |
|------------------|-----------|-------------------|-----------------------------------------------------|
| `qwen25_7b`      | vLLM      | Decoder, 7B       | [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) |
| `gemma2_9b`      | vLLM      | Decoder, 9B       | [google/gemma-2-9b-it](https://huggingface.co/google/gemma-2-9b-it) |
| `ministral_8b`   | vLLM      | Decoder, 8B       | [mistralai/Ministral-8B-Instruct-2410](https://huggingface.co/mistralai/Ministral-8B-Instruct-2410) |
| `aya_8b`         | vLLM      | Decoder, 8B       | [CohereForAI/aya-expanse-8b](https://huggingface.co/CohereForAI/aya-expanse-8b) |
| `llama31_8b`     | vLLM      | Decoder, 8B       | [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) |
| `lucie_7b`       | vLLM      | Decoder, 7B       | [OpenLLM-France/Lucie-7B-Instruct-v1.1](https://huggingface.co/OpenLLM-France/Lucie-7B-Instruct-v1.1) |
| `gliner2_multi`  | GLiNER2   | Encoder, 200M     | [gliner-community/gliner2_multi](https://huggingface.co/gliner-community) |

Any other HuggingFace model compatible with vLLM can be added by
writing a single YAML file under `configs/models/`. See
[Adding a new model](https://lounesmechouek.github.io/parhaf-clibench/guide/adding-a-model/).

## Quickstart

```bash
# 1. Install with vLLM support (requires a CUDA GPU)
uv sync --extra dev --extra vllm

# 2. Smoke test against the mock runtime
make smoke

# 3. Full campaign over the shipped suite
uv run parhaf-clibench run \
  --suite configs/suites/v1_full.yaml \
  --output-dir results/v1_full

# 4. Build the figures
uv run python analysis/build_figures.py --run-dir results/v1_full

# 5. Explore the results
uv run --extra ui streamlit run ui/app.py
```

Full details on install profiles, prefetching caches, and
troubleshooting are on the
[Getting started](https://lounesmechouek.github.io/parhaf-clibench/getting-started/installation/)
pages.

## Documentation

The full documentation lives at
[lounesmechouek.github.io/parhaf-clibench](https://lounesmechouek.github.io/parhaf-clibench/).
Build it locally with:

```bash
make docs-serve
```

## Results of the Pilot Study

The first study published with this codebase, *Small Language
Models Meet the Clinic*, evaluated seven systems on the four PARHAF
tasks across 65,065 predictions. The full write-up is on the
[companion blog post](https://lounesmechouek.com/writing/slm_benchmark/).

The package is independent of that study. Any reference number
cited in the blog post is tied to a specific Git SHA, a specific
model set, and a specific suite file. Running the same suite on a
new model set will produce a different result, and that is the
intended use.

## UI and monitoring

### Streamlit UI

```bash
uv run --extra ui streamlit run ui/app.py
```

![Streamlit UI](https://lounesmechouek.github.io/parhaf-clibench/assets/streamlit.png)

The UI loads directly from a run directory. Nine pages cover the
overview, per-task deep dives, model cards, head-to-head paired
deltas, robustness, subgroups, error taxonomy, and methodology.

### Terminal monitoring dashboard

```bash
uv run python -m monitoring.dashboard --run-dir results/<run_id>
```

![Terminal monitoring dashboard](https://lounesmechouek.github.io/parhaf-clibench/assets/dashboard_parhaf.png)

The dashboard attaches to a live or finished run directory and
displays the active cell, latency distribution, throughput, output
validity, and the latest events. It is safe to attach and detach as
many instances as needed.

## LLMOps and reproducibility

Reproducibility is a first-class property of every run. The package
pins and hashes:

- **Model weights and tokenizer** through pinned HuggingFace commit
  SHAs in `configs/models/`.
- **Datasets** through pinned revisions in `configs/tasks/` plus a
  content fingerprint (SHA-256 over name, example count, document
  IDs, and text digests).
- **Prompts and the few-shot bank** through SHA-256 of the rendered
  template, stamped into the run manifest.
- **Runtime configuration** through YAML hashes.
- **Code** through the Git SHA and a dirty-tree flag captured at run
  start.
- **Inference parameters and bootstrap seed** through the suite
  config.

Every run writes a `manifest.json` that collects all of the above.
An independent scoring audit reloads predictions from JSONL,
reparses them through the canonical schema, and recomputes scores
to six decimal places. CI (`ruff`, `mypy --strict`, `pytest` with a
70% coverage floor) keeps the stack honest, and a Docker image on
DockerHub pins the runtime environment itself.

The full LLMOps story (versioning, manifests, determinism, scoring
audit, CI, Docker, RunPod, artifacts) is documented on the
[LLMOps section](https://lounesmechouek.github.io/parhaf-clibench/llmops/)
of the docs site.

## Deployment

The reference target is a GPU pod on RunPod running the vLLM Docker
image built by the CD pipeline.

```bash
# Pull a pinned image
docker pull <registry>/parhaf-clibench-vllm:sha-<short>

# Launch, poll, and collect a RunPod campaign
uv run python -m parhaf_clibench.ops.launch_runpod   --name v1-full --image <registry>/parhaf-clibench-vllm:sha-<short>

uv run python -m parhaf_clibench.ops.poll_runpod     --name v1-full

uv run python -m parhaf_clibench.ops.collect_results --name v1-full --remote /workspace/results/v1_full --local results/v1_full

uv run python -m parhaf_clibench.ops.stop_runpod     --name v1-full
```

Any GPU cloud that can run the Docker image works. See the
[Deployment pages](https://lounesmechouek.github.io/parhaf-clibench/llmops/docker/)
for image tags, entrypoints, and RunPod specifications.

## Project layout

```
parhaf-clibench/
  src/parhaf_clibench/     # Python package: orchestration, runtimes, tasks, scoring, reporting
  configs/                  # Suites, models, tasks, runtimes, dataset contracts
  prompts/                  # Jinja2 templates per task and track
  assets/fewshot/           # Fixed demonstration banks
  ui/                       # Streamlit result explorer
  monitoring/               # Rich-based terminal dashboard
  analysis/                 # Figure builder and post-hoc analysis scripts
  infra/docker/             # Dockerfiles and entrypoints
  infra/runpod/             # RunPod pod specs and helpers
  tests/                    # Unit, integration, and smoke tests
  docs/                     # MkDocs site sources
  .github/workflows/        # CI, Docker CD, docs deployment
```

## Contributing

```bash
make install        # uv sync --extra dev + pre-commit hooks
make gate           # ruff + mypy strict + pytest with coverage
make docs-serve     # live docs preview
```

Full contribution guidelines are on the
[Contributing page](https://lounesmechouek.github.io/parhaf-clibench/about/contributing/).

## Citation

```bibtex
@software{parhaf_clibench,
  author  = {Mechouek, Lounes},
  title   = {parhaf-clibench: A reproducible benchmark for clinical information extraction with language models on PARHAF},
  year    = {2026},
  url     = {https://github.com/lounesmechouek/parhaf-clibench},
  version = {0.1.0}
}
```

The [study v1](https://lounesmechouek.com/writing/slm_benchmark/)
has its own reference documented on the companion page.

## License and acknowledgements

Released under the [MIT License](LICENSE).

The PARHAF corpus is released by the
[Health Data Hub](https://www.health-data-hub.fr/) under its own
terms; follow the dataset license before using it. Thanks to the
authors of [vLLM](https://github.com/vllm-project/vllm) and
[GLiNER](https://github.com/urchade/GLiNER) for the runtimes this
package builds on.
