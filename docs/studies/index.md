# Published studies

The `parhaf-clinbench` package is a general benchmark runner. This
page collects the specific studies that have used it, each at a
pinned revision of the repository.

## Study v1: Small language models meet the clinic

A zero-shot and few-shot evaluation of seven systems (six 7B to 9B
instruction-tuned LLMs and one 200M encoder) on the four PARHAF
clinical information extraction tasks.

- **Read**: [Small Language Models Meet the Clinic](https://lounesmechouek.com/writing/slm_benchmark/)
- **Systems evaluated**: Gemma2-9B-it, Ministral-8B-Instruct,
  Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, Aya-Expanse-8B,
  Lucie-7B-Instruct, and GLiNER2-multi.
- **Predictions**: 65,065 across four tasks on 5,005 documents.
- **Date of the run**: April 2026.
- **Suite**: `configs/suites/v1_full.yaml`.

The study is a snapshot of what a specific model set can do on a
specific corpus. The package evolves independently, and any
reference number in the blog post is tied to the exact revision
used for that run.

## Running your own study

Any new study becomes a new suite under `configs/suites/` and a new
entry on this page. The package does not distinguish between "the"
published study and any other run: they share the same artifact
layout, the same manifest schema, and the same scoring audit.

If you publish a study built on `parhaf-clinbench`, open a pull
request against this page with a short summary, a link, and the
suite file you used.
