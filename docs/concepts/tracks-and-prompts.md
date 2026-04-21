# Tracks and prompts

A **track** is a prompting regime. The package defines two:

- `zero-shot`: instruction, schema, and document. No examples.
- `few-shot`: the same prompt augmented with a fixed bank of three
  demonstration examples.

Tracks are selected at suite level. Each combination of model, task,
and track is a distinct benchmark cell with its own directory on
disk.

## Prompt templates

Templates live under `prompts/<task>/<track>.jinja2` and are rendered
through
[`parhaf_clinbench.prompting.render`](../reference/parhaf_clinbench/prompting/render.md).
The rendering step receives a dynamic context produced by
[`parhaf_clinbench.prompting.contracts`](../reference/parhaf_clinbench/prompting/contracts.md),
which assembles:

- The canonical JSON schema expected by the task.
- The task-specific label inventory and attribute domain.
- The offset policy (byte-level for `pseudo`, text-only for the
  others).
- For chunked documents, the begin and end markers that let the
  merger realign spans against the full document.

Templates are written in French to match the corpus language. Any
instruction change goes through a template edit, which means the
template hash changes and the run is clearly different from the
previous one.

## Few-shot bank

Demonstrations are stored under `assets/fewshot/<task>_examples.txt`
and loaded once per task-track pair by the orchestrator. The bank is
a constant input of the benchmark, not a hyperparameter. If you want
to test a different demonstration set, write a new file and reference
it from a suite override.

!!! warning "GLiNER2 does not support few-shot"
    The GLiNER2 runtime rejects the `few-shot` track and raises a
    clear error at campaign start. Encoder-based extraction does not
    use a prompt, so the notion of demonstrations does not apply.

## Prompt hashing

Every rendered template is hashed with SHA-256 at the start of a
campaign by
[`parhaf_clinbench.orchestration.runner`](../reference/parhaf_clinbench/orchestration/runner.md),
and the resulting map is stamped into the run manifest. Two runs that
share the same prompt hash share the exact same instruction. This is
the cheapest way to diagnose whether a score delta is due to a
prompt change or to something else.

See [Versioning and hashing](../llmops/versioning.md) for the
complete list of objects that the package hashes.

## Structured output contract

Every prompt ends with the same structured-output contract: the model
must emit a JSON object matching the canonical schema of the task.
This contract is enforced at decoding time by the vLLM runtime
through the xgrammar backend. The prompt restates it in natural
language so that models without guided decoding still have a chance
to comply.

## Editing a template

1. Edit the file under `prompts/<task>/<track>.jinja2`.
2. Run `make smoke` to confirm the template still renders and the
   schema still parses.
3. Commit the change. The prompt hash in the next run's manifest
   will reflect the new template.
