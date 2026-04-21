# Runtimes

A **runtime** is the adapter that turns an inference request into a
structured JSON response. All runtimes implement the same small
interface, which decouples the orchestrator from the specifics of
each backend.

## The `RuntimeBackend` interface

Source:
[`parhaf_clinbench.runtimes.base`](../reference/parhaf_clinbench/runtimes/base.md).

```python
class RuntimeBackend(ABC):
    @property
    def name(self) -> str: ...
    @property
    def version(self) -> str: ...
    def infer(self, request: InferenceRequest) -> str: ...
    def close(self) -> None: ...
```

`infer` takes an `InferenceRequest` (document ID, task, track, prompt,
text, optional gold) and returns a raw JSON string. The parser,
validator, and aligner downstream are runtime-agnostic: they see only
the JSON.

## Built-in runtimes

### vLLM

Source:
[`parhaf_clinbench.runtimes.vllm`](../reference/parhaf_clinbench/runtimes/vllm.md).

The vLLM runtime speaks the OpenAI-compatible HTTP API, so any model
that vLLM can serve works without code changes. It uses the
`xgrammar` guided-decoding backend to constrain the output to the
task-specific JSON schema, with enums for labels and attributes.

Configured through `configs/runtimes/vllm.yaml`:

```yaml title="configs/runtimes/vllm.yaml"
runtime_id: vllm
api_base: http://127.0.0.1:8000/v1
healthcheck_url: http://127.0.0.1:8000/health
chat_endpoint: /chat/completions
timeout_seconds: 240
startup_timeout_seconds: 1200
max_workers: 16
max_num_seqs: 128
gpu_memory_utilization: 0.92
guided_decoding_backend: xgrammar
disable_log_requests: true
enable_chunked_prefill: true
enable_prefix_caching: true
```

Chunking for long documents is handled automatically when the prompt
plus the document exceeds the model context window. Chunks are
re-aligned after inference by the merger in
[`parhaf_clinbench.chunking`](../reference/parhaf_clinbench/chunking/index.md).

### GLiNER2

Source:
[`parhaf_clinbench.runtimes.gliner`](../reference/parhaf_clinbench/runtimes/gliner.md).

GLiNER2 runs locally through its own Python package. It takes
task-specific label descriptions rather than a prompt, produces
aligned spans directly, and applies a small heuristic for negation
detection on the `infectio` task. The configuration under
`configs/runtimes/gliner.yaml` exposes the confidence threshold and
the negation context window.

The GLiNER2 runtime refuses the `few-shot` track: demonstrations are
meaningless for a non-generative model.

### Mock

Source:
[`parhaf_clinbench.runtimes.mock`](../reference/parhaf_clinbench/runtimes/mock.md).

The mock runtime returns a fixed, schema-valid JSON blob. Its only
purpose is to let CI exercise the pipeline on a CPU-only runner.

## Choosing a runtime for a model

A model is wired to a runtime through the suite config:

```yaml
runtime_default: vllm
runtime_overrides:
  gliner2_multi: gliner
```

Any model not explicitly overridden uses the default. Overrides are
the idiomatic way to mix LLMs and encoders in the same suite.

## Adding a new runtime

1. Implement `RuntimeBackend` in a new module under
   `parhaf_clinbench/runtimes/`.
2. Declare its configuration schema alongside the existing ones.
3. Register it in the runtime factory in
   `parhaf_clinbench.runtimes.__init__`.
4. Add a configuration file under `configs/runtimes/`.

This is rare. The common extension path is **adding a new model
under the existing vLLM runtime**, which is a one-YAML-file change.
See [Adding a new model](../guide/adding-a-model.md).
