# Adding a new model

Any model served by vLLM can be benchmarked by the package. This
page walks through the full flow, from picking a HuggingFace
identifier to reading the first F1.

## The common case: one YAML file

1. **Pick a HuggingFace revision**. Browse the model repository on
   HuggingFace and copy a full commit SHA. Never use `main`. Pinning
   to `main` means a silent upgrade can invalidate your results.

2. **Create `configs/models/<id>.yaml`**. Use a short, filesystem-safe
   identifier. Four characters to two words is the norm.

    ```yaml title="configs/models/mistral_small_24b.yaml"
    model_id: mistral_small_24b
    hf_id: mistralai/Mistral-Small-24B-Instruct-2501
    revision: abc123def456abc123def456abc123def456abc1
    tokenizer_revision: abc123def456abc123def456abc123def456abc1
    family: llm
    max_context_tokens: 32768
    ```

3. **Reference the model from a suite** under `configs/suites/`:

    ```yaml
    models:
      - mistral_small_24b
    runtime_default: vllm
    ```

4. **Prefetch the weights** to validate the pin:

    ```bash
    uv run parhaf-clinbench prefetch --model mistral_small_24b
    ```

5. **Run a smoke** against the new model to confirm vLLM loads it
   and that the prompts render correctly:

    ```bash
    uv run parhaf-clinbench smoke \
      --suite configs/suites/v1_smoke.yaml \
      --output-dir results/smoke
    ```

6. **Launch a real run** once the smoke passes:

    ```bash
    uv run parhaf-clinbench run \
      --suite configs/suites/<your_suite>.yaml \
      --output-dir results/<your_run>
    ```

That is the whole procedure.

## Field reference

| Field                  | Required | Meaning                                                                   |
|------------------------|----------|---------------------------------------------------------------------------|
| `model_id`             | yes      | Short ID used in run directories, UI tables, and the suite list.          |
| `hf_id`                | yes      | HuggingFace repository path (`owner/name`).                               |
| `revision`             | yes      | Full commit SHA of the model weights.                                     |
| `tokenizer_revision`   | yes      | Full commit SHA of the tokenizer. Often identical to `revision`.          |
| `family`               | yes      | `llm` for a generative model, `encoder` for a span-labeling encoder.      |
| `max_context_tokens`   | yes      | Context window in tokens. Used by the chunker to keep prompts in range.   |

The loader refuses to start without any of these fields.

## When to override the runtime

Most LLMs run under the default vLLM runtime. Override it only when
the architecture is not supported by vLLM or when you want to pin a
specific custom backend:

```yaml title="configs/suites/my_suite.yaml"
runtime_overrides:
  mistral_small_24b: vllm
  gliner2_multi: gliner
```

A runtime override at the suite level keeps the model config generic
and lets you swap runtimes without editing the model file.

## Serving the model

The vLLM server is launched externally to the benchmark. The
reference path is the Docker image documented on the
[Docker page](../llmops/docker.md). On a workstation, a direct
command looks like:

```bash
uv run --extra vllm python -m vllm.entrypoints.openai.api_server \
  --model mistralai/Mistral-Small-24B-Instruct-2501 \
  --revision abc123def456abc123def456abc123def456abc1 \
  --guided-decoding-backend xgrammar \
  --max-num-seqs 128 \
  --gpu-memory-utilization 0.92
```

The benchmark client hits `http://127.0.0.1:8000/v1` by default.
Override the endpoint in `configs/runtimes/vllm.yaml` if your server
listens elsewhere.

## Troubleshooting

- **Model out of memory**: lower `gpu_memory_utilization` in the
  runtime config, or pick a smaller `max_num_seqs`. 7B to 9B bf16
  models fit on a single 24 GB GPU at `0.85`.
- **Tokenizer mismatch**: a different tokenizer revision than
  weights. Pin them to the same SHA unless you know what you are
  doing.
- **Prompt longer than context**: the chunker will split the
  document. If you see an error, raise `max_context_tokens` in the
  model YAML, or lower `max_tokens` in the suite parameters.
