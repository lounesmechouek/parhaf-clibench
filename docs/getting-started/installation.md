# Installation

`parhaf-clinbench` targets Python 3.11 or newer and uses
[`uv`](https://docs.astral.sh/uv/) as its package manager.

## Requirements

- Python 3.11 or newer.
- `uv` for dependency resolution and virtual-environment management.
- A HuggingFace access token for private or gated datasets and models,
  exported as `HF_TOKEN`.
- For any real run with an LLM: an NVIDIA GPU with CUDA 12 drivers.
  A single RTX A6000 is sufficient for the 7B to 9B class.

## Install `uv`

=== "macOS and Linux"

    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```

=== "Windows"

    ```powershell
    powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
    ```

=== "pip fallback"

    ```bash
    pip install uv
    ```

## Clone and sync

```bash
git clone https://github.com/lounesmechouek/parhaf-clinbench.git
cd parhaf-clinbench
uv sync --extra dev
```

The `dev` extra brings in the linter, type checker, and test tooling
used by `make gate`.

## Install profiles

The `pyproject.toml` exposes a set of optional dependency groups.
Pick the ones that match your situation.

| Profile              | Command                            | When to use                                                  |
|----------------------|------------------------------------|--------------------------------------------------------------|
| Base                 | `uv sync`                          | Import the package, run benchmarks against an existing cache.|
| Development          | `uv sync --extra dev`              | Lint, type-check, and test the package.                      |
| Streamlit UI         | `uv sync --extra ui`               | Launch the result explorer against a run directory.          |
| Local vLLM inference | `uv sync --extra vllm`             | Run vLLM on the same machine (requires a CUDA GPU).          |
| GLiNER baseline      | `uv sync --extra gliner`           | Run the GLiNER2 encoder runtime.                             |
| Monitoring dashboard | `uv sync --extra monitoring`       | Install `rich` for the terminal dashboard without `dev`.     |
| Documentation        | `uv sync --extra docs`             | Build or serve this site locally.                            |

The `install-vllm`, `install-gliner`, `install-ui`, and `install-docs`
Makefile targets wrap the common combinations.

## HuggingFace credentials

Several of the PARHAF subsets are gated. Export your token before
running `prefetch-suite` or `run`:

```bash
export HF_TOKEN="hf_xxx"
```

`parhaf-clinbench` reads the token from the environment through the
settings layer in [`parhaf_clinbench.core.settings`](../reference/parhaf_clinbench/core/settings.md).

## Pre-commit hooks

If you plan to contribute, install the pre-commit hooks once:

```bash
make install
```

This runs `uv sync --extra dev` and installs the `ruff` and `mypy`
pre-commit hooks locally.

## Verifying the install

The fastest self-check is the smoke suite, which runs the mock runtime
end to end against a tiny local dataset:

```bash
make smoke
```

See [Smoke test](smoke.md) for what to expect.
