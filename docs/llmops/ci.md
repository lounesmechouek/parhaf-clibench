# CI pipeline

The pull-request gate runs in GitHub Actions at
`.github/workflows/ci.yml`. It is the contract between the code on
`main` and the rest of the LLMOps stack: if the gate passes, the
code can be merged, the Docker image can be built, and any manifest
written by a subsequent run is trustworthy.

## What the gate runs

```yaml title=".github/workflows/ci.yml"
jobs:
  quality:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v6
      - uses: actions/setup-python@v6
        with:
          python-version: '3.11'
      - run: pip install uv
      - run: uv sync --frozen --extra dev
      - run: make gate
      - run: uv run parhaf-clinbench smoke \
               --suite configs/suites/v1_smoke.yaml \
               --output-dir results/smoke
```

The `make gate` target chains three checks:

| Check   | Tool                                                | Purpose                                              |
|---------|-----------------------------------------------------|------------------------------------------------------|
| Lint    | `ruff check src tests`                              | Style, imports, common bug patterns.                 |
| Type    | `mypy` in strict mode over `src`, `tests`, `ui`, `monitoring` | Type correctness, unused types, missing stubs. |
| Tests   | `pytest` with a 70% coverage floor on the source tree | Functional correctness.                              |

The smoke run that follows the gate exercises the end-to-end code
path against the mock runtime. It does not require a GPU.

## Why 70% coverage

The threshold is set in `pyproject.toml` under
`[tool.coverage.report]`. It is not a quality bar by itself; it is a
regression tripwire. Any deletion of a tested branch or any new
untested code path above the threshold moves the number below 70%
and fails the gate.

## Running the gate locally

```bash
make gate
```

The command is identical to the one CI runs, so a clean local pass
is a very strong predictor of CI success.

## Pre-commit

`.pre-commit-config.yaml` wires the same `ruff` and `mypy` checks at
the pre-commit stage. Install them once with:

```bash
make install
```

Pre-commit is not a replacement for CI, but it catches the same
issues on the author's machine before the push.

## Release signals beyond CI

The gate is the only gate for merging into `main`. The separate
[Docker workflow](docker.md) is a release signal: after a merge,
GitHub Actions builds and pushes a new image tagged with the
immutable commit SHA. A failed Docker build does not roll back the
code merge; it is a follow-up alert on the release.
