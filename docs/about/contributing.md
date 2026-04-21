# Contributing

Contributions are welcome. The project is small and opinionated, so
please read this page before sending a large change.

## Scope

Pull requests that fit the scope of the package:

- New models under `configs/models/`.
- New suites under `configs/suites/`.
- New prompt templates or few-shot banks.
- Bug fixes and performance improvements in the runtimes, scorers,
  or reporting layer.
- Documentation improvements.
- New robustness metrics alongside the existing ones.

Out of scope for the main branch:

- New clinical datasets. PARHAF is the object of study.
- New UI frameworks. Streamlit and Rich cover the operator needs.
- Alternative package managers. `uv` is the supported one.

## Development workflow

```bash
git clone https://github.com/lounesmechouek/parhaf-clinbench.git
cd parhaf-clinbench
make install
```

`make install` runs `uv sync --extra dev` and installs the
pre-commit hooks.

Before sending a pull request:

```bash
make gate
```

The gate chains `ruff`, `mypy` (strict), and `pytest` with a 70%
coverage floor. It is the same command CI runs, so a clean local
pass is a very strong predictor of a green pull request.

## Style rules

- Type everything. The codebase is `mypy --strict`.
- Google-style docstrings. The API reference renders them directly.
- Short functions. Prefer composition over flags.
- No hidden side effects. Public functions take explicit inputs and
  return explicit outputs.
- English in code and comments.

## Writing documentation

The docs site is built with MkDocs Material. See the
[mkdocs.yml](https://github.com/lounesmechouek/parhaf-clinbench/blob/main/mkdocs.yml)
file for the navigation. To preview your changes:

```bash
make docs-serve
```

When writing new pages, match the tone of the existing ones:

- No em-dashes.
- Short sentences.
- Concrete references. Link to a file or a function rather than
  restating what it does.
- Admonitions for warnings and tips, not for decoration.

## Commit messages

The repository uses conventional-style messages when the change is
large enough to need a category:

- `feat:` for new functionality.
- `fix:` for bug fixes.
- `refactor:` for changes that do not affect behavior.
- `docs:` for documentation-only changes.
- `chore:` for tooling and CI.

A one-line summary is enough for small changes.

## Reporting bugs

Open an issue with:

- The command you ran.
- The full output, including the traceback if any.
- The `manifest.json` of the run when relevant.
- The Git SHA and the Python version.

The manifest is usually enough to reproduce the issue without
additional back-and-forth.
