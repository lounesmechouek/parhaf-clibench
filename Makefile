.PHONY: install install-hooks install-vllm install-gliner install-ui install-docs lint fmt type test precommit smoke run-example gate docs-serve docs-build docs-deploy

install:
	uv sync --extra dev
	uv run pre-commit install --install-hooks --hook-type pre-commit

install-hooks:
	uv run pre-commit install --install-hooks --hook-type pre-commit

install-vllm:
	uv sync --extra dev --extra vllm

install-gliner:
	uv sync --extra dev --extra gliner

install-ui:
	uv sync --extra ui

lint:
	uv run ruff check src tests

fmt:
	uv run ruff format src tests

type:
	uv run mypy

test:
	uv run pytest

precommit:
	uv run pre-commit run --all-files --show-diff-on-failure

gate:
	uv run ruff check src tests
	uv run mypy
	uv run pytest

smoke:
	uv run parhaf-clinbench smoke --suite configs/suites/v1_smoke.yaml --output-dir results/smoke

run-example:
	uv run parhaf-clinbench run --task all --track all --model all --output-dir results/example

install-docs:
	uv sync --extra docs

docs-serve:
	uv run --extra docs mkdocs serve

docs-build:
	uv run --extra docs mkdocs build --strict

docs-deploy:
	uv run --extra docs mkdocs gh-deploy --force --clean --remote-branch gh-pages
