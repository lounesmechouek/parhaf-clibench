.PHONY: install install-hooks install-locale lint fmt type test precommit smoke run-example gate

install:
	uv sync --extra dev
	uv run pre-commit install --install-hooks --hook-type pre-commit

install-hooks:
	uv run pre-commit install --install-hooks --hook-type pre-commit

install-locale:
	uv sync --extra dev --extra locale

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
	uv run pre-commit run --all-files --show-diff-on-failure
	uv run pytest

smoke:
	uv run parhaf-clinbench smoke --suite configs/suites/v1_smoke.yaml --output-dir results/smoke

run-example:
	uv run parhaf-clinbench run --task all --track all --model all --output-dir results/example
