.PHONY: install install-locale lint type test smoke run-example gate

install:
	uv sync --extra dev

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

gate:
	uv run ruff check src tests
	uv run mypy
	uv run pytest

smoke:
	uv run parhaf-clinbench smoke --suite configs/suites/v1_smoke.yaml --output-dir results/smoke

run-example:
	uv run parhaf-clinbench run --task all --track all --model all --output-dir results/example
