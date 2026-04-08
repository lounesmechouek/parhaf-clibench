"""Tests for concurrent inference behaviour (ThreadPoolExecutor) in runner.py."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
import yaml

from parhaf_clinbench.core.enums import TaskId
from parhaf_clinbench.core.models import InferenceRequest
from parhaf_clinbench.orchestration.runner import run_campaign

# ---------------------------------------------------------------------------
# Shared fake runtime helpers
# ---------------------------------------------------------------------------


def _valid_output(request: InferenceRequest) -> str:
    """Build a minimal valid CanonicalDocument JSON for any task."""
    # scenario task requires a non-null speciality — use the smoke doc's value
    speciality = "CARDIOLOGIE" if request.task == TaskId.SCENARIO else None
    return json.dumps(
        {
            "document_id": request.document_id,
            "task": request.task.value,
            "speciality": speciality,
            "records": [],
        },
        ensure_ascii=False,
    )


class _FakeRuntime:
    """Always succeeds — returns a minimal valid CanonicalDocument JSON."""

    @property
    def name(self) -> str:
        return "vllm"

    @property
    def version(self) -> str:
        return "test"

    def infer(self, request: InferenceRequest) -> str:
        return _valid_output(request)

    def close(self) -> None:
        return None


class _AlwaysErrorRuntime:
    """Always raises — every call to infer() throws a RuntimeError."""

    @property
    def name(self) -> str:
        return "vllm"

    @property
    def version(self) -> str:
        return "test"

    def infer(self, request: InferenceRequest) -> str:
        raise RuntimeError(f"Simulated inference error for {request.document_id}")

    def close(self) -> None:
        return None


class _PartialErrorRuntime:
    """Raises only for the pseudo document (deterministic by document_id)."""

    ERROR_DOC_ID = "doc-pseudo-1"

    @property
    def name(self) -> str:
        return "vllm"

    @property
    def version(self) -> str:
        return "test"

    def infer(self, request: InferenceRequest) -> str:
        if request.document_id == self.ERROR_DOC_ID:
            raise RuntimeError(f"Simulated error for {request.document_id}")
        return _valid_output(request)

    def close(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Shared monkeypatching helpers
# ---------------------------------------------------------------------------


def _patch_runner(monkeypatch: pytest.MonkeyPatch, runtime: Any) -> None:
    """Apply standard monkeypatches so run_campaign works without real GPU/vLLM."""
    import parhaf_clinbench.orchestration.runner as runner_module

    @contextmanager
    def fake_server_context(**kwargs: Any) -> Iterator[None]:
        del kwargs
        yield

    monkeypatch.setattr(runner_module, "_managed_vllm_server", fake_server_context)
    monkeypatch.setattr(
        runner_module,
        "run_healthcheck",
        lambda runtime_name, runtime_cfg: None,
    )
    monkeypatch.setattr(
        runner_module,
        "_resolve_model_reference",
        lambda *, model_cfg, runtime_name: ("fake-model", None),
    )
    monkeypatch.setattr(
        runner_module,
        "_build_runtime",
        lambda *a, **kw: runtime,
    )


def _smoke_suite_yaml(tmp_path: Path, max_workers: int = 2) -> Path:
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        yaml.safe_dump(
            {
                "suite_id": "concurrent_test",
                "benchmark_version": "v1",
                "tracks": ["zero-shot"],
                "tasks": ["pseudo", "infectio", "response", "scenario"],
                "models": ["qwen25_7b"],
                "runtime_default": "vllm",
                "smoke_dataset": "assets/smoke_sets/v1_smoke.jsonl",
                "parameters": {
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_tokens": 256,
                    "seed": 1,
                    "max_workers": max_workers,
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return suite_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_all_documents_present_in_predictions_jsonl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every example in the smoke set must produce exactly one prediction entry."""
    _patch_runner(monkeypatch, _FakeRuntime())
    suite_path = _smoke_suite_yaml(tmp_path, max_workers=4)

    runs = run_campaign(
        suite_path=suite_path,
        task_selection="all",
        track_selection="all",
        model_selection="all",
        output_dir=tmp_path / "runs",
    )

    assert runs, "Expected at least one run directory"
    predictions_file = runs[0] / "predictions.jsonl"
    assert predictions_file.exists(), "predictions.jsonl must be created"

    lines = [ln for ln in predictions_file.read_text().splitlines() if ln.strip()]
    assert len(lines) == 4, f"Smoke set has 4 docs; got {len(lines)} prediction lines"


def test_timings_jsonl_matches_document_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """timings.jsonl must have exactly one entry per document."""
    _patch_runner(monkeypatch, _FakeRuntime())
    suite_path = _smoke_suite_yaml(tmp_path, max_workers=2)

    runs = run_campaign(
        suite_path=suite_path,
        task_selection="all",
        track_selection="all",
        model_selection="all",
        output_dir=tmp_path / "runs",
    )

    timings_file = runs[0] / "timings.jsonl"
    lines = [ln for ln in timings_file.read_text().splitlines() if ln.strip()]
    assert len(lines) == 4


def test_inference_error_captured_in_errors_jsonl(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When infer() raises, the error must appear in errors.jsonl."""
    _patch_runner(monkeypatch, _AlwaysErrorRuntime())
    suite_path = _smoke_suite_yaml(tmp_path, max_workers=2)

    runs = run_campaign(
        suite_path=suite_path,
        task_selection="all",
        track_selection="all",
        model_selection="all",
        output_dir=tmp_path / "runs",
    )

    errors_file = runs[0] / "errors.jsonl"
    assert errors_file.exists()
    error_lines = [ln for ln in errors_file.read_text().splitlines() if ln.strip()]
    assert len(error_lines) == 4, "All 4 docs should produce errors"

    first_error = json.loads(error_lines[0])
    assert "Runtime error" in first_error["error"]
    assert "Simulated inference error" in first_error["error"]


def test_inference_error_produces_empty_prediction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When infer() raises, predictions.jsonl still has an entry (empty prediction)."""
    _patch_runner(monkeypatch, _AlwaysErrorRuntime())
    suite_path = _smoke_suite_yaml(tmp_path, max_workers=2)

    runs = run_campaign(
        suite_path=suite_path,
        task_selection="all",
        track_selection="all",
        model_selection="all",
        output_dir=tmp_path / "runs",
    )

    predictions_file = runs[0] / "predictions.jsonl"
    lines = [ln for ln in predictions_file.read_text().splitlines() if ln.strip()]
    assert len(lines) == 4, "Error docs still need a prediction entry"

    # All entries should have is_schema_valid=False (empty predictions)
    for line in lines:
        entry = json.loads(line)
        assert entry["is_schema_valid"] is False


def test_partial_error_predictions_count_unaffected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With mixed success/error, total predictions.jsonl count remains = doc count."""
    _patch_runner(monkeypatch, _PartialErrorRuntime())
    suite_path = _smoke_suite_yaml(tmp_path, max_workers=2)

    runs = run_campaign(
        suite_path=suite_path,
        task_selection="all",
        track_selection="all",
        model_selection="all",
        output_dir=tmp_path / "runs",
    )

    predictions_file = runs[0] / "predictions.jsonl"
    lines = [ln for ln in predictions_file.read_text().splitlines() if ln.strip()]
    assert len(lines) == 4

    errors_file = runs[0] / "errors.jsonl"
    error_lines = [ln for ln in errors_file.read_text().splitlines() if ln.strip()]
    assert len(error_lines) == 1, "Only doc-pseudo-1 should produce an error"
    assert json.loads(error_lines[0])["document_id"] == _PartialErrorRuntime.ERROR_DOC_ID


def test_max_workers_one_still_processes_all_docs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """max_workers=1 (sequential-equivalent) must process all documents correctly."""
    _patch_runner(monkeypatch, _FakeRuntime())
    suite_path = _smoke_suite_yaml(tmp_path, max_workers=1)

    runs = run_campaign(
        suite_path=suite_path,
        task_selection="all",
        track_selection="all",
        model_selection="all",
        output_dir=tmp_path / "runs",
    )

    predictions_file = runs[0] / "predictions.jsonl"
    lines = [ln for ln in predictions_file.read_text().splitlines() if ln.strip()]
    assert len(lines) == 4
