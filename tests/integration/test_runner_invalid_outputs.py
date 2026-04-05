from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from parhaf_clinbench.core.models import InferenceRequest
from parhaf_clinbench.orchestration.runner import run_campaign


class _FakeRuntime:
    def __init__(self, *, mode: str) -> None:
        self._mode = mode
        self.calls = 0

    @property
    def name(self) -> str:
        return "mock"

    @property
    def version(self) -> str:
        return "test-invalid-output"

    def infer(self, request: InferenceRequest) -> str:
        self.calls += 1
        if self._mode == "invalid-json":
            return "not-a-json-payload"
        if self._mode == "invalid-speciality":
            return json.dumps(
                {
                    "document_id": request.document_id,
                    "task": "scenario",
                    "speciality": "INVALID_SPECIALITY",
                    "records": [],
                },
                ensure_ascii=False,
            )
        raise AssertionError(f"Unknown fake runtime mode: {self._mode}")

    def close(self) -> None:
        return None


def _write_smoke_suite(tmp_path: Path, *, task: str) -> Path:
    suite_path = tmp_path / f"suite_invalid_{task}.yaml"
    suite_payload = {
        "suite_id": f"invalid_{task}",
        "benchmark_version": "v1",
        "tracks": ["zero-shot"],
        "tasks": [task],
        "models": ["qwen25_7b"],
        "runtime_default": "mock",
        "smoke_dataset": "assets/smoke_sets/v1_smoke.jsonl",
        "parameters": {"temperature": 0.0},
    }
    suite_path.write_text(yaml.safe_dump(suite_payload, sort_keys=False), encoding="utf-8")
    return suite_path


def test_non_json_output_degrades_robustness_without_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import parhaf_clinbench.orchestration.runner as runner_module

    runtime = _FakeRuntime(mode="invalid-json")

    def fake_build_runtime(
        runtime_name: Any,
        model_reference: str,
        runtime_payload: dict[str, Any],
        *,
        hf_token: str | None,
    ) -> _FakeRuntime:
        del runtime_name
        del model_reference
        del runtime_payload
        del hf_token
        return runtime

    monkeypatch.setattr(runner_module, "_build_runtime", fake_build_runtime)
    suite_path = _write_smoke_suite(tmp_path, task="pseudo")

    runs = run_campaign(
        suite_path=suite_path,
        task_selection="all",
        track_selection="all",
        model_selection="all",
        output_dir=tmp_path / "runs",
    )

    payload = json.loads((runs[0] / "metrics.json").read_text(encoding="utf-8"))
    robustness = payload["tracks"][0]["per_task"]["pseudo"]["robustness"]
    assert robustness["raw_json_valid_rate"] == 0.0
    assert robustness["schema_conformity_rate"] == 0.0
    assert robustness["empty_output_rate"] == 1.0
    assert runtime.calls == 1


def test_invalid_speciality_degrades_robustness_without_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import parhaf_clinbench.orchestration.runner as runner_module

    runtime = _FakeRuntime(mode="invalid-speciality")

    def fake_build_runtime(
        runtime_name: Any,
        model_reference: str,
        runtime_payload: dict[str, Any],
        *,
        hf_token: str | None,
    ) -> _FakeRuntime:
        del runtime_name
        del model_reference
        del runtime_payload
        del hf_token
        return runtime

    monkeypatch.setattr(runner_module, "_build_runtime", fake_build_runtime)
    suite_path = _write_smoke_suite(tmp_path, task="scenario")

    runs = run_campaign(
        suite_path=suite_path,
        task_selection="all",
        track_selection="all",
        model_selection="all",
        output_dir=tmp_path / "runs",
    )

    payload = json.loads((runs[0] / "metrics.json").read_text(encoding="utf-8"))
    robustness = payload["tracks"][0]["per_task"]["scenario"]["robustness"]
    assert robustness["raw_json_valid_rate"] == 1.0
    assert robustness["schema_conformity_rate"] == 0.0
    assert robustness["empty_output_rate"] == 1.0
    assert runtime.calls == 1
