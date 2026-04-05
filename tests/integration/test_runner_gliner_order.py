from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from parhaf_clinbench.core.models import InferenceRequest
from parhaf_clinbench.orchestration.runner import run_campaign
from parhaf_clinbench.runtimes.prefetch import PrefetchResult


class _FakeRuntime:
    def __init__(self, runtime_name: str) -> None:
        self._runtime_name = runtime_name

    @property
    def name(self) -> str:
        return self._runtime_name

    @property
    def version(self) -> str:
        return "test"

    def infer(self, request: InferenceRequest) -> str:
        payload: dict[str, Any] = {
            "document_id": request.document_id,
            "task": request.task.value,
            "records": [],
        }
        if request.task.value == "scenario":
            payload["speciality"] = request.gold.speciality if request.gold is not None else "unknown"
        return json.dumps(payload, ensure_ascii=False)

    def close(self) -> None:
        return None


def test_run_campaign_prioritizes_gliner_model_first(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite_path = tmp_path / "suite_gliner_first.yaml"
    suite_payload = {
        "suite_id": "test_gliner_first",
        "benchmark_version": "v1",
        "tracks": ["zero-shot"],
        "tasks": ["pseudo"],
        "models": ["qwen25_7b", "gliner2_multi"],
        "runtime_default": "mock",
        "runtime_overrides": {"gliner2_multi": "gliner"},
        "smoke_dataset": "assets/smoke_sets/v1_smoke.jsonl",
        "parameters": {"temperature": 0.0},
    }
    suite_path.write_text(yaml.safe_dump(suite_payload, sort_keys=False), encoding="utf-8")

    import parhaf_clinbench.orchestration.runner as runner_module

    def fake_build_runtime(
        runtime_name: Any,
        model_reference: str,
        runtime_payload: dict[str, Any],
        *,
        hf_token: str | None,
    ) -> _FakeRuntime:
        del model_reference
        del runtime_payload
        del hf_token
        return _FakeRuntime(runtime_name.value)

    def fake_resolve_model_reference(*, model_cfg: Any, runtime_name: Any) -> tuple[str, PrefetchResult]:
        del runtime_name
        return (
            model_cfg.hf_id,
            PrefetchResult(
                hf_id=model_cfg.hf_id,
                revision=model_cfg.revision,
                local_path=f"/workspace/models/{model_cfg.model_id}",
                cache_hit=True,
            ),
        )

    monkeypatch.setattr(runner_module, "_build_runtime", fake_build_runtime)
    monkeypatch.setattr(runner_module, "_resolve_model_reference", fake_resolve_model_reference)

    runs = run_campaign(
        suite_path=suite_path,
        task_selection="all",
        track_selection="all",
        model_selection="all",
        output_dir=tmp_path / "runs",
    )

    assert len(runs) == 2
    assert runs[0].name.startswith("gliner2_multi_")

    first_metadata = json.loads((runs[0] / "run_metadata.json").read_text(encoding="utf-8"))
    second_metadata = json.loads((runs[1] / "run_metadata.json").read_text(encoding="utf-8"))

    assert first_metadata["model_execution_order"] == ["gliner2_multi", "qwen25_7b"]
    assert first_metadata["model_execution_index"] == 1
    assert second_metadata["model_execution_index"] == 2
