from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
import yaml

from parhaf_clinbench.core.models import InferenceRequest
from parhaf_clinbench.orchestration.runner import run_campaign


class _FakeRuntime:
    @property
    def name(self) -> str:
        return "vllm"

    @property
    def version(self) -> str:
        return "test"

    def infer(self, request: InferenceRequest) -> str:
        return json.dumps(
            {
                "document_id": request.document_id,
                "task": request.task.value,
                "speciality": None,
                "records": [],
            },
            ensure_ascii=False,
        )

    def close(self) -> None:
        return None


def test_run_campaign_applies_suite_generation_params_to_vllm_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite_path = tmp_path / "suite_vllm_params.yaml"
    suite_payload = {
        "suite_id": "test_vllm_params",
        "benchmark_version": "v1",
        "tracks": ["zero-shot"],
        "tasks": ["pseudo"],
        "models": ["qwen25_7b"],
        "runtime_default": "vllm",
        "smoke_dataset": "assets/smoke_sets/v1_smoke.jsonl",
        "parameters": {
            "temperature": 0.23,
            "top_p": 0.81,
            "max_tokens": 321,
            "seed": 17,
        },
    }
    suite_path.write_text(yaml.safe_dump(suite_payload, sort_keys=False), encoding="utf-8")

    import parhaf_clinbench.orchestration.runner as runner_module

    captured_runtime_payloads: list[dict[str, Any]] = []

    @contextmanager
    def fake_server_context(**kwargs: Any) -> Iterator[None]:
        del kwargs
        yield

    def fake_build_runtime(
        runtime_name: Any,
        model_reference: str,
        runtime_payload: dict[str, Any],
        *,
        hf_token: str | None,
        tokenizer_revision: str = "main",
        max_context_tokens: int = 131072,
    ) -> _FakeRuntime:
        del runtime_name
        del model_reference
        del hf_token
        del tokenizer_revision
        del max_context_tokens
        captured_runtime_payloads.append(dict(runtime_payload))
        return _FakeRuntime()

    def fake_run_healthcheck(runtime_name: Any, runtime_cfg: Any) -> None:
        del runtime_name
        del runtime_cfg

    monkeypatch.setattr(runner_module, "_managed_vllm_server", fake_server_context)
    monkeypatch.setattr(runner_module, "run_healthcheck", fake_run_healthcheck)
    monkeypatch.setattr(runner_module, "_preflight_check_hf_access", lambda *a, **k: None)
    monkeypatch.setattr(
        runner_module,
        "_resolve_model_reference",
        lambda *, model_cfg, runtime_name: ("fake-model", None),
    )
    monkeypatch.setattr(runner_module, "_build_runtime", fake_build_runtime)

    runs = run_campaign(
        suite_path=suite_path,
        task_selection="all",
        track_selection="all",
        model_selection="all",
        output_dir=tmp_path / "runs",
    )

    assert runs
    assert captured_runtime_payloads
    runtime_payload = captured_runtime_payloads[0]
    assert runtime_payload["temperature"] == 0.23
    assert runtime_payload["top_p"] == 0.81
    assert runtime_payload["max_tokens"] == 321
    assert runtime_payload["seed"] == 17


def test_run_campaign_propagates_max_workers_to_runtime_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    suite_path = tmp_path / "suite_max_workers.yaml"
    suite_payload = {
        "suite_id": "test_max_workers",
        "benchmark_version": "v1",
        "tracks": ["zero-shot"],
        "tasks": ["pseudo"],
        "models": ["qwen25_7b"],
        "runtime_default": "vllm",
        "smoke_dataset": "assets/smoke_sets/v1_smoke.jsonl",
        "parameters": {
            "temperature": 0.0,
            "max_tokens": 256,
            "max_workers": 4,
        },
    }
    suite_path.write_text(yaml.safe_dump(suite_payload, sort_keys=False), encoding="utf-8")

    import parhaf_clinbench.orchestration.runner as runner_module

    captured: list[dict[str, Any]] = []

    @contextmanager
    def fake_server_context(**kwargs: Any) -> Iterator[None]:
        del kwargs
        yield

    def fake_build_runtime(
        runtime_name: Any,
        model_reference: str,
        runtime_payload: dict[str, Any],
        *,
        hf_token: str | None,
        tokenizer_revision: str = "main",
        max_context_tokens: int = 131072,
    ) -> _FakeRuntime:
        del runtime_name, model_reference, hf_token, tokenizer_revision, max_context_tokens
        captured.append(dict(runtime_payload))
        return _FakeRuntime()

    monkeypatch.setattr(runner_module, "_managed_vllm_server", fake_server_context)
    monkeypatch.setattr(runner_module, "run_healthcheck", lambda *a, **kw: None)
    monkeypatch.setattr(runner_module, "_preflight_check_hf_access", lambda *a, **k: None)
    monkeypatch.setattr(
        runner_module,
        "_resolve_model_reference",
        lambda *, model_cfg, runtime_name: ("fake-model", None),
    )
    monkeypatch.setattr(runner_module, "_build_runtime", fake_build_runtime)

    run_campaign(
        suite_path=suite_path,
        task_selection="all",
        track_selection="all",
        model_selection="all",
        output_dir=tmp_path / "runs",
    )

    assert captured
    assert captured[0]["max_workers"] == 4
