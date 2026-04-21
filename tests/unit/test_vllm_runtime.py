from __future__ import annotations

from typing import Any

import pytest

from parhaf_clinbench.core.enums import TaskId, TrackId
from parhaf_clinbench.core.models import InferenceRequest
from parhaf_clinbench.runtimes.vllm import VllmRuntime


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


def test_vllm_runtime_passes_generation_params(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_payloads: list[dict[str, Any]] = []

    def fake_post(url: str, *, json: dict[str, Any], timeout: int) -> _FakeResponse:
        assert url == "http://127.0.0.1:8000/v1/chat/completions"
        assert timeout == 77
        captured_payloads.append(json)
        return _FakeResponse({"choices": [{"message": {"content": '{"ok":true}'}}]})

    monkeypatch.setattr("requests.post", fake_post)

    runtime = VllmRuntime(
        api_base="http://127.0.0.1:8000/v1",
        model_hf_id="Qwen/Qwen2.5-7B-Instruct",
        timeout_seconds=77,
        temperature=0.3,
        top_p=0.9,
        max_tokens=512,
        seed=42,
    )
    request = InferenceRequest(
        document_id="doc-1",
        task=TaskId.PSEUDO,
        track=TrackId.ZEROSHOT,
        prompt="hello",
        text="hello",
    )
    output = runtime.infer(request)

    assert output == '{"ok":true}'
    assert captured_payloads
    payload = captured_payloads[0]
    assert payload["temperature"] == 0.3
    assert payload["top_p"] == 0.9
    assert payload["max_tokens"] == 512
    assert payload["seed"] == 42
    response_format = payload["response_format"]
    assert response_format["type"] == "json_schema"
    json_schema = response_format["json_schema"]
    assert json_schema["strict"] is True
    assert json_schema["schema"]["properties"]["task"]["const"] == "pseudo"


def test_vllm_runtime_uses_scenario_speciality_enum_in_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_payloads: list[dict[str, Any]] = []

    def fake_post(url: str, *, json: dict[str, Any], timeout: int) -> _FakeResponse:
        del url
        del timeout
        captured_payloads.append(json)
        return _FakeResponse({"choices": [{"message": {"content": '{"ok":true}'}}]})

    monkeypatch.setattr("requests.post", fake_post)

    runtime = VllmRuntime(
        api_base="http://127.0.0.1:8000/v1",
        model_hf_id="Qwen/Qwen2.5-7B-Instruct",
        timeout_seconds=10,
    )
    request = InferenceRequest(
        document_id="doc-scen",
        task=TaskId.SCENARIO,
        track=TrackId.ZEROSHOT,
        prompt="hello",
        text="hello",
    )
    _ = runtime.infer(request)

    payload = captured_payloads[0]
    speciality_schema = payload["response_format"]["json_schema"]["schema"]["properties"]["speciality"]
    assert speciality_schema["type"] == "string"
    assert "PNEUMOLOGIE" in speciality_schema["enum"]


def test_vllm_runtime_keeps_raw_non_json_output(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_post(url: str, *, json: dict[str, Any], timeout: int) -> _FakeResponse:
        del url
        del json
        del timeout
        return _FakeResponse({"choices": [{"message": {"content": "not-json"}}]})

    monkeypatch.setattr("requests.post", fake_post)

    runtime = VllmRuntime(api_base="http://127.0.0.1:8000/v1", model_hf_id="model", timeout_seconds=10)
    request = InferenceRequest(
        document_id="doc-2",
        task=TaskId.PSEUDO,
        track=TrackId.ZEROSHOT,
        prompt="prompt",
        text="text",
    )
    assert runtime.infer(request) == "not-json"
