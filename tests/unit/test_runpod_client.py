from __future__ import annotations

from typing import Any, cast

import pytest
import requests

from parhaf_clinbench.ops.runpod_client import RunpodClient


class _FakeResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200, text: str = "") -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = text

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            error = requests.HTTPError("http error")
            error.response = cast(Any, self)
            raise error

    def json(self) -> dict[str, Any]:
        return self._payload


def test_runpod_client_launch_pod_mock_api(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[str, str]] = []

    def fake_request(**kwargs: Any) -> _FakeResponse:
        seen.append((str(kwargs["method"]), str(kwargs["url"])))
        return _FakeResponse({"id": "pod_123", "status": "PENDING"})

    monkeypatch.setattr("requests.request", fake_request)
    client = RunpodClient(api_base="https://api.test/v1", api_key="token")
    payload = client.launch_pod({"templateId": "tpl"})
    assert payload["id"] == "pod_123"
    assert seen == [("POST", "https://api.test/v1/pods")]


def test_runpod_client_start_pod_mock_api(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[str, str]] = []

    def fake_request(**kwargs: Any) -> _FakeResponse:
        seen.append((str(kwargs["method"]), str(kwargs["url"])))
        return _FakeResponse({"id": "pod_123", "status": "STARTING"})

    monkeypatch.setattr("requests.request", fake_request)
    client = RunpodClient(api_base="https://api.test/v1", api_key="token")
    payload = client.start_pod("pod_123")
    assert payload["status"] == "STARTING"
    assert seen == [("POST", "https://api.test/v1/pods/pod_123/start")]


def test_runpod_client_http_error_exposes_response_body(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_request(**kwargs: Any) -> _FakeResponse:
        return _FakeResponse(
            {"error": "Pod cannot be started from current state"},
            status_code=500,
            text='{"error":"Pod cannot be started from current state"}',
        )

    monkeypatch.setattr("requests.request", fake_request)
    client = RunpodClient(api_base="https://api.test/v1", api_key="token", max_retries=1)
    with pytest.raises(RuntimeError) as exc_info:
        client.start_pod("pod_123")
    message = str(exc_info.value)
    assert "HTTP 500 POST https://api.test/v1/pods/pod_123/start" in message
    assert "Pod cannot be started from current state" in message


def test_runpod_client_terminate_uses_delete(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[tuple[str, str]] = []

    def fake_request(**kwargs: Any) -> _FakeResponse:
        seen.append((str(kwargs["method"]), str(kwargs["url"])))
        return _FakeResponse({"id": "pod_123", "status": "TERMINATED"})

    monkeypatch.setattr("requests.request", fake_request)
    client = RunpodClient(api_base="https://api.test/v1", api_key="token")
    payload = client.terminate_pod("pod_123")
    assert payload["status"] == "TERMINATED"
    assert seen == [("DELETE", "https://api.test/v1/pods/pod_123")]


def test_wait_pod_fails_fast_on_terminal_status(monkeypatch: pytest.MonkeyPatch) -> None:
    client = RunpodClient(api_base="https://api.test/v1", api_key="token")
    statuses = iter(
        [
            {"id": "pod_123", "status": "STARTING"},
            {"id": "pod_123", "status": "FAILED"},
        ]
    )

    monkeypatch.setattr(RunpodClient, "get_pod", lambda self, pod_id: next(statuses))
    monkeypatch.setattr("time.sleep", lambda seconds: None)

    with pytest.raises(RuntimeError, match="terminal status `FAILED`"):
        client.wait_pod(pod_id="pod_123", target_status="RUNNING", timeout_seconds=1800)
