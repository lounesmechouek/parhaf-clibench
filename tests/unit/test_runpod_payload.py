from __future__ import annotations

import json
import sys
from typing import Any

import pytest

from parhaf_clinbench.core.settings import get_settings
from parhaf_clinbench.ops.launch_runpod import launch


def test_launch_runpod_starts_existing_pod(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[str, str]] = []

    class _FakeClient:
        def __init__(self, api_base: str, api_key: str) -> None:
            seen.append(("init", f"{api_base}|{api_key}"))

        def get_pod(self, pod_id: str) -> dict[str, Any]:
            seen.append(("get", pod_id))
            return {"id": pod_id, "status": "STOPPED"}

        def start_pod(self, pod_id: str) -> dict[str, Any]:
            seen.append(("start", pod_id))
            return {"id": pod_id, "status": "STARTING"}

    monkeypatch.setenv("RUNPOD_API_KEY", "rp_test")
    monkeypatch.setenv("RUNPOD_POD_ID", "pod_abc")
    get_settings.cache_clear()
    monkeypatch.setattr("parhaf_clinbench.ops.launch_runpod.RunpodClient", _FakeClient)
    monkeypatch.setattr("parhaf_clinbench.ops.launch_runpod.run_local_quality_gate", lambda: None)
    monkeypatch.setattr(sys, "argv", ["parhaf-launch-runpod"])

    launch()

    assert seen == [
        ("init", "https://rest.runpod.io/v1|rp_test"),
        ("get", "pod_abc"),
        ("start", "pod_abc"),
    ]


def test_launch_runpod_dry_run_prints_start_endpoint(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("RUNPOD_POD_ID", "pod_123")
    get_settings.cache_clear()
    monkeypatch.setattr("parhaf_clinbench.ops.launch_runpod.run_local_quality_gate", lambda: None)
    monkeypatch.setattr(sys, "argv", ["parhaf-launch-runpod", "--dry-run"])

    launch()

    payload = json.loads(capsys.readouterr().out)
    assert payload["action"] == "start_existing_pod"
    assert payload["pod_id"] == "pod_123"
    assert payload["endpoint"].endswith("/pods/pod_123/start")


def test_launch_runpod_skips_start_if_already_running(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: list[tuple[str, str]] = []

    class _FakeClient:
        def __init__(self, api_base: str, api_key: str) -> None:
            seen.append(("init", f"{api_base}|{api_key}"))

        def get_pod(self, pod_id: str) -> dict[str, Any]:
            seen.append(("get", pod_id))
            return {"id": pod_id, "status": "RUNNING"}

        def start_pod(self, pod_id: str) -> dict[str, Any]:
            raise AssertionError("start_pod should not be called when already RUNNING")

    monkeypatch.setenv("RUNPOD_API_KEY", "rp_test")
    monkeypatch.setenv("RUNPOD_POD_ID", "pod_run")
    get_settings.cache_clear()
    monkeypatch.setattr("parhaf_clinbench.ops.launch_runpod.RunpodClient", _FakeClient)
    monkeypatch.setattr("parhaf_clinbench.ops.launch_runpod.run_local_quality_gate", lambda: None)
    monkeypatch.setattr(sys, "argv", ["parhaf-launch-runpod"])

    launch()

    payload = json.loads(capsys.readouterr().out)
    assert payload["action"] == "already_active_no_start_called"
    assert payload["status"] == "RUNNING"
    assert seen == [
        ("init", "https://rest.runpod.io/v1|rp_test"),
        ("get", "pod_run"),
    ]
