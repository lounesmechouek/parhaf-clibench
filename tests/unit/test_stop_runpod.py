from __future__ import annotations

import json
import sys
import types
from typing import Any

import pytest

from parhaf_clinbench.core.settings import get_settings
from parhaf_clinbench.ops.stop_runpod import stop_or_terminate


def test_stop_runpod_dry_run_outputs_action(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("RUNPOD_POD_ID", "pod_abc")
    get_settings.cache_clear()
    monkeypatch.setattr(sys, "argv", ["parhaf-stop-runpod", "--dry-run", "--terminate"])

    stop_or_terminate()

    payload = json.loads(capsys.readouterr().out)
    assert payload == {"action": "terminate", "pod_id": "pod_abc"}


def test_stop_runpod_calls_stop_api(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: list[tuple[str, Any]] = []

    class _FakeClient:
        def __init__(self, api_base: str, api_key: str) -> None:
            seen.append(("init", (api_base, api_key)))

        def stop_pod(self, pod_id: str) -> dict[str, str]:
            seen.append(("stop", pod_id))
            return {"id": pod_id, "status": "STOPPED"}

        def terminate_pod(self, pod_id: str) -> dict[str, str]:
            raise AssertionError("terminate_pod should not be called")

    monkeypatch.setenv("RUNPOD_API_KEY", "rp_test")
    monkeypatch.setenv("RUNPOD_POD_ID", "pod_abc")
    get_settings.cache_clear()
    monkeypatch.setattr("parhaf_clinbench.ops.stop_runpod.RunpodClient", _FakeClient)
    monkeypatch.setattr(sys, "argv", ["parhaf-stop-runpod"])

    stop_or_terminate()

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "STOPPED"
    assert seen == [
        ("init", ("https://rest.runpod.io/v1", "rp_test")),
        ("stop", "pod_abc"),
    ]


def test_stop_runpod_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "parhaf_clinbench.ops.stop_runpod.get_settings",
        lambda: types.SimpleNamespace(
            runpod_pod_id="pod_abc",
            runpod_api_key=None,
            runpod_api_base="https://rest.runpod.io/v1",
        ),
    )
    monkeypatch.setattr(sys, "argv", ["parhaf-stop-runpod"])

    with pytest.raises(ValueError, match="RUNPOD_API_KEY is required"):
        stop_or_terminate()
