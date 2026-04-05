from __future__ import annotations

import json
import sys
import types
from typing import Any

import pytest

from parhaf_clinbench.core.settings import get_settings
from parhaf_clinbench.ops.poll_runpod import poll


def test_poll_runpod_waits_until_target_status(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    seen: list[tuple[str, Any]] = []

    class _FakeClient:
        def __init__(self, api_base: str, api_key: str) -> None:
            seen.append(("init", (api_base, api_key)))

        def wait_pod(
            self,
            *,
            pod_id: str,
            target_status: str,
            timeout_seconds: int,
            poll_interval_seconds: int,
        ) -> dict[str, str]:
            seen.append(
                (
                    "wait",
                    {
                        "pod_id": pod_id,
                        "target_status": target_status,
                        "timeout_seconds": timeout_seconds,
                        "poll_interval_seconds": poll_interval_seconds,
                    },
                )
            )
            return {"id": pod_id, "status": target_status}

    monkeypatch.setenv("RUNPOD_API_KEY", "rp_test")
    monkeypatch.setenv("RUNPOD_POD_ID", "pod_123")
    get_settings.cache_clear()
    monkeypatch.setattr("parhaf_clinbench.ops.poll_runpod.RunpodClient", _FakeClient)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "parhaf-poll-runpod",
            "--target-status",
            "RUNNING",
            "--timeout-seconds",
            "120",
            "--poll-interval-seconds",
            "5",
        ],
    )

    poll()

    payload = json.loads(capsys.readouterr().out)
    assert payload["id"] == "pod_123"
    assert payload["status"] == "RUNNING"
    assert seen[0] == ("init", ("https://rest.runpod.io/v1", "rp_test"))
    assert seen[1][0] == "wait"


def test_poll_runpod_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "parhaf_clinbench.ops.poll_runpod.get_settings",
        lambda: types.SimpleNamespace(
            runpod_pod_id="pod_123",
            runpod_api_key=None,
            runpod_api_base="https://rest.runpod.io/v1",
        ),
    )
    monkeypatch.setattr(sys, "argv", ["parhaf-poll-runpod"])

    with pytest.raises(ValueError, match="RUNPOD_API_KEY est requis"):
        poll()
