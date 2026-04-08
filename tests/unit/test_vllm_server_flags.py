"""Tests for vLLM server launch flag injection in _managed_vllm_server."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

import parhaf_clinbench.orchestration.runner as runner_module


def _make_fake_process(already_exited: bool = True) -> MagicMock:
    """Return a mock subprocess.Popen whose poll() signals it has already exited."""
    proc = MagicMock(spec=subprocess.Popen)
    proc.pid = 99999
    proc.returncode = 0
    # poll() returns 0 (not None) → the finally-block skips killpg
    proc.poll.return_value = 0 if already_exited else None
    proc.wait.return_value = 0
    return proc


def _base_payload() -> dict[str, Any]:
    return {
        "api_base": "http://127.0.0.1:8000/v1",
        "healthcheck_url": "http://127.0.0.1:8000/health",
        "startup_timeout_seconds": 60,
    }


def _run_server(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    runtime_payload: dict[str, Any],
    max_model_len: int | None = None,
) -> list[str]:
    """Start and immediately exit the managed server; return the captured cmd."""
    captured: list[list[str]] = []
    fake_proc = _make_fake_process()

    def fake_popen(cmd: list[str], **kwargs: Any) -> MagicMock:
        captured.append(list(cmd))
        return fake_proc

    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(runner_module, "_wait_http_ready", lambda *a, **kw: None)

    log_path = tmp_path / "vllm_server.log"
    logger = logging.getLogger("test_vllm_server_flags")

    with runner_module._managed_vllm_server(
        model_reference="test-model",
        runtime_payload=runtime_payload,
        logger=logger,
        log_path=log_path,
        max_model_len=max_model_len,
    ):
        pass

    assert captured, "Popen was never called"
    return captured[0]


def test_extra_flags_max_num_seqs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {**_base_payload(), "max_num_seqs": 64}
    cmd = _run_server(monkeypatch, tmp_path, payload)
    assert "--max-num-seqs" in cmd
    assert "64" in cmd


def test_extra_flags_gpu_memory_utilization(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {**_base_payload(), "gpu_memory_utilization": 0.92}
    cmd = _run_server(monkeypatch, tmp_path, payload)
    assert "--gpu-memory-utilization" in cmd
    assert "0.92" in cmd


def test_extra_flags_disable_log_requests(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {**_base_payload(), "disable_log_requests": True}
    cmd = _run_server(monkeypatch, tmp_path, payload)
    assert "--disable-log-requests" in cmd


def test_extra_flags_enable_chunked_prefill(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {**_base_payload(), "enable_chunked_prefill": True}
    cmd = _run_server(monkeypatch, tmp_path, payload)
    assert "--enable-chunked-prefill" in cmd


def test_extra_flags_all_together(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = {
        **_base_payload(),
        "max_num_seqs": 128,
        "gpu_memory_utilization": 0.95,
        "disable_log_requests": True,
        "enable_chunked_prefill": True,
    }
    cmd = _run_server(monkeypatch, tmp_path, payload)
    assert "--max-num-seqs" in cmd
    assert "128" in cmd
    assert "--gpu-memory-utilization" in cmd
    assert "0.95" in cmd
    assert "--disable-log-requests" in cmd
    assert "--enable-chunked-prefill" in cmd


def test_false_flags_not_in_cmd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Flags with value=False must not be added to the command."""
    payload = {
        **_base_payload(),
        "disable_log_requests": False,
        "enable_chunked_prefill": False,
    }
    cmd = _run_server(monkeypatch, tmp_path, payload)
    assert "--disable-log-requests" not in cmd
    assert "--enable-chunked-prefill" not in cmd


def test_absent_flags_not_in_cmd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Flags absent from the payload must not appear in the command."""
    cmd = _run_server(monkeypatch, tmp_path, _base_payload())
    assert "--max-num-seqs" not in cmd
    assert "--gpu-memory-utilization" not in cmd
    assert "--disable-log-requests" not in cmd
    assert "--enable-chunked-prefill" not in cmd


def test_max_model_len_still_included(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Existing --max-model-len flag must coexist with new extra flags."""
    payload = {**_base_payload(), "max_num_seqs": 32}
    cmd = _run_server(monkeypatch, tmp_path, payload, max_model_len=8192)
    assert "--max-model-len" in cmd
    assert "8192" in cmd
    assert "--max-num-seqs" in cmd
    assert "32" in cmd
