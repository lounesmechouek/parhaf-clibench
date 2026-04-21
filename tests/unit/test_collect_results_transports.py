from __future__ import annotations

import base64
import subprocess
from pathlib import Path
from typing import Any

import pytest

from parhaf_clinbench.ops.collect_results import _copy_via_ssh_cat, _copy_via_ssh_pty_base64


class _Completed:
    def __init__(
        self,
        *,
        returncode: int = 0,
        stdout: bytes = b"",
        stderr: bytes = b"",
    ) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


def test_copy_via_ssh_pty_base64_success(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    payload = b"hello-archive"
    encoded = base64.b64encode(payload).decode("ascii")
    stdout = f"noise__PARHAF_BEGIN__{encoded}__PARHAF_END__noise".encode()

    def fake_run(*args: Any, **kwargs: Any) -> _Completed:
        del args
        del kwargs
        return _Completed(returncode=0, stdout=stdout)

    monkeypatch.setattr("parhaf_clinbench.ops.collect_results.subprocess.run", fake_run)
    destination = tmp_path / "results.tar.zst"

    code = _copy_via_ssh_pty_base64(
        ssh_target="root@1.2.3.4",
        ssh_port=22,
        identity_file=None,
        remote_path="/workspace/results.tar.zst",
        destination=destination,
        timeout_seconds=10,
    )

    assert code == 0
    assert destination.read_bytes() == payload


def test_copy_via_ssh_pty_base64_rejects_missing_markers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_run(*args: Any, **kwargs: Any) -> _Completed:
        del args
        del kwargs
        return _Completed(returncode=0, stdout=b"no-markers-here")

    monkeypatch.setattr("parhaf_clinbench.ops.collect_results.subprocess.run", fake_run)
    destination = tmp_path / "results.tar.zst"
    destination.write_bytes(b"stale")

    code = _copy_via_ssh_pty_base64(
        ssh_target="root@1.2.3.4",
        ssh_port=22,
        identity_file=None,
        remote_path="/workspace/results.tar.zst",
        destination=destination,
        timeout_seconds=10,
    )

    assert code == 98
    assert destination.exists() is False


def test_copy_via_ssh_cat_timeout_cleans_partial_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_run(*args: Any, **kwargs: Any) -> _Completed:
        del args
        del kwargs
        raise subprocess.TimeoutExpired(cmd="ssh", timeout=10)

    monkeypatch.setattr("parhaf_clinbench.ops.collect_results.subprocess.run", fake_run)
    destination = tmp_path / "results.tar.zst"

    code = _copy_via_ssh_cat(
        ssh_target="root@1.2.3.4",
        ssh_port=22,
        identity_file=None,
        remote_path="/workspace/results.tar.zst",
        destination=destination,
        timeout_seconds=10,
    )

    assert code == 124
    assert destination.exists() is False
