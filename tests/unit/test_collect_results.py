from __future__ import annotations

from pathlib import Path

import pytest

from parhaf_clinbench.ops.collect_results import (
    _build_rsync_cmd,
    _build_ssh_cmd,
    _is_valid_downloaded_file,
    _validate_ssh_target,
    _validate_transport_for_target,
)


def test_validate_ssh_target_rejects_double_at() -> None:
    with pytest.raises(ValueError, match="Format --ssh-target invalide"):
        _validate_ssh_target("root@abc@ssh.runpod.io")


def test_build_rsync_cmd_includes_ssh_port_and_identity_file(tmp_path: Path) -> None:
    cmd = _build_rsync_cmd(
        ssh_target="root@1.2.3.4",
        remote_path="/workspace/results.tar.zst",
        destination=tmp_path / "results.tar.zst",
        ssh_port=2222,
        identity_file="/home/user/.ssh/id_ed25519",
    )
    assert cmd[:4] == ["rsync", "-avz", "-e", "ssh -p 2222 -i /home/user/.ssh/id_ed25519"]
    assert cmd[4] == "root@1.2.3.4:/workspace/results.tar.zst"


def test_build_ssh_cmd_includes_ssh_port_and_identity_file() -> None:
    cmd = _build_ssh_cmd(
        ssh_target="user@host",
        ssh_port=2222,
        identity_file="/home/user/.ssh/id_ed25519",
        remote_cmd="exec cat /workspace/results.tar.zst",
    )
    assert cmd[0] == "ssh"
    assert "-p" in cmd and "2222" in cmd
    assert "-i" in cmd and "/home/user/.ssh/id_ed25519" in cmd
    assert "BatchMode=yes" in cmd
    assert "StrictHostKeyChecking=accept-new" in cmd
    assert cmd[-2:] == ["user@host", "exec cat /workspace/results.tar.zst"]


def test_is_valid_downloaded_file_checks_zstd_magic(tmp_path: Path) -> None:
    file_path = tmp_path / "results.tar.zst"
    file_path.write_bytes(b"Error: not a binary archive")
    assert _is_valid_downloaded_file(file_path, "/workspace/results.tar.zst") is False
    file_path.write_bytes(b"\x28\xb5\x2f\xfdABC")
    assert _is_valid_downloaded_file(file_path, "/workspace/results.tar.zst") is True


def test_validate_transport_for_target_rejects_ssh_cat_on_runpod_proxy() -> None:
    with pytest.raises(ValueError, match="n'est pas compatible"):
        _validate_transport_for_target("abc123@ssh.runpod.io", "ssh-cat")
