from __future__ import annotations

import errno
import sys
import types
from pathlib import Path

import pytest

from parhaf_clinbench.runtimes.prefetch import prefetch_hf_model


def test_prefetch_idempotent_cache_hit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    def fake_snapshot_download(**kwargs: object) -> None:
        local_dir = kwargs["local_dir"]
        calls.append(str(local_dir))
        Path(str(local_dir)).mkdir(parents=True, exist_ok=True)
        (Path(str(local_dir)) / "model.safetensors").write_text("x", encoding="utf-8")

    fake_module = types.SimpleNamespace(snapshot_download=fake_snapshot_download)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)

    first = prefetch_hf_model(
        hf_id="Qwen/Qwen2.5-7B-Instruct",
        revision="main",
        cache_root=tmp_path,
        hf_token=None,
    )
    second = prefetch_hf_model(
        hf_id="Qwen/Qwen2.5-7B-Instruct",
        revision="main",
        cache_root=tmp_path,
        hf_token=None,
    )

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert len(calls) == 1


def test_prefetch_wraps_download_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_snapshot_download(**kwargs: object) -> None:
        raise OSError(errno.ENOSPC, "No space left on device")

    fake_module = types.SimpleNamespace(snapshot_download=fake_snapshot_download)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_module)

    with pytest.raises(RuntimeError) as exc_info:
        prefetch_hf_model(
            hf_id="Qwen/Qwen2.5-7B-Instruct",
            revision="main",
            cache_root=tmp_path,
            hf_token=None,
        )

    message = str(exc_info.value)
    assert "snapshot_download" in message
    assert "Qwen/Qwen2.5-7B-Instruct" in message
    assert "revision=main" in message
    assert "insufficient disk space" in message
