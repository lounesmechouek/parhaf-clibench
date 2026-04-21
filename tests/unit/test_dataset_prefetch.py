from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from parhaf_clinbench.data.prefetch import prefetch_hf_dataset


def test_dataset_prefetch_idempotent_cache_hit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str | None, str | None, str | None]] = []

    def fake_get_dataset_config_names(dataset_name: str, revision: str | None = None) -> list[str]:
        assert dataset_name == "HealthDataHub/PARHAF-pseudo-annotated"
        assert revision == "rev-123"
        return ["document_metadata", "spans"]

    def fake_load_dataset(
        path: str,
        name: str | None = None,
        *,
        revision: str | None = None,
        cache_dir: str | None = None,
        token: str | None = None,
    ) -> dict[str, list[dict[str, str]]]:
        calls.append((path, name, revision, cache_dir))
        assert token == "hf_test_token"
        if cache_dir is not None:
            cache_path = Path(cache_dir)
            cache_path.mkdir(parents=True, exist_ok=True)
            suffix = name or "default"
            (cache_path / f"dataset-{suffix}.arrow").write_text("x", encoding="utf-8")
        return {"train": [{"id": "x"}]}

    fake_module = types.SimpleNamespace(
        get_dataset_config_names=fake_get_dataset_config_names,
        load_dataset=fake_load_dataset,
    )
    monkeypatch.setitem(sys.modules, "datasets", fake_module)

    first = prefetch_hf_dataset(
        dataset_name="HealthDataHub/PARHAF-pseudo-annotated",
        revision="rev-123",
        cache_root=tmp_path,
        hf_token="hf_test_token",
    )
    second = prefetch_hf_dataset(
        dataset_name="HealthDataHub/PARHAF-pseudo-annotated",
        revision="rev-123",
        cache_root=tmp_path,
        hf_token="hf_test_token",
    )

    assert first.cache_hit is False
    assert second.cache_hit is True
    assert len(calls) == 2
    assert {calls[0][1], calls[1][1]} == {"document_metadata", "spans"}


def test_dataset_prefetch_wraps_download_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_get_dataset_config_names(dataset_name: str, revision: str | None = None) -> list[str]:
        return ["document_metadata"]

    def fake_load_dataset(
        path: str,
        name: str | None = None,
        *,
        revision: str | None = None,
        cache_dir: str | None = None,
        token: str | None = None,
    ) -> dict[str, list[dict[str, str]]]:
        raise RuntimeError("401 Unauthorized")

    fake_module = types.SimpleNamespace(
        get_dataset_config_names=fake_get_dataset_config_names,
        load_dataset=fake_load_dataset,
    )
    monkeypatch.setitem(sys.modules, "datasets", fake_module)

    with pytest.raises(RuntimeError) as exc_info:
        prefetch_hf_dataset(
            dataset_name="HealthDataHub/PARHAF-pseudo-annotated",
            revision="rev-123",
            cache_root=tmp_path,
            hf_token="hf_test_token",
        )

    message = str(exc_info.value)
    assert "load_dataset" in message
    assert "HealthDataHub/PARHAF-pseudo-annotated" in message
    assert "revision=rev-123" in message
    assert "Check HF_TOKEN access" in message
