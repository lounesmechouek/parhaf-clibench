from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import ui.data_loader as loader


def test_load_manifest_and_parquet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    monkeypatch.setattr(loader, "DATA_DIR", data_dir)

    assert loader.load_manifest() == {}

    payload = {"n_models": 2}
    (data_dir / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    loader.load_manifest.clear()
    assert loader.load_manifest()["n_models"] == 2

    df = pd.DataFrame([{"model": "m1", "f1": 0.4}])
    df.to_parquet(data_dir / "scores.parquet")
    out = loader.load_scores()
    assert not out.empty
