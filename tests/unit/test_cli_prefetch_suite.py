from __future__ import annotations

import argparse
import json
import types
from pathlib import Path
from typing import Any

import pytest

from parhaf_clinbench.core.enums import RuntimeName, TaskId
from parhaf_clinbench.data.prefetch import DatasetPrefetchResult
from parhaf_clinbench.runtimes.prefetch import PrefetchResult


def test_prefetch_suite_includes_gliner_models(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    import parhaf_clinbench.cli.main as cli_module

    suite = types.SimpleNamespace(
        suite_id="suite-test",
        models=["qwen25_7b", "gliner2_multi"],
        tasks=[TaskId.PSEUDO],
        runtime_default=RuntimeName.VLLM,
        runtime_overrides={"gliner2_multi": RuntimeName.GLINER},
    )
    task_cfg = types.SimpleNamespace(
        dataset="HealthDataHub/PARHAF-pseudo-annotated",
        dataset_revision="rev-1",
    )
    model_calls: list[str] = []

    def fake_prefetch_model(
        *,
        hf_id: str,
        revision: str,
        cache_root: Path,
        hf_token: str | None,
    ) -> PrefetchResult:
        del revision
        del cache_root
        del hf_token
        model_calls.append(hf_id)
        return PrefetchResult(
            hf_id=hf_id,
            revision="main",
            local_path=f"/workspace/models/{hf_id.replace('/', '-')}",
            cache_hit=True,
        )

    def fake_prefetch_dataset(
        *,
        dataset_name: str,
        revision: str,
        cache_root: Path,
        hf_token: str | None,
        configs: list[str] | None,
    ) -> DatasetPrefetchResult:
        del cache_root
        del hf_token
        del configs
        return DatasetPrefetchResult(
            dataset_name=dataset_name,
            revision=revision,
            local_path=f"/workspace/datasets/{dataset_name.replace('/', '-')}",
            cache_hit=True,
        )

    def fake_load_model(model_id: str) -> Any:
        return types.SimpleNamespace(model_id=model_id, hf_id=f"hf/{model_id}", revision="main")

    monkeypatch.setattr(cli_module, "get_settings", lambda: types.SimpleNamespace(hf_token="hf_test"))
    monkeypatch.setattr(cli_module, "load_suite", lambda _: suite)
    monkeypatch.setattr(cli_module, "load_task", lambda _: task_cfg)
    monkeypatch.setattr(cli_module, "load_model", fake_load_model)
    monkeypatch.setattr(cli_module, "prefetch_hf_model", fake_prefetch_model)
    monkeypatch.setattr(cli_module, "prefetch_hf_dataset", fake_prefetch_dataset)

    args = argparse.Namespace(
        suite=str(tmp_path / "unused.yaml"),
        model_cache_root=str(tmp_path / "models"),
        dataset_cache_root=str(tmp_path / "datasets"),
        output_json="",
    )

    code = cli_module._cmd_prefetch_suite(args)
    assert code == 0
    assert len(model_calls) == 2

    captured = capsys.readouterr().out
    payload = json.loads(captured)
    runtimes = {entry["model_id"]: entry["runtime"] for entry in payload["models"]}
    assert runtimes["qwen25_7b"] == RuntimeName.VLLM.value
    assert runtimes["gliner2_multi"] == RuntimeName.GLINER.value
