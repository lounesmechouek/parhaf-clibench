from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from parhaf_clinbench.core.enums import TaskId
from parhaf_clinbench.core.models import (
    INFECTIO_LABELS,
    INFECTIO_NEGATIONS,
    PSEUDO_LABELS,
    RESPONSE_LABELS,
    SCENARIO_FIELDS,
    SCENARIO_SPECIALITIES,
)
from parhaf_clinbench.data import contracts_audit


def _fake_dataset(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {"train": rows}


def test_audit_suite_contracts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def fake_load_dataset(dataset_name: str, *, revision: str, cache_dir: Path, name: str | None = None, hf_token: str | None = None) -> Any:
        del revision, cache_dir, hf_token
        if dataset_name == "ds-pseudo":
            if name == "spans":
                return _fake_dataset(
                    [
                        {
                            "attribute_Categorie": next(iter(PSEUDO_LABELS)),
                            "attribute_RolePER": "PATIENT",
                            "begin": 0,
                            "end": 1,
                            "report": "r",
                            "span_text": "x",
                        }
                    ]
                )
            return _fake_dataset([{"full_text": "t", "report": "r"}])
        if dataset_name == "ds-infectio":
            return _fake_dataset(
                [
                    {
                        "attribute_LABEL": next(iter(INFECTIO_LABELS)),
                        "attribute_NEGATION": next(iter(INFECTIO_NEGATIONS)),
                        "span_text": "x",
                    }
                ]
            )
        if dataset_name == "ds-response":
            if name == "spans":
                return _fake_dataset(
                    [
                        {
                            "attribute_Nomenclature": next(iter(RESPONSE_LABELS)),
                            "attribute_Justification": "x",
                            "span_text": "x",
                        }
                    ]
                )
            return _fake_dataset(
                [
                    {
                        "full_text": "t",
                        "report": "r",
                        "attribute_Nomenclature": next(iter(RESPONSE_LABELS)),
                    }
                ]
            )
        return _fake_dataset(
            [
                {
                    "documents": "doc",
                    "suggested_scenario": {field: "x" for field in SCENARIO_FIELDS},
                    "speciality": next(iter(SCENARIO_SPECIALITIES)),
                }
            ]
        )

    def fake_load_suite(_path: Path) -> Any:
        return SimpleNamespace(suite_id="suite", tasks=[TaskId.PSEUDO, TaskId.INFECTIO, TaskId.RESPONSE, TaskId.SCENARIO])

    def fake_load_task(task: TaskId) -> Any:
        return SimpleNamespace(dataset=f"ds-{task.value}", dataset_revision="rev")

    monkeypatch.setattr(contracts_audit, "_load_dataset", fake_load_dataset)
    monkeypatch.setattr(contracts_audit, "load_suite", fake_load_suite)
    monkeypatch.setattr(contracts_audit, "load_task", fake_load_task)
    monkeypatch.setattr(contracts_audit, "resolve_local_dataset_path", lambda root, dataset, revision: root / dataset / revision)

    report = contracts_audit.audit_suite_contracts(
        suite_path=tmp_path / "suite.yaml",
        dataset_cache_root=tmp_path,
        hf_token=None,
    )

    assert report.suite_id == "suite"
    assert report.all_ok is False
    assert len(report.tasks) == 4
