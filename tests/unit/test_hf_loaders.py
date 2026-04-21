from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

from parhaf_clinbench.core.enums import TaskId
from parhaf_clinbench.data.hf_loaders import load_hf_examples


def test_response_loader_merges_document_metadata_and_spans(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_load_dataset(
        path: str,
        name: str | None = None,
        *,
        revision: str | None = None,
        cache_dir: str | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        assert path == "HealthDataHub/PARHAF-response_to_treatment-annotated"
        assert revision == "rev-x"
        assert cache_dir == str(tmp_path)
        if name == "document_metadata":
            return {
                "train": [
                    {
                        "report": "R1",
                        "full_text": "Le patient a une reponse partielle objective.",
                        "attribute_Nomenclature": "ReponsePartielle",
                    }
                ]
            }
        if name == "spans":
            return {
                "train": [
                    {
                        "report": "R1",
                        "span_text": "reponse partielle",
                        "begin": 18,
                        "end": 35,
                        "attribute_Justification": "Justification",
                    }
                ]
            }
        raise AssertionError(f"Unexpected config: {name}")

    fake_module = types.SimpleNamespace(load_dataset=fake_load_dataset)
    monkeypatch.setitem(sys.modules, "datasets", fake_module)

    examples = load_hf_examples(
        task=TaskId.RESPONSE,
        dataset_name="HealthDataHub/PARHAF-response_to_treatment-annotated",
        dataset_revision="rev-x",
        cache_dir=tmp_path,
    )

    assert len(examples) == 1
    records = examples[0].gold.records
    assert len(records) == 1
    assert records[0].label == "ReponsePartielle"
    assert records[0].text == "reponse partielle"
    assert records[0].start == 18
    assert records[0].end == 35


def test_response_loader_keeps_document_label_when_no_span(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_load_dataset(
        path: str,
        name: str | None = None,
        *,
        revision: str | None = None,
        cache_dir: str | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        assert path == "HealthDataHub/PARHAF-response_to_treatment-annotated"
        assert revision == "rev-y"
        assert cache_dir == str(tmp_path)
        if name == "document_metadata":
            return {
                "train": [
                    {
                        "report": "R1",
                        "full_text": "Évaluation impossible à ce stade.",
                        "attribute_Nomenclature": "NonDetermine",
                    }
                ]
            }
        if name == "spans":
            return {"train": []}
        raise AssertionError(f"Unexpected config: {name}")

    fake_module = types.SimpleNamespace(load_dataset=fake_load_dataset)
    monkeypatch.setitem(sys.modules, "datasets", fake_module)

    examples = load_hf_examples(
        task=TaskId.RESPONSE,
        dataset_name="HealthDataHub/PARHAF-response_to_treatment-annotated",
        dataset_revision="rev-y",
        cache_dir=tmp_path,
    )

    assert len(examples) == 1
    records = examples[0].gold.records
    assert len(records) == 1
    assert records[0].label == "NonDetermine"
    assert records[0].text == ""


def test_scenario_loader_supports_documents_text_list_and_specialty(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_load_dataset(
        path: str,
        name: str | None = None,
        *,
        revision: str | None = None,
        cache_dir: str | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        assert name is None
        assert path == "HealthDataHub/PARHAF"
        assert revision == "rev-s"
        assert cache_dir == str(tmp_path)
        return {
            "train": [
                {
                    "id": "DOC-1",
                    "specialty": "CARDIOLOGIE",
                    "documents": {"text": ["Patient Alice, 76 ans."]},
                    "suggested_scenario": {
                        "name": "Alice",
                        "age": {"value": 76, "unit": "ans"},
                        "sex": "F",
                        "admission_mode": "domicile",
                        "discharge_mode": "retour domicile",
                        "primary_procedure": None,
                        "primary_diagnosis": {"description": ["insuffisance cardiaque"]},
                        "type_of_care": None,
                    },
                }
            ]
        }

    fake_module = types.SimpleNamespace(load_dataset=fake_load_dataset)
    monkeypatch.setitem(sys.modules, "datasets", fake_module)

    examples = load_hf_examples(
        task=TaskId.SCENARIO,
        dataset_name="HealthDataHub/PARHAF",
        dataset_revision="rev-s",
        cache_dir=tmp_path,
    )

    assert len(examples) == 1
    example = examples[0]
    assert example.text == "Patient Alice, 76 ans."
    assert example.speciality == "CARDIOLOGIE"
    labels = {record.label: record for record in example.gold.records}
    assert labels["name"].text == "Alice"
    assert labels["age"].text == "76 ans"
    assert "sex" not in labels
    assert "primary_diagnosis" not in labels
    assert all(record.start is not None and record.end is not None for record in example.gold.records)


def test_scenario_loader_skips_rows_without_speciality(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def fake_load_dataset(
        path: str,
        name: str | None = None,
        *,
        revision: str | None = None,
        cache_dir: str | None = None,
    ) -> dict[str, list[dict[str, Any]]]:
        assert name is None
        assert path == "HealthDataHub/PARHAF"
        assert revision == "rev-no-spec"
        assert cache_dir == str(tmp_path)
        return {
            "train": [
                {
                    "id": "DOC-NO-SPEC",
                    "documents": {"text": ["Patient Bob, 44 ans."]},
                    "suggested_scenario": {
                        "name": "Bob",
                        "age": "44 ans",
                    },
                }
            ]
        }

    fake_module = types.SimpleNamespace(load_dataset=fake_load_dataset)
    monkeypatch.setitem(sys.modules, "datasets", fake_module)

    examples = load_hf_examples(
        task=TaskId.SCENARIO,
        dataset_name="HealthDataHub/PARHAF",
        dataset_revision="rev-no-spec",
        cache_dir=tmp_path,
    )

    assert examples == []
