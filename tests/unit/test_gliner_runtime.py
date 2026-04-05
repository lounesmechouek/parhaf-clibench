from __future__ import annotations

import json
import sys
import types
from typing import Any

import pytest

from parhaf_clinbench.core.enums import TaskId, TrackId
from parhaf_clinbench.core.models import CanonicalDocument, InferenceRequest
from parhaf_clinbench.parsing.validate import validate_and_parse
from parhaf_clinbench.runtimes.gliner import GlinerRuntime


class _FakeGlinerModel:
    def __init__(self, entities: list[dict[str, Any]]) -> None:
        self._entities = entities

    def extract_entities(
        self,
        text: str,
        labels: dict[str, str],
        *,
        include_spans: bool,
        include_confidence: bool,
        threshold: float = 0.5,
    ) -> dict[str, dict[str, list[dict[str, Any]]]]:
        assert isinstance(text, str)
        assert isinstance(labels, dict)
        assert labels
        assert include_spans is True
        assert include_confidence is True
        assert isinstance(threshold, float)
        grouped: dict[str, list[dict[str, Any]]] = {}
        for item in self._entities:
            label = str(item["label"])
            payload = {
                "text": str(item["text"]),
                "start": int(item["start"]),
                "end": int(item["end"]),
                "confidence": float(item.get("score", 0.9)),
            }
            grouped.setdefault(label, []).append(payload)
        return {"entities": grouped}


def _install_fake_gliner(
    monkeypatch: pytest.MonkeyPatch,
    *,
    entities: list[dict[str, Any]],
) -> None:
    class _FakeGLiNER:
        @classmethod
        def from_pretrained(cls, *args: Any, **kwargs: Any) -> _FakeGlinerModel:
            assert kwargs.get("local_files_only") is True
            return _FakeGlinerModel(entities=entities)

    fake_module = types.SimpleNamespace(GLiNER2=_FakeGLiNER, __version__="0.3.1")
    monkeypatch.setitem(sys.modules, "gliner2", fake_module)


@pytest.mark.parametrize(
    ("task", "text", "span_text", "entity_label", "expected_label", "speciality"),
    [
        (
            TaskId.PSEUDO,
            "Le patient Martin est vu en consultation.",
            "Martin",
            "last_name",
            "LAST_NAME",
            None,
        ),
        (
            TaskId.INFECTIO,
            "Infection pulmonaire documentée.",
            "Infection",
            "infection",
            "Infection",
            None,
        ),
        (
            TaskId.RESPONSE,
            "Progression hépatique objectivée au scanner.",
            "Progression",
            "progression",
            "MaladieProgressive",
            None,
        ),
        (
            TaskId.SCENARIO,
            "Admission pour pneumonie en pneumologie.",
            "pneumonie",
            "primary diagnosis",
            "primary_diagnosis",
            "PNEUMOLOGIE",
        ),
    ],
)
def test_gliner_runtime_returns_schema_valid_json(
    monkeypatch: pytest.MonkeyPatch,
    task: TaskId,
    text: str,
    span_text: str,
    entity_label: str,
    expected_label: str,
    speciality: str | None,
) -> None:
    start = text.find(span_text)
    assert start != -1
    end = start + len(span_text)

    _install_fake_gliner(
        monkeypatch,
        entities=[
            {
                "label": entity_label,
                "text": span_text,
                "start": start,
                "end": end,
                "score": 0.9,
            }
        ],
    )

    runtime = GlinerRuntime(
        model_reference="/workspace/models/fake",
        hf_token=None,
        device="cpu",
        threshold=0.5,
        flat_ner=True,
        multi_label=False,
        batch_size=4,
        negation_window_chars=32,
    )
    gold = CanonicalDocument(
        document_id="doc-1",
        task=task,
        speciality=speciality if task == TaskId.SCENARIO else None,
        records=[],
    )
    request = InferenceRequest(
        document_id="doc-1",
        task=task,
        track=TrackId.ZEROSHOT,
        prompt="unused-for-gliner",
        text=text,
        gold=gold,
    )

    output = runtime.infer(request)
    parsed, raw_json_valid, repair_applied, schema_valid, error = validate_and_parse(output, task)

    assert raw_json_valid is True
    assert repair_applied is False
    assert schema_valid is True
    assert error is None
    assert parsed is not None
    assert parsed.records
    assert parsed.records[0].label == expected_label
    if task == TaskId.SCENARIO:
        assert parsed.speciality == "PNEUMOLOGIE"
    runtime.close()


def test_gliner_runtime_infectio_negation_heuristic(monkeypatch: pytest.MonkeyPatch) -> None:
    text = "Pas de foyer pulmonaire, bactériémie possible."
    site_text = "foyer pulmonaire"
    bact_text = "bactériémie"
    _install_fake_gliner(
        monkeypatch,
        entities=[
            {
                "label": "Site",
                "text": site_text,
                "start": text.find(site_text),
                "end": text.find(site_text) + len(site_text),
                "score": 0.8,
            },
            {
                "label": "Bacteriemie",
                "text": bact_text,
                "start": text.find(bact_text),
                "end": text.find(bact_text) + len(bact_text),
                "score": 0.8,
            },
        ],
    )

    runtime = GlinerRuntime(
        model_reference="/workspace/models/fake",
        hf_token=None,
        device="cpu",
        threshold=0.5,
        flat_ner=True,
        multi_label=False,
        batch_size=4,
        negation_window_chars=24,
    )
    request = InferenceRequest(
        document_id="doc-inf",
        task=TaskId.INFECTIO,
        track=TrackId.ZEROSHOT,
        prompt="unused-for-gliner",
        text=text,
        gold=CanonicalDocument(document_id="doc-inf", task=TaskId.INFECTIO, records=[]),
    )

    payload = json.loads(runtime.infer(request))
    records = payload["records"]
    negations = {item["label"]: item["attributes"].get("negation") for item in records}
    assert negations["Site"] == "Absent"
    assert negations["Bacteriemie"] == "Indetermine"
    runtime.close()
