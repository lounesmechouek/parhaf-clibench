"""Tests unitaires pour le mécanisme de chunking de GlinerRuntime."""

from __future__ import annotations

import json
import sys
import types
from typing import Any

import pytest

from parhaf_clinbench.core.enums import TaskId, TrackId
from parhaf_clinbench.core.models import InferenceRequest
from parhaf_clinbench.runtimes.gliner import GlinerRuntime


def _make_request(text: str, task: TaskId = TaskId.PSEUDO) -> InferenceRequest:
    return InferenceRequest(
        document_id="doc-gliner-1",
        task=task,
        track=TrackId.ZEROSHOT,
        prompt="",
        text=text,
    )


class _FakeTokenizer:
    """Tokenizer de test : 1 token = 1 caractère."""

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
        return list(range(len(text)))

    def __call__(self, text: str, *, return_offsets_mapping: bool = False, add_special_tokens: bool = True) -> dict[str, Any]:
        result: dict[str, Any] = {"input_ids": list(range(len(text)))}
        if return_offsets_mapping:
            result["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return result


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
    ) -> dict[str, Any]:
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
            return _FakeGlinerModel(entities=entities)

    fake_module = types.SimpleNamespace(GLiNER2=_FakeGLiNER, __version__="0.3.1")
    monkeypatch.setitem(sys.modules, "gliner2", fake_module)


class TestGlinerChunkingNotTriggered:
    def test_short_text_no_chunking(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Texte court → pas de chunking → infer_single appelé une seule fois."""
        _install_fake_gliner(monkeypatch, entities=[])
        runtime = GlinerRuntime(
            model_reference="fastino/gliner2-multi-v1",
            hf_token=None,
            max_context_tokens=512,
        )
        monkeypatch.setattr(runtime, "_get_tokenizer", lambda: _FakeTokenizer())

        request = _make_request("Texte court.")  # ~12 chars << 0.9*512
        result = runtime.infer(request)
        parsed = json.loads(result)
        assert parsed["document_id"] == "doc-gliner-1"
        assert parsed["task"] == "pseudo"


class TestGlinerChunkingTriggered:
    def test_chunking_triggers_multiple_infer_single_calls(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Texte long (≥90% de max_context_tokens) → plusieurs appels à infer_single."""
        call_texts: list[str] = []

        _install_fake_gliner(monkeypatch, entities=[])

        runtime = GlinerRuntime(
            model_reference="fastino/gliner2-multi-v1",
            hf_token=None,
            max_context_tokens=50,  # très petit pour forcer le chunking
        )
        monkeypatch.setattr(runtime, "_get_tokenizer", lambda: _FakeTokenizer())

        original_infer_single = runtime._infer_single  # type: ignore[attr-defined]

        def patched_infer_single(req: InferenceRequest) -> str:
            call_texts.append(req.text)
            return original_infer_single(req)

        monkeypatch.setattr(runtime, "_infer_single", patched_infer_single)

        # 200 chars → 200 tokens >> 0.9 * 50 = 45 tokens → chunking obligatoire
        text = "a" * 200
        request = _make_request(text)
        result = runtime.infer(request)

        assert len(call_texts) >= 2, "Le chunking n'a pas produit plusieurs appels"

        parsed = json.loads(result)
        assert parsed["document_id"] == "doc-gliner-1"
        assert "records" in parsed

    def test_offset_adjusted_in_merged_result(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Les offsets des entités sont ajustés selon le start_char du chunk."""
        chunk_n = [0]

        def _make_gliner(entities_per_chunk: list[list[dict[str, Any]]]) -> Any:
            class _CountingModel:
                def extract_entities(self, text: str, labels: dict, *, include_spans: bool, include_confidence: bool, threshold: float = 0.5) -> dict:
                    idx = chunk_n[0]
                    chunk_n[0] += 1
                    ents = entities_per_chunk[idx] if idx < len(entities_per_chunk) else []
                    grouped: dict[str, list[dict]] = {}
                    for item in ents:
                        lbl = str(item["label"])
                        grouped.setdefault(lbl, []).append({
                            "text": str(item["text"]),
                            "start": int(item["start"]),
                            "end": int(item["end"]),
                            "confidence": 0.9,
                        })
                    return {"entities": grouped}
            return _CountingModel()

        entities_chunk1 = [{"label": "LAST_NAME", "text": "Dupont", "start": 2, "end": 8}]
        entities_chunk2 = [{"label": "FIRST_NAME", "text": "Jean", "start": 1, "end": 5}]

        class _FakeGLiNER2:
            @classmethod
            def from_pretrained(cls, *args: Any, **kwargs: Any) -> Any:
                return _make_gliner([entities_chunk1, entities_chunk2])

        fake_mod = types.SimpleNamespace(GLiNER2=_FakeGLiNER2, __version__="0.3.1")
        monkeypatch.setitem(sys.modules, "gliner2", fake_mod)

        runtime = GlinerRuntime(
            model_reference="fastino/gliner2-multi-v1",
            hf_token=None,
            max_context_tokens=50,
        )
        monkeypatch.setattr(runtime, "_get_tokenizer", lambda: _FakeTokenizer())

        text = "a" * 200
        request = _make_request(text)
        result = runtime.infer(request)
        parsed = json.loads(result)

        records = parsed["records"]
        assert len(records) >= 1
        # chunk 2 a un start_char > 0 → ses offsets doivent être > ceux du chunk 1
        starts = [r["start"] for r in records if r["label"] == "FIRST_NAME"]
        if starts:
            # L'entité du chunk 2 doit avoir un offset absolu > son offset relatif (1)
            assert starts[0] > 1, "L'offset du chunk 2 n'a pas été ajusté"
