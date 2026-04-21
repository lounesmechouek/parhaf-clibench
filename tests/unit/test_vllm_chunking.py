"""Tests unitaires pour le mécanisme de chunking de VllmRuntime."""

from __future__ import annotations

import json
from typing import Any

import pytest

from parhaf_clinbench.core.enums import TaskId, TrackId
from parhaf_clinbench.core.models import InferenceRequest
from parhaf_clinbench.runtimes.vllm import VllmRuntime

# Prompt template minimal avec les marqueurs obligatoires.
_PROMPT_TMPL = "Instructions.\n<<<BEGIN_TEXT>>>\n{text}\n<<<END_TEXT>>>\nExtraire."


def _make_request(text: str, task: TaskId = TaskId.PSEUDO) -> InferenceRequest:
    return InferenceRequest(
        document_id="doc-chunk-1",
        task=task,
        track=TrackId.ZEROSHOT,
        prompt=_PROMPT_TMPL.format(text=text),
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

    def get(self, key: str, default: Any = None) -> Any:
        return default


class _FakeResponse:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


def _vllm_response(records: list[dict[str, Any]], document_id: str = "doc-chunk-1") -> dict[str, Any]:
    return {
        "choices": [
            {
                "message": {
                    "content": json.dumps({
                        "document_id": document_id,
                        "task": "pseudo",
                        "records": records,
                    })
                }
            }
        ]
    }


class TestVllmChunkingNotTriggered:
    def test_short_prompt_no_chunking(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Prompt court → pas de chunking → un seul appel API."""
        call_count = [0]

        def fake_post(url: str, *, json: Any, timeout: int) -> _FakeResponse:
            call_count[0] += 1
            return _FakeResponse(_vllm_response([]))

        monkeypatch.setattr("requests.post", fake_post)

        # max_context_tokens=10000, prompt ~30 chars, max_tokens=100 → bien en dessous des 90%
        runtime = VllmRuntime(
            api_base="http://127.0.0.1:8000/v1",
            model_hf_id="test/model",
            max_tokens=100,
            max_context_tokens=10000,
        )
        monkeypatch.setattr(runtime, "_get_tokenizer", lambda: _FakeTokenizer())

        request = _make_request("Texte court.")
        runtime.infer(request)
        assert call_count[0] == 1


class TestVllmChunkingTriggered:
    def _make_long_text(self, n_chars: int) -> str:
        # Texte de n_chars caractères avec des espaces pour le découpage propre
        word = "abcde "  # 6 chars
        repeats = n_chars // len(word) + 1
        return (word * repeats)[:n_chars]

    def test_chunking_triggers_multiple_api_calls(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Prompt long (≥90% du contexte) → plusieurs appels API."""
        call_count = [0]
        captured_texts: list[str] = []

        def fake_post(url: str, *, json: Any, timeout: int) -> _FakeResponse:
            call_count[0] += 1
            # Extraire le texte du message pour vérifier les chunks
            content = json["messages"][0]["content"]
            captured_texts.append(content)
            return _FakeResponse(_vllm_response([]))

        monkeypatch.setattr("requests.post", fake_post)

        # max_context_tokens=100, max_tokens=10
        # threshold = 0.9 * 100 = 90 tokens
        # Avec _FakeTokenizer (1 char = 1 token), un prompt de 85+ chars déclenche le chunking
        runtime = VllmRuntime(
            api_base="http://127.0.0.1:8000/v1",
            model_hf_id="test/model",
            max_tokens=10,
            max_context_tokens=100,
        )
        monkeypatch.setattr(runtime, "_get_tokenizer", lambda: _FakeTokenizer())

        # Texte de 200 chars → prompt total bien au-dessus de 90
        text = self._make_long_text(200)
        request = _make_request(text)
        runtime.infer(request)
        assert call_count[0] >= 2

    def test_merged_output_is_valid_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """La sortie mergée est un JSON CanonicalDocument valide."""
        chunk_n = [0]

        def fake_post(url: str, *, json: Any, timeout: int) -> _FakeResponse:
            chunk_n[0] += 1
            # Chunk 1 contient une entité à offset 0, chunk 2 en contient une autre
            records = []
            if chunk_n[0] == 1:
                records = [{"label": "LAST_NAME", "text": "Dupont", "start": 0, "end": 6, "attributes": {}}]
            elif chunk_n[0] == 2:
                records = [{"label": "FIRST_NAME", "text": "Jean", "start": 0, "end": 4, "attributes": {}}]
            return _FakeResponse(_vllm_response(records))

        monkeypatch.setattr("requests.post", fake_post)

        runtime = VllmRuntime(
            api_base="http://127.0.0.1:8000/v1",
            model_hf_id="test/model",
            max_tokens=10,
            max_context_tokens=100,
        )
        monkeypatch.setattr(runtime, "_get_tokenizer", lambda: _FakeTokenizer())

        text = "a" * 200
        request = _make_request(text)
        result = runtime.infer(request)

        parsed = json.loads(result)
        assert parsed["document_id"] == "doc-chunk-1"
        assert parsed["task"] == "pseudo"
        assert isinstance(parsed["records"], list)
        assert len(parsed["records"]) >= 1

    def test_duplicate_in_overlap_removed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Même entité dans deux chunks successifs → conservée une seule fois."""
        def fake_post(url: str, *, json: Any, timeout: int) -> _FakeResponse:
            # Tous les chunks retournent la même entité au même offset relatif 0→6
            records = [{"label": "LAST_NAME", "text": "Dupont", "start": 0, "end": 6, "attributes": {}}]
            return _FakeResponse(_vllm_response(records))

        monkeypatch.setattr("requests.post", fake_post)

        runtime = VllmRuntime(
            api_base="http://127.0.0.1:8000/v1",
            model_hf_id="test/model",
            max_tokens=10,
            max_context_tokens=100,
        )

        class _SameOffsetTokenizer:
            """Simule des chunks qui produisent des offsets identiques (zone overlap)."""
            _call = 0

            def encode(self, text: str, *, add_special_tokens: bool = True) -> list[int]:
                return list(range(len(text)))

            def __call__(self, text: str, *, return_offsets_mapping: bool = False, add_special_tokens: bool = True) -> dict[str, Any]:
                result: dict[str, Any] = {"input_ids": list(range(len(text)))}
                if return_offsets_mapping:
                    result["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
                return result

        monkeypatch.setattr(runtime, "_get_tokenizer", lambda: _SameOffsetTokenizer())

        text = "a" * 200
        request = _make_request(text)
        result = runtime.infer(request)
        parsed = json.loads(result)
        # L'entité à (0, 6, LAST_NAME) ne doit apparaître qu'une fois dans la sortie finale
        # (même si plusieurs chunks retournent le même offset absolu après ajustement)
        records = parsed["records"]
        keys = [(r["start"], r["end"], r["label"]) for r in records]
        assert len(keys) == len(set(keys)), "Duplicates not removed after merge"

    def test_empty_result_when_all_chunks_fail(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Si tous les chunks échouent au parse, retourne chaîne vide."""
        def fake_post(url: str, *, json: Any, timeout: int) -> _FakeResponse:
            return _FakeResponse({"choices": [{"message": {"content": "invalid json!!!"}}]})

        monkeypatch.setattr("requests.post", fake_post)

        runtime = VllmRuntime(
            api_base="http://127.0.0.1:8000/v1",
            model_hf_id="test/model",
            max_tokens=10,
            max_context_tokens=100,
        )
        monkeypatch.setattr(runtime, "_get_tokenizer", lambda: _FakeTokenizer())

        text = "a" * 200
        request = _make_request(text)
        result = runtime.infer(request)
        assert result == ""
