"""Tests unitaires pour parhaf_clinbench.chunking.splitter."""

from __future__ import annotations

from typing import Any

from parhaf_clinbench.chunking.splitter import TextChunk, make_chunks


class _FakeEncoding:
    """Simule le retour d'un fast tokenizer HuggingFace."""

    def __init__(self, text: str) -> None:
        # Tokenisation naïve : un token = un caractère (pour simplifier les tests).
        self._chars = list(text)

    def get(self, key: str, default: Any = None) -> Any:
        if key == "offset_mapping":
            return [(i, i + 1) for i in range(len(self._chars))]
        if key == "input_ids":
            return list(range(len(self._chars)))
        return default

    def __len__(self) -> int:
        return len(self._chars)


def _fake_tokenizer(text: str, *, return_offsets_mapping: bool = False, add_special_tokens: bool = True) -> _FakeEncoding:
    return _FakeEncoding(text)


def _slow_tokenizer(text: str, *, return_offsets_mapping: bool = False, add_special_tokens: bool = True) -> Any:
    """Simule un slow tokenizer sans offset_mapping."""
    enc = _FakeEncoding(text)
    if return_offsets_mapping:
        # Pas d'offset_mapping dans l'encodage retourné.
        class _NoOffsets:
            def get(self, key: str, default: Any = None) -> Any:
                if key == "input_ids":
                    return list(range(len(text)))
                return default  # offset_mapping absent

        return _NoOffsets()
    return enc


class TestMakeChunksNoSplit:
    def test_short_text_returns_single_chunk(self) -> None:
        text = "abc"
        chunks = make_chunks(text, _fake_tokenizer, max_tokens=10)
        assert len(chunks) == 1
        assert chunks[0] == TextChunk(text=text, start_char=0)

    def test_exact_limit_returns_single_chunk(self) -> None:
        text = "abcde"  # 5 chars = 5 tokens avec fake tokenizer
        chunks = make_chunks(text, _fake_tokenizer, max_tokens=5)
        assert len(chunks) == 1
        assert chunks[0].start_char == 0

    def test_empty_text(self) -> None:
        chunks = make_chunks("", _fake_tokenizer, max_tokens=10)
        assert len(chunks) == 1
        assert chunks[0].text == ""
        assert chunks[0].start_char == 0


class TestMakeChunksSplit:
    def test_two_chunks_produced(self) -> None:
        # overlap_ratio=0.0 → overlap = max(1, 0) = 1 (minimum pour éviter stride=0)
        # stride = 6 - 1 = 5 → chunks démarrent à 0 et 5
        text = "abcdefghij"  # 10 tokens
        chunks = make_chunks(text, _fake_tokenizer, max_tokens=6, overlap_ratio=0.0)
        assert len(chunks) == 2
        assert chunks[0].start_char == 0
        assert chunks[0].text == text[:6]
        assert chunks[1].start_char == 5
        assert chunks[1].text == text[5:]

    def test_overlap_creates_third_chunk(self) -> None:
        # 12 chars, max_tokens=6, overlap=2 (33%), stride=4
        text = "abcdefghijkl"
        chunks = make_chunks(text, _fake_tokenizer, max_tokens=6, overlap_ratio=0.33)
        # stride=4 → starts: 0, 4, 8
        assert len(chunks) >= 2
        # Dernier chunk doit atteindre la fin du texte
        assert chunks[-1].start_char + len(chunks[-1].text) == len(text)

    def test_chunks_cover_full_text(self) -> None:
        text = "abcdefghijklmnopqrstu"
        chunks = make_chunks(text, _fake_tokenizer, max_tokens=7, overlap_ratio=0.2)
        # Vérifier que le premier chunk commence à 0 et le dernier couvre la fin
        assert chunks[0].start_char == 0
        last = chunks[-1]
        assert last.start_char + len(last.text) == len(text)

    def test_start_chars_increasing(self) -> None:
        text = "x" * 30
        chunks = make_chunks(text, _fake_tokenizer, max_tokens=10, overlap_ratio=0.1)
        starts = [c.start_char for c in chunks]
        assert starts == sorted(starts)
        assert len(set(starts)) == len(starts)  # tous distincts

    def test_chunk_text_matches_original(self) -> None:
        text = "hello world foo bar baz"
        chunks = make_chunks(text, _fake_tokenizer, max_tokens=8, overlap_ratio=0.2)
        for chunk in chunks:
            assert text[chunk.start_char : chunk.start_char + len(chunk.text)] == chunk.text


class TestMakeChunksSlowTokenizerFallback:
    def test_fallback_produces_chunks(self) -> None:
        text = "a" * 200
        chunks = make_chunks(text, _slow_tokenizer, max_tokens=50)
        assert len(chunks) >= 1
        # Le texte doit être intégralement couvert
        first_start = chunks[0].start_char
        assert first_start == 0

    def test_fallback_single_chunk_for_short_text(self) -> None:
        # Un slow tokenizer qui ne supporte pas offset_mapping mais texte court
        chunks = make_chunks("bonjour", _slow_tokenizer, max_tokens=1000)
        assert len(chunks) == 1
