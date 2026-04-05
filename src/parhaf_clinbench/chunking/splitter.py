"""Split long texts into overlapping sliding-window chunks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TextChunk:
    """Text fragment and its character offset in the source document."""

    text: str
    start_char: int  # NOTE: Offset of the first character in the source text.


def make_chunks(
    text: str,
    tokenizer: Any,
    max_tokens: int,
    overlap_ratio: float = 0.15,
) -> list[TextChunk]:
    """Split `text` into overlapping chunks constrained by token budget.

    The function first tries to use tokenizer `offset_mapping` for exact
    character-aligned chunk boundaries. If offsets are not available, it falls
    back to a proportional character-based strategy.

    Args:
        text: Full source text.
        tokenizer: Tokenizer object exposing a callable interface.
        max_tokens: Maximum number of tokens per chunk.
        overlap_ratio: Fraction of overlap between consecutive chunks.

    Returns:
        Chunks sorted by ascending `start_char`. If one chunk is enough, returns
        `[TextChunk(text=text, start_char=0)]`.

    Examples:
        >>> chunks = make_chunks("alpha beta gamma", tokenizer, max_tokens=2)
        >>> chunks[0].start_char
        0
    """
    if not text:
        return [TextChunk(text="", start_char=0)]

    overlap = max(1, int(max_tokens * overlap_ratio))
    stride = max(1, max_tokens - overlap)

    try:
        encoding = tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        offset_mapping: list[tuple[int, int]] | None = encoding.get("offset_mapping")
        if offset_mapping is None:
            raise ValueError("offset_mapping absent")
    except Exception:
        offset_mapping = None

    if offset_mapping is not None:
        return _chunks_from_offset_mapping(text, offset_mapping, max_tokens, stride)

    # NOTE: Fallback to proportional character-based splitting.
    return _chunks_proportional(text, tokenizer, max_tokens, stride)


def _chunks_from_offset_mapping(
    text: str,
    offset_mapping: list[tuple[int, int]],
    max_tokens: int,
    stride: int,
) -> list[TextChunk]:
    """Build chunks from tokenizer offsets with exact character boundaries."""

    n_tokens = len(offset_mapping)
    if n_tokens <= max_tokens:
        return [TextChunk(text=text, start_char=0)]

    chunks: list[TextChunk] = []
    start_tok = 0
    while start_tok < n_tokens:
        end_tok = min(start_tok + max_tokens, n_tokens)
        char_start = offset_mapping[start_tok][0]
        char_end = offset_mapping[end_tok - 1][1]
        chunks.append(TextChunk(text=text[char_start:char_end], start_char=char_start))
        if end_tok == n_tokens:
            break
        start_tok += stride

    return chunks


def _chunks_proportional(
    text: str,
    tokenizer: Any,
    max_tokens: int,
    stride: int,
) -> list[TextChunk]:
    """Estimate chunk boundaries from an approximate token-to-character ratio."""
    words = text.split()
    if not words:
        return [TextChunk(text=text, start_char=0)]

    # NOTE: Estimate token/character ratio from a representative text sample.
    sample = text[:min(len(text), 2000)]
    try:
        sample_tok = tokenizer(sample, add_special_tokens=False)
        ratio = len(sample_tok["input_ids"]) / max(len(sample), 1)
    except Exception:
        ratio = 1.0 / 4  # NOTE: Rough default: 4 characters per token.

    max_chars = int(max_tokens / max(ratio, 1e-6))
    stride_chars = int(stride / max(ratio, 1e-6))

    chunks: list[TextChunk] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        # NOTE: Align to a word boundary when possible.
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start:
                end = boundary
        chunks.append(TextChunk(text=text[start:end], start_char=start))
        if end >= len(text):
            break
        start += stride_chars
        if start >= len(text):
            break

    return chunks if chunks else [TextChunk(text=text, start_char=0)]
