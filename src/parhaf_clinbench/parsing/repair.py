"""Lightweight repair helpers for noisy JSON model outputs."""

from __future__ import annotations


def extract_json_block(text: str) -> str:
    """Extract the first plausible JSON object substring from raw text.

    Args:
        text: Raw model output.

    Returns:
        Original text or extracted JSON object substring.
    """

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return text
    return text[start : end + 1]
