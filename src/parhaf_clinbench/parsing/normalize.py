"""Text normalization helpers used by scoring metrics."""

from __future__ import annotations

import re
import unicodedata

_TERMINAL_PUNCT_RE = re.compile(r"[\s\.,;:!?]+$")
_SPACES_RE = re.compile(r"\s+")



def normalize_text(value: str | None) -> str:
    """Normalize text according to benchmark v1 comparison policy.

    Args:
        value: Input text.

    Returns:
        Lowercased, whitespace-normalized text with trailing punctuation removed.

    Examples:
        >>> normalize_text("  Pneumonie.  ")
        'pneumonie'
    """

    if value is None:
        return ""
    step = unicodedata.normalize("NFC", value)
    step = step.strip()
    step = _SPACES_RE.sub(" ", step)
    step = step.lower()
    step = _TERMINAL_PUNCT_RE.sub("", step)
    return step
