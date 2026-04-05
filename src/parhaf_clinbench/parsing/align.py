"""Best-effort alignment of missing offsets against source text."""

from __future__ import annotations

from parhaf_clinbench.core.models import Record


def align_offsets(record: Record, source_text: str) -> Record:
    """Fill missing `start`/`end` offsets when `record.text` can be located.

    Args:
        record: Input record that may miss offsets.
        source_text: Original document text.

    Returns:
        The same record or an updated copy with inferred offsets.

    Examples:
        >>> aligned = align_offsets(Record(label="X", text="abc"), "xxabcxx")
        >>> (aligned.start, aligned.end)
        (2, 5)
    """

    if record.start is not None and record.end is not None:
        return record
    if not record.text:
        return record
    idx = source_text.find(record.text)
    if idx == -1:
        return record
    return record.model_copy(update={"start": idx, "end": idx + len(record.text)})
