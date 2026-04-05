"""Merge chunk-level canonical documents back into one document."""

from __future__ import annotations

from parhaf_clinbench.core.models import CanonicalDocument, Record


def merge_canonical_documents(
    chunks: list[tuple[CanonicalDocument, int]],
) -> CanonicalDocument:
    """Merge chunk-level documents into a single canonical document.

    Args:
        chunks: Ordered list of ``(document, chunk_start_char)`` tuples where
            `chunk_start_char` is the first-character offset of the chunk in
            the original source text.

    Returns:
        A merged document with offsets shifted to source coordinates and
        duplicates removed in overlap areas.

    Examples:
        >>> merged = merge_canonical_documents([(doc_a, 0), (doc_b, 512)])
        >>> merged.document_id == doc_a.document_id
        True
    """
    if not chunks:
        raise ValueError("La liste de chunks ne peut pas être vide")

    first_doc = chunks[0][0]
    seen: dict[tuple[int | None, int | None, str, str], None] = {}
    merged_records: list[Record] = []

    for doc, start_char in chunks:
        for rec in doc.records:
            adjusted = _adjust_offsets(rec, start_char)
            key = _dedup_key(adjusted)
            if key not in seen:
                seen[key] = None
                merged_records.append(adjusted)

    speciality = next(
        (doc.speciality for doc, _ in chunks if doc.speciality is not None),
        None,
    )

    return CanonicalDocument(
        document_id=first_doc.document_id,
        task=first_doc.task,
        speciality=speciality,
        records=merged_records,
    )


def _adjust_offsets(record: Record, start_char: int) -> Record:
    """Return a copy of `record` with offsets shifted by `start_char`."""
    if start_char == 0 or record.start is None:
        return record
    return record.model_copy(
        update={
            "start": record.start + start_char,
            "end": record.end + start_char,  # type: ignore[operator]
        }
    )


def _dedup_key(record: Record) -> tuple[int | None, int | None, str, str]:
    """Build the deduplication key used while merging chunk outputs."""
    if record.start is not None:
        return (record.start, record.end, record.label, "")
    return (None, None, record.label, record.text or "")
