"""Conversion helpers between JSON dictionaries and canonical models."""

from __future__ import annotations

from typing import Any

from parhaf_clinbench.core.models import CanonicalDocument


def dict_to_canonical_document(payload: dict[str, Any]) -> CanonicalDocument:
    """Convert a JSON dictionary into a validated canonical document.

    Args:
        payload: JSON-like dictionary representation of a canonical document.

    Returns:
        Validated `CanonicalDocument` instance.

    Examples:
        >>> doc = dict_to_canonical_document({"document_id": "d1", "task": "pseudo", "records": []})
        >>> doc.document_id
        'd1'
    """

    return CanonicalDocument.model_validate(payload)



def canonical_to_dict(document: CanonicalDocument) -> dict[str, Any]:
    """Convert a canonical document into a JSON-serializable dictionary.

    Args:
        document: Canonical document model.

    Returns:
        JSON-serializable dictionary payload.

    Examples:
        >>> payload = canonical_to_dict(CanonicalDocument(document_id="d1", task="pseudo", records=[]))
        >>> payload["document_id"]
        'd1'
    """

    return document.model_dump(mode="json")
