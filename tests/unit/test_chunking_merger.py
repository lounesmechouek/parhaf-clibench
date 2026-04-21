"""Tests unitaires pour parhaf_clinbench.chunking.merger."""

from __future__ import annotations

import pytest

from parhaf_clinbench.chunking.merger import merge_canonical_documents
from parhaf_clinbench.core.enums import TaskId
from parhaf_clinbench.core.models import CanonicalDocument, Record


def _pseudo_doc(document_id: str, records: list[Record]) -> CanonicalDocument:
    return CanonicalDocument(document_id=document_id, task=TaskId.PSEUDO, records=records)


def _infectio_doc(document_id: str, records: list[Record]) -> CanonicalDocument:
    return CanonicalDocument(document_id=document_id, task=TaskId.INFECTIO, records=records)


def _record(label: str, text: str, start: int, end: int, **attrs: str) -> Record:
    return Record(label=label, text=text, start=start, end=end, attributes=dict(attrs))


class TestMergeOffsetAdjustment:
    def test_single_chunk_no_offset(self) -> None:
        rec = _record("LAST_NAME", "Dupont", 10, 16)
        doc = _pseudo_doc("doc1", [rec])
        merged = merge_canonical_documents([(doc, 0)])
        assert len(merged.records) == 1
        assert merged.records[0].start == 10
        assert merged.records[0].end == 16

    def test_offset_added_to_start_end(self) -> None:
        rec = _record("LAST_NAME", "Dupont", 5, 11)
        doc = _pseudo_doc("doc1", [rec])
        merged = merge_canonical_documents([(doc, 100)])
        assert merged.records[0].start == 105
        assert merged.records[0].end == 111

    def test_two_chunks_offsets_adjusted(self) -> None:
        r1 = _record("FIRST_NAME", "Jean", 0, 4)
        r2 = _record("LAST_NAME", "Dupont", 2, 8)
        doc1 = _pseudo_doc("doc1", [r1])
        doc2 = _pseudo_doc("doc1", [r2])
        merged = merge_canonical_documents([(doc1, 0), (doc2, 50)])
        starts = {r.start for r in merged.records}
        assert 0 in starts   # r1 inchangé
        assert 52 in starts  # r2: 2 + 50

    def test_none_start_no_offset_added(self) -> None:
        rec = Record(label="Infection", text="pneumonie", start=None, end=None, attributes={})
        doc = _infectio_doc("doc1", [rec])
        merged = merge_canonical_documents([(doc, 50)])
        assert merged.records[0].start is None
        assert merged.records[0].end is None


class TestMergeDeduplication:
    def test_duplicate_in_overlap_removed(self) -> None:
        """Même entité dans deux chunks chevauchants → conservée une seule fois."""
        # Chunk 1: texte 0-100, chunk 2: texte 70-170 (overlap 70-100)
        rec_shared = _record("LAST_NAME", "Martin", 5, 11)  # dans l'overlap
        doc1 = _pseudo_doc("doc1", [rec_shared])
        # Dans chunk 2, l'entité a offset 5-11 dans le chunk, start_char=70
        # → après ajustement: 75-81 ≠ 5-11, donc PAS de doublon ici
        # Pour avoir un vrai doublon, le start absolu doit être identique
        rec_dup = _record("LAST_NAME", "Martin", 5, 11)  # même offset absolu après correction
        doc2 = _pseudo_doc("doc1", [rec_dup])
        merged = merge_canonical_documents([(doc1, 0), (doc2, 0)])
        assert len(merged.records) == 1

    def test_different_entities_both_kept(self) -> None:
        r1 = _record("LAST_NAME", "Dupont", 0, 6)
        r2 = _record("FIRST_NAME", "Jean", 10, 14)
        doc1 = _pseudo_doc("doc1", [r1])
        doc2 = _pseudo_doc("doc1", [r2])
        merged = merge_canonical_documents([(doc1, 0), (doc2, 0)])
        assert len(merged.records) == 2

    def test_same_position_different_label_both_kept(self) -> None:
        r1 = _record("LAST_NAME", "Martin", 5, 11)
        r2 = _record("FIRST_NAME", "Martin", 5, 11)
        doc1 = _pseudo_doc("doc1", [r1])
        doc2 = _pseudo_doc("doc1", [r2])
        merged = merge_canonical_documents([(doc1, 0), (doc2, 0)])
        assert len(merged.records) == 2

    def test_no_span_dedup_by_text_label(self) -> None:
        rec1 = Record(label="Infection", text="pneumonie", start=None, end=None, attributes={})
        rec2 = Record(label="Infection", text="pneumonie", start=None, end=None, attributes={})
        doc1 = _infectio_doc("doc1", [rec1])
        doc2 = _infectio_doc("doc1", [rec2])
        merged = merge_canonical_documents([(doc1, 0), (doc2, 0)])
        assert len(merged.records) == 1


class TestMergeMetadata:
    def test_document_id_from_first_chunk(self) -> None:
        doc1 = _pseudo_doc("doc-A", [])
        doc2 = _pseudo_doc("doc-A", [])
        merged = merge_canonical_documents([(doc1, 0), (doc2, 100)])
        assert merged.document_id == "doc-A"

    def test_task_preserved(self) -> None:
        doc = _pseudo_doc("doc1", [])
        merged = merge_canonical_documents([(doc, 0)])
        assert merged.task == TaskId.PSEUDO

    def test_speciality_first_non_none(self) -> None:
        # On utilise PSEUDO (pas de contrainte sur speciality) pour tester la logique
        # de fusion : le merger doit retourner la première valeur non-None.
        doc1 = CanonicalDocument(document_id="d", task=TaskId.PSEUDO, speciality=None, records=[])
        doc2 = CanonicalDocument(document_id="d", task=TaskId.PSEUDO, speciality="Cardiologie", records=[])
        merged = merge_canonical_documents([(doc1, 0), (doc2, 50)])
        assert merged.speciality == "Cardiologie"

    def test_speciality_none_when_all_none(self) -> None:
        doc1 = _pseudo_doc("doc1", [])
        doc2 = _pseudo_doc("doc1", [])
        merged = merge_canonical_documents([(doc1, 0), (doc2, 50)])
        assert merged.speciality is None


class TestMergeEdgeCases:
    def test_single_chunk(self) -> None:
        r = _record("LAST_NAME", "X", 3, 4)
        doc = _pseudo_doc("doc1", [r])
        merged = merge_canonical_documents([(doc, 0)])
        assert merged.records == [r]

    def test_empty_chunks_list_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            merge_canonical_documents([])

    def test_three_chunks_all_records_kept(self) -> None:
        r1 = _record("LAST_NAME", "A", 0, 1)
        r2 = _record("LAST_NAME", "B", 0, 1)
        r3 = _record("LAST_NAME", "C", 0, 1)
        docs = [
            (_pseudo_doc("doc1", [r1]), 0),
            (_pseudo_doc("doc1", [r2]), 50),
            (_pseudo_doc("doc1", [r3]), 100),
        ]
        merged = merge_canonical_documents(docs)
        # r1 à (0,1), r2 ajusté à (50,51), r3 ajusté à (100,101) → tous distincts
        assert len(merged.records) == 3
