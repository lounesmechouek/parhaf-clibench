from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from parhaf_clinbench.core.enums import TaskId
from parhaf_clinbench.orchestration.runner import score_from_jsonl


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    rendered = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
    path.write_text(rendered, encoding="utf-8")


def test_score_from_jsonl_aligns_on_document_id_not_row_order(tmp_path: Path) -> None:
    gold_path = tmp_path / "gold.jsonl"
    preds_path = tmp_path / "preds.jsonl"

    gold_rows = [
        {
            "document_id": "d1",
            "task": "pseudo",
            "speciality": None,
            "records": [{"label": "FIRST_NAME", "text": "Jean", "start": 0, "end": 4, "attributes": {}}],
        },
        {
            "document_id": "d2",
            "task": "pseudo",
            "speciality": None,
            "records": [{"label": "LAST_NAME", "text": "Martin", "start": 10, "end": 16, "attributes": {}}],
        },
    ]
    pred_rows = [
        {"parsed": gold_rows[1]},
        {"parsed": gold_rows[0]},
    ]

    _write_jsonl(gold_path, gold_rows)
    _write_jsonl(preds_path, pred_rows)

    result = score_from_jsonl(
        task=TaskId.PSEUDO,
        predictions_path=preds_path,
        gold_path=gold_path,
    )
    assert result["f1"] == 1.0


def test_score_from_jsonl_fails_on_missing_document_id(tmp_path: Path) -> None:
    gold_path = tmp_path / "gold.jsonl"
    preds_path = tmp_path / "preds.jsonl"

    _write_jsonl(
        gold_path,
        [
            {
                "document_id": "d1",
                "task": "pseudo",
                "speciality": None,
                "records": [],
            },
            {
                "document_id": "d2",
                "task": "pseudo",
                "speciality": None,
                "records": [],
            },
        ],
    )
    _write_jsonl(
        preds_path,
        [
            {
                "parsed": {
                    "document_id": "d1",
                    "task": "pseudo",
                    "speciality": None,
                    "records": [],
                }
            }
        ],
    )

    with pytest.raises(ValueError, match="not aligned by document_id"):
        score_from_jsonl(
            task=TaskId.PSEUDO,
            predictions_path=preds_path,
            gold_path=gold_path,
        )
