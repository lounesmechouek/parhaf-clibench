from __future__ import annotations

from pathlib import Path

from parhaf_clinbench.core.enums import TaskId
from parhaf_clinbench.orchestration.runner import score_from_jsonl


def test_score_offline_from_fixtures() -> None:
    result = score_from_jsonl(
        task=TaskId.PSEUDO,
        predictions_path=Path("tests/fixtures/sample_predictions.jsonl"),
        gold_path=Path("tests/fixtures/sample_gold.jsonl"),
    )
    assert result["f1"] == 1.0
