from __future__ import annotations

import json
from pathlib import Path

from parhaf_clinbench.orchestration.artifact_store import ArtifactStore


def test_artifact_store_append_jsonl_writes_one_line_per_call(tmp_path: Path) -> None:
    store = ArtifactStore(tmp_path)
    store.write_text("events.jsonl", "")

    store.append_jsonl("events.jsonl", {"id": 1, "status": "ok"})
    store.append_jsonl("events.jsonl", {"id": 2, "status": "ok"})

    lines = (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"id": 1, "status": "ok"}
    assert json.loads(lines[1]) == {"id": 2, "status": "ok"}
