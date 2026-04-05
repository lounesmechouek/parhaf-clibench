"""Storage helpers for artifacts produced by one run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class ArtifactStore:
    """Write helper for artifacts under `results/<run_id>/...`."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def write_json(self, name: str, payload: dict[str, Any]) -> None:
        (self.base_dir / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def write_jsonl(self, name: str, rows: list[dict[str, Any]]) -> None:
        lines = [json.dumps(row, ensure_ascii=False) for row in rows]
        (self.base_dir / name).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    def append_jsonl(self, name: str, row: dict[str, Any]) -> None:
        """Append one JSON row to a `.jsonl` file."""

        with (self.base_dir / name).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    def write_text(self, name: str, text: str) -> None:
        (self.base_dir / name).write_text(text, encoding="utf-8")
