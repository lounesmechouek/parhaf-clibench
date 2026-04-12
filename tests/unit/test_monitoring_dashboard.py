from __future__ import annotations

import importlib
import json
import sys
import types
from pathlib import Path
from typing import Any, Literal

import pytest


class _FakeText:
    def __init__(self, value: Any = "", **_kwargs: Any) -> None:
        self.value = str(value)

    def append(self, *_args: Any, **_kwargs: Any) -> None:
        return None


class _FakePanel:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.args = _args
        self.kwargs = _kwargs


class _FakeTable:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def add_column(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def add_row(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    @staticmethod
    def grid(*_args: Any, **_kwargs: Any) -> _FakeTable:
        return _FakeTable()


class _FakeLayout:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        self.children: dict[str, _FakeLayout] = {}

    def split_column(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def split_row(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def update(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def __getitem__(self, name: str) -> _FakeLayout:
        self.children.setdefault(name, _FakeLayout())
        return self.children[name]


class _FakeConsole:
    pass


class _FakeLive:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        return None

    def __enter__(self) -> _FakeLive:
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> Literal[False]:
        return False

    def update(self, *_args: Any, **_kwargs: Any) -> None:
        return None


def _load_dashboard_with_fake_rich() -> Any:
    rich_mod: Any = types.ModuleType("rich")
    console_mod: Any = types.ModuleType("rich.console")
    layout_mod: Any = types.ModuleType("rich.layout")
    live_mod: Any = types.ModuleType("rich.live")
    panel_mod: Any = types.ModuleType("rich.panel")
    table_mod: Any = types.ModuleType("rich.table")
    text_mod: Any = types.ModuleType("rich.text")

    console_mod.Console = _FakeConsole
    layout_mod.Layout = _FakeLayout
    live_mod.Live = _FakeLive
    panel_mod.Panel = _FakePanel
    table_mod.Table = _FakeTable
    text_mod.Text = _FakeText

    sys.modules.update(
        {
            "rich": rich_mod,
            "rich.console": console_mod,
            "rich.layout": layout_mod,
            "rich.live": live_mod,
            "rich.panel": panel_mod,
            "rich.table": table_mod,
            "rich.text": text_mod,
        }
    )
    module = importlib.import_module("monitoring.dashboard")
    return importlib.reload(module)


dashboard = _load_dashboard_with_fake_rich()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def test_format_helpers() -> None:
    assert dashboard._fmt_elapsed(None) == "—"
    assert dashboard._fmt_elapsed(59) == "59s"
    assert dashboard._fmt_elapsed(61) == "1m01s"
    assert dashboard._fmt_elapsed(3661) == "1h01m01s"
    assert dashboard._fmt_f1(None) == "—"
    assert dashboard._fmt_f1(0.1234) == "0.123"
    assert dashboard._pct(None) == "—"
    assert dashboard._pct(0.25) == "25.0%"


def test_readers(tmp_path: Path) -> None:
    json_path = tmp_path / "data.json"
    _write_json(json_path, {"a": 1})
    assert dashboard._read_json(json_path)["a"] == 1

    list_path = tmp_path / "list.json"
    list_path.write_text("[1, 2]", encoding="utf-8")
    assert dashboard._read_json(list_path) == {}

    jsonl_path = tmp_path / "data.jsonl"
    _write_jsonl(jsonl_path, [{"a": 1}, {"b": 2}])
    assert len(dashboard._read_jsonl(jsonl_path)) == 2

    log_path = tmp_path / "run.log"
    log_path.write_text(
        "2024-01-01 12:00:00,000 INFO {\"event\": \"start\", \"model_id\": \"m1\"}\n",
        encoding="utf-8",
    )
    events = dashboard._read_log_events(log_path, last_n=5)
    assert events[0]["event"] == "start"


def test_stats_helpers() -> None:
    timings = [
        {"latency_ms": 100, "input_tokens": 10, "output_tokens": 20},
        {"latency_ms": 200, "input_tokens": 12, "output_tokens": 18},
    ]
    stats = dashboard._timings_stats(timings)
    assert stats["count"] == 2

    quality = dashboard._quality_stats(
        [{"raw_json_valid": True, "repair_applied": False, "is_schema_valid": True}],
        [{"error": "x"}],
    )
    assert quality["valid_json_rate"] == 1.0
    assert quality["error_count"] == 1

    metrics = {"tracks": [{"global_score": 0.2}, {"global_score": 0.4}]}
    assert dashboard._global_score_from_metrics(metrics) == pytest.approx(0.3)


def test_layout_build(tmp_path: Path) -> None:
    run_dir = tmp_path / "modelA_20240101T000000Z_abcd"
    (run_dir / "logs").mkdir(parents=True)
    _write_json(
        run_dir / "run_metadata.json",
        {"model_id": "modelA", "runtime_name": "rt", "started_at_utc": "2024-01-01T00:00:00"},
    )
    _write_json(run_dir / "run_status.json", {"status": "running", "elapsed_seconds": 5})
    _write_json(run_dir / "metrics.json", {"tracks": [{"global_score": 0.2}]})
    _write_jsonl(
        run_dir / "timings.jsonl",
        [{"latency_ms": 100, "input_tokens": 10, "output_tokens": 15, "task": "pseudo", "track": "zero-shot"}],
    )
    _write_jsonl(
        run_dir / "predictions.jsonl",
        [{"raw_json_valid": True, "repair_applied": False, "is_schema_valid": True}],
    )
    _write_jsonl(run_dir / "errors.jsonl", [])
    (run_dir / "logs" / "run.log").write_text(
        "2024-01-01 12:00:00,000 INFO {\"event\": \"start\", \"model_id\": \"modelA\"}\n",
        encoding="utf-8",
    )

    run_dirs = dashboard._find_run_dirs(tmp_path)
    assert run_dirs

    panel = dashboard._render_current_run(run_dirs[0])
    assert panel is not None

    table_panel = dashboard._render_model_table(run_dirs)
    assert table_panel is not None

    events_panel = dashboard._render_events(run_dirs[0])
    assert events_panel is not None

    layout = dashboard._build_layout(run_dirs)
    assert layout is not None
