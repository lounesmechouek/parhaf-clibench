"""Terminal dashboard for live benchmark monitoring.

This monitor watches the run directories emitted by PARHAF-LM-CLINBENCH and
surfaces the execution state that matters operationally: which model is
running, how quickly documents are processed, whether outputs remain valid
JSON, and what the latest run events say about failures or progress.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def _fmt_elapsed(seconds: float | None) -> str:
    """Format a duration in seconds into a compact human-readable string."""

    if seconds is None:
        return "—"
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _fmt_f1(value: float | None) -> str:
    """Format an F1 score for the dashboard table."""

    if value is None:
        return "—"
    return f"{value:.3f}"


def _pct(value: float | None) -> str:
    """Format a ratio as a percentage string."""

    if value is None:
        return "—"
    return f"{value * 100:.1f}%"


def _now_elapsed(started_at_utc: str | None) -> float | None:
    """Compute elapsed seconds from an ISO-8601 UTC timestamp."""

    if started_at_utc is None:
        return None
    try:
        start = datetime.fromisoformat(started_at_utc).replace(tzinfo=UTC)
        return (datetime.now(tz=UTC) - start).total_seconds()
    except ValueError:
        return None


def _pct95(values: list[float]) -> float:
    """Return the empirical 95th percentile of a numeric list."""

    if not values:
        return 0.0
    s = sorted(values)
    idx = max(0, min(math.ceil(0.95 * len(s)) - 1, len(s) - 1))
    return s[idx]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file and skip malformed lines."""

    rows: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    except FileNotFoundError:
        pass
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    """Read a JSON object from disk and degrade to an empty mapping."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return cast(dict[str, Any], payload)


def _read_log_events(path: Path, last_n: int = 12) -> list[dict[str, Any]]:
    """Read the latest structured log events from a run log file."""

    events: list[dict[str, Any]] = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            # Format: "2024-01-01 12:00:00,000 INFO {...json...}"
            # Extract JSON portion after the third space-separated token
            parts = line.split(" ", 3)
            if len(parts) < 4:
                continue
            payload_str = parts[3].strip()
            try:
                payload = json.loads(payload_str)
                if isinstance(payload, dict):
                    events.append({"_ts": f"{parts[0]} {parts[1]}", "_lvl": parts[2], **payload})
            except json.JSONDecodeError:
                pass
    except FileNotFoundError:
        pass
    return events[-last_n:]


def _find_run_dirs(output_dir: Path) -> list[Path]:
    """Return all run directories sorted oldest -> newest by name."""

    if not output_dir.exists():
        return []
    dirs = [d for d in output_dir.iterdir() if d.is_dir() and (d / "logs").exists()]
    return sorted(dirs, key=lambda d: d.name)


def _timings_stats(timings: list[dict[str, Any]]) -> dict[str, float]:
    """Aggregate latency and token statistics from timing rows."""

    latencies = [row["latency_ms"] for row in timings if "latency_ms" in row]
    in_tok = [row["input_tokens"] for row in timings if "input_tokens" in row]
    out_tok = [row["output_tokens"] for row in timings if "output_tokens" in row]
    total_out = sum(out_tok)
    total_lat_s = sum(latencies) / 1000.0
    return {
        "count": len(latencies),
        "lat_median": statistics.median(latencies) if latencies else 0.0,
        "lat_p95": _pct95(latencies),
        "lat_mean": statistics.mean(latencies) if latencies else 0.0,
        "in_tok_mean": statistics.mean(in_tok) if in_tok else 0.0,
        "out_tok_mean": statistics.mean(out_tok) if out_tok else 0.0,
        "throughput": total_out / total_lat_s if total_lat_s > 0 else 0.0,
    }


def _quality_stats(predictions: list[dict[str, Any]], errors: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute output-validity diagnostics for a run."""

    n = len(predictions)
    valid_json = sum(1 for r in predictions if r.get("raw_json_valid"))
    repaired = sum(1 for r in predictions if r.get("repair_applied"))
    schema_ok = sum(1 for r in predictions if r.get("is_schema_valid"))
    return {
        "n": n,
        "valid_json_rate": valid_json / n if n else 0.0,
        "repair_rate": repaired / n if n else 0.0,
        "schema_rate": schema_ok / n if n else 0.0,
        "error_count": len(errors),
    }


def _global_score_from_metrics(metrics_json: dict[str, Any]) -> float | None:
    """Return the average global score across available prompting tracks."""

    tracks = metrics_json.get("tracks", [])
    if not tracks:
        return None
    scores = [t.get("global_score") for t in tracks if t.get("global_score") is not None]
    if not scores:
        return None
    return float(statistics.mean(scores))


def _render_header(run_dirs: list[Path], campaign_start: str | None) -> Panel:
    """Render the top status banner for the full benchmark campaign."""

    elapsed = _now_elapsed(campaign_start)
    n_done = sum(1 for d in run_dirs if _read_json(d / "run_status.json").get("status") == "success")
    n_total = len(run_dirs)

    status_color = "green" if n_done == n_total and n_total > 0 else "yellow"
    status_label = "DONE" if n_done == n_total and n_total > 0 else "RUNNING"

    text = Text()
    text.append("prehaf-clibench", style="bold cyan")
    text.append("  •  ", style="dim")
    text.append(f"Models: {n_done}/{n_total}", style="bold")
    text.append("  •  ", style="dim")
    text.append(f"Elapsed: {_fmt_elapsed(elapsed)}", style="white")
    text.append("  •  ", style="dim")
    text.append(f"[{status_label}]", style=f"bold {status_color}")

    return Panel(text, style="bold", padding=(0, 1))


def _render_current_run(run_dir: Path | None) -> Panel:
    """Render the summary panel for the currently active model run."""

    if run_dir is None:
        return Panel(Text("No active run found.", style="dim"), title="Current Run")

    metadata = _read_json(run_dir / "run_metadata.json")
    timings = _read_jsonl(run_dir / "timings.jsonl")
    predictions = _read_jsonl(run_dir / "predictions.jsonl")
    errors = _read_jsonl(run_dir / "errors.jsonl")
    run_status = _read_json(run_dir / "run_status.json")

    model_id = metadata.get("model_id", run_dir.name)
    runtime = metadata.get("runtime_name", "?")
    started_at = metadata.get("started_at_utc")
    elapsed = _now_elapsed(started_at)
    idx = metadata.get("model_execution_index", "?")
    total = metadata.get("model_execution_total", "?")

    stats = _timings_stats(timings)
    quality = _quality_stats(predictions, errors)

    # Infer current task/track from latest timing entry
    current_task = "—"
    current_track = "—"
    if timings:
        last = timings[-1]
        current_task = last.get("task", "—")
        current_track = last.get("track", "—")

    status = run_status.get("status", "running")
    status_style = "green" if status == "success" else ("red" if status == "failed" else "yellow")

    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="dim", width=22)
    grid.add_column()

    grid.add_row("Model", Text(f"{model_id}", style="bold cyan"))
    grid.add_row("Runtime", runtime)
    grid.add_row("Position", f"{idx}/{total}")
    grid.add_row("Status", Text(status.upper(), style=f"bold {status_style}"))
    grid.add_row("Elapsed", _fmt_elapsed(elapsed))
    grid.add_row("", "")
    grid.add_row("Current task", f"{current_task}  [{current_track}]")
    grid.add_row("Docs processed", str(stats["count"]))
    grid.add_row("", "")
    grid.add_row("Latency  p50 / p95", f"{stats['lat_median']:.0f}ms  /  {stats['lat_p95']:.0f}ms")
    grid.add_row("Throughput", f"{stats['throughput']:.1f} tok/s")
    grid.add_row("Tokens  in / out", f"{stats['in_tok_mean']:.0f}  /  {stats['out_tok_mean']:.0f} avg")
    grid.add_row("", "")
    grid.add_row("JSON valid", _pct(quality["valid_json_rate"]))
    grid.add_row("Schema OK", _pct(quality["schema_rate"]))
    grid.add_row("Repairs applied", _pct(quality["repair_rate"]))
    grid.add_row("Errors", str(quality["error_count"]))

    return Panel(grid, title=f"[bold]Active Run[/bold]  •  {run_dir.name}")


def _render_model_table(run_dirs: list[Path]) -> Panel:
    """Render the model-by-model execution summary table."""

    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
    table.add_column("#", style="dim", width=3)
    table.add_column("Model", min_width=20)
    table.add_column("Runtime", width=8)
    table.add_column("Status", width=10)
    table.add_column("Docs", width=6, justify="right")
    table.add_column("Global F1", width=10, justify="right")
    table.add_column("Elapsed", width=10, justify="right")

    for i, run_dir in enumerate(run_dirs, start=1):
        metadata = _read_json(run_dir / "run_metadata.json")
        timings = _read_jsonl(run_dir / "timings.jsonl")
        run_status = _read_json(run_dir / "run_status.json")
        metrics_json = _read_json(run_dir / "metrics.json")

        model_id = metadata.get("model_id", run_dir.name)
        runtime = metadata.get("runtime_name", "?")
        status = run_status.get("status", "running")
        elapsed_s = run_status.get("elapsed_seconds") or _now_elapsed(metadata.get("started_at_utc"))
        global_f1 = _global_score_from_metrics(metrics_json)

        if status == "success":
            status_text = Text("✓ DONE", style="green")
        elif status == "failed":
            status_text = Text("✗ FAILED", style="red")
        else:
            status_text = Text("● RUNNING", style="yellow")

        table.add_row(
            str(i),
            model_id,
            runtime,
            status_text,
            str(len(timings)),
            _fmt_f1(global_f1),
            _fmt_elapsed(elapsed_s),
        )

    return Panel(table, title="[bold]Models[/bold]")


def _render_events(run_dir: Path | None) -> Panel:
    """Render the latest structured log events for the active run."""

    if run_dir is None:
        return Panel(Text("No events.", style="dim"), title="Recent Events")

    events = _read_log_events(run_dir / "logs" / "run.log", last_n=10)
    if not events:
        return Panel(Text("No events yet.", style="dim"), title="Recent Events")

    lines = Text()
    for evt in events:
        ts = evt.get("_ts", "")
        lvl = evt.get("_lvl", "INFO")
        event_name = evt.get("event", "")
        lvl_style = "yellow" if lvl == "WARNING" else ("red" if lvl == "ERROR" else "dim")

        lines.append(f"{ts} ", style="dim")
        lines.append(f"{lvl:<8}", style=lvl_style)

        if event_name:
            lines.append(f"{event_name}", style="bold")
            # Show a few key fields
            extra_fields = {k: v for k, v in evt.items() if k not in ("_ts", "_lvl", "event")}
            if extra_fields:
                summary_parts = []
                for key in ("model_id", "runtime", "hf_id", "task", "run_id", "elapsed_seconds", "error"):
                    if key in extra_fields:
                        val = extra_fields[key]
                        if key == "elapsed_seconds" and isinstance(val, (int, float)):
                            val = _fmt_elapsed(float(val))
                        summary_parts.append(f"{key}={val}")
                if summary_parts:
                    lines.append(f"  {' '.join(summary_parts[:4])}", style="dim")
        else:
            lines.append(f"{evt}", style="dim")
        lines.append("\n")

    return Panel(lines, title="[bold]Recent Events[/bold]")


def _build_layout(run_dirs: list[Path]) -> Layout:
    """Assemble the Rich layout tree from the discovered run directories."""

    active_run = None
    for d in reversed(run_dirs):
        status = _read_json(d / "run_status.json").get("status", "running")
        if status == "running":
            active_run = d
            break
    # Fallback to latest run if nothing is explicitly running
    if active_run is None and run_dirs:
        active_run = run_dirs[-1]

    campaign_start = None
    if run_dirs:
        first_meta = _read_json(run_dirs[0] / "run_metadata.json")
        campaign_start = first_meta.get("started_at_utc")

    layout = Layout()
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="body"),
        Layout(name="footer", size=14),
    )
    layout["body"].split_row(
        Layout(name="current", ratio=2),
        Layout(name="models", ratio=3),
    )

    layout["header"].update(_render_header(run_dirs, campaign_start))
    layout["current"].update(_render_current_run(active_run))
    layout["models"].update(_render_model_table(run_dirs))
    layout["footer"].update(_render_events(active_run))

    return layout


def main() -> None:
    """Start the live terminal dashboard and refresh it until interrupted."""

    parser = argparse.ArgumentParser(
        prog="dashboard",
        description="Live monitoring dashboard for prehaf-clibench benchmark runs.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/default",
        help="Output directory to watch (default: results/default)",
    )
    parser.add_argument(
        "--refresh",
        type=float,
        default=2.0,
        help="Refresh interval in seconds (default: 2)",
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    console = Console()

    with Live(console=console, refresh_per_second=1, screen=True) as live:
        while True:
            run_dirs = _find_run_dirs(output_dir)
            layout = _build_layout(run_dirs)
            live.update(layout)
            time.sleep(args.refresh)


if __name__ == "__main__":
    main()
