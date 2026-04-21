"""Run-artefact loaders used by the analysis layer.

A benchmark run is a directory with a fixed file set (metrics, predictions,
errors, timings, metadata, configs). :class:`RunArtifacts` bundles these into a
single in-memory object, and :func:`load_run_suite` sweeps a results root into
a mapping keyed by short model id so notebooks and the Streamlit app share a
single entry point.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import pandas as pd

_RUN_DIR_RE = re.compile(r"^(?P<model>[^_]+(?:_[^_]+)*?)_\d{8}T\d{6}Z_[0-9a-f]+$")


@dataclass
class RunArtifacts:
    """Container holding every artefact produced by a single benchmark run."""

    run_dir: Path
    model_id: str
    metrics: dict[str, Any]
    metrics_csv: pd.DataFrame
    predictions: pd.DataFrame
    errors: pd.DataFrame
    timings: pd.DataFrame
    run_metadata: dict[str, Any]
    resolved_config: dict[str, Any]
    report_md: str
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def tracks(self) -> list[str]:
        return [t["track"] for t in self.metrics.get("tracks", [])]


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return cast(dict[str, Any], payload)


def _read_jsonl(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return pd.DataFrame(rows)


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    import yaml  # local dep already present in pyproject

    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _parse_model_id(run_dir: Path) -> str:
    """Short model id extracted from the run-dir naming convention."""

    match = _RUN_DIR_RE.match(run_dir.name)
    if match:
        return match.group("model")
    return run_dir.name.split("_")[0]


def load_run(run_dir: Path) -> RunArtifacts:
    """Load one benchmark run folder into a :class:`RunArtifacts` object."""

    run_dir = Path(run_dir)
    metrics = _read_json(run_dir / "metrics.json")
    metrics_csv = (
        pd.read_csv(run_dir / "metrics.csv")
        if (run_dir / "metrics.csv").exists()
        else pd.DataFrame()
    )
    preds = _read_jsonl(run_dir / "predictions.jsonl")
    errors = _read_jsonl(run_dir / "errors.jsonl")
    timings = _read_jsonl(run_dir / "timings.jsonl")
    run_metadata = _read_json(run_dir / "run_metadata.json")
    resolved_config = _read_yaml(run_dir / "resolved_config.yaml")
    report_md = (run_dir / "report.md").read_text(encoding="utf-8") if (run_dir / "report.md").exists() else ""
    return RunArtifacts(
        run_dir=run_dir,
        model_id=_parse_model_id(run_dir),
        metrics=metrics,
        metrics_csv=metrics_csv,
        predictions=preds,
        errors=errors,
        timings=timings,
        run_metadata=run_metadata,
        resolved_config=resolved_config,
        report_md=report_md,
    )


def load_run_suite(root: Path) -> dict[str, RunArtifacts]:
    """Load every run under a results root into a dict keyed by model id.

    When several runs exist for the same short model id only the most recent
    one (lexicographic sort on directory name, which includes an ISO8601 stamp)
    is kept. This matches the publish-latest convention used by the runner.
    """

    root = Path(root)
    candidates: dict[str, list[Path]] = {}
    for child in sorted(root.iterdir()):
        if not child.is_dir():
            continue
        if not (child / "metrics.json").exists():
            continue
        model_id = _parse_model_id(child)
        candidates.setdefault(model_id, []).append(child)

    suite: dict[str, RunArtifacts] = {}
    for model_id, dirs in candidates.items():
        dirs.sort()
        suite[model_id] = load_run(dirs[-1])
    return suite
