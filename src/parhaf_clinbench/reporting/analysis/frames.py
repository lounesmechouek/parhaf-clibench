"""Tidy dataframe builders that turn :class:`RunArtifacts` into analysis frames.

Every function takes a ``suite`` mapping ``{model_id: RunArtifacts}`` and
returns a single flat pandas DataFrame. The frames are the only currency the
plotting and Streamlit layers consume, which keeps them decoupled from the
on-disk artefact format.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from parhaf_clinbench.reporting.loader import RunArtifacts


def build_scores_frame(suite: dict[str, RunArtifacts]) -> pd.DataFrame:
    """Long-format per (model, track, task) score frame.

    Columns: ``model, track, task, metric_kind, metric_name, precision, recall,
    f1, ci_low, ci_high``. ``metric_kind`` is ``official`` or ``secondary``.
    """

    rows: list[dict[str, Any]] = []
    for model, run in suite.items():
        ci_lookup: dict[tuple[str, str], tuple[float, float]] = {}
        if not run.metrics_csv.empty:
            for _, r in run.metrics_csv.iterrows():
                if r["task"] == "GLOBAL":
                    continue
                ci_lookup[(str(r["track"]), str(r["task"]))] = (
                    float(r["ci_low"]),
                    float(r["ci_high"]),
                )
        for track_block in run.metrics.get("tracks", []):
            track = track_block["track"]
            for task, tb in track_block.get("per_task", {}).items():
                off = tb["official"]
                ci_low, ci_high = ci_lookup.get((track, task), (None, None))
                rows.append(
                    {
                        "model": model,
                        "track": track,
                        "task": task,
                        "metric_kind": "official",
                        "metric_name": tb["official_name"],
                        "precision": off["precision"],
                        "recall": off["recall"],
                        "f1": off["f1"],
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                    }
                )
                for name, triplet in (tb.get("secondary") or {}).items():
                    rows.append(
                        {
                            "model": model,
                            "track": track,
                            "task": task,
                            "metric_kind": "secondary",
                            "metric_name": name,
                            "precision": triplet["precision"],
                            "recall": triplet["recall"],
                            "f1": triplet["f1"],
                            "ci_low": None,
                            "ci_high": None,
                        }
                    )
    return pd.DataFrame(rows)


def build_global_scores_frame(suite: dict[str, RunArtifacts]) -> pd.DataFrame:
    """Per (model, track) global score — mean of official task F1s.

    Pulls the GLOBAL row from ``metrics.csv`` when available (which carries the
    bootstrap CI for the global score), otherwise computes the arithmetic mean
    from the task-level frame.
    """

    rows: list[dict[str, Any]] = []
    for model, run in suite.items():
        if run.metrics_csv.empty:
            continue
        csv = run.metrics_csv
        for _, r in csv[csv["task"] == "GLOBAL"].iterrows():
            rows.append(
                {
                    "model": model,
                    "track": r["track"],
                    "global_f1": float(r["f1"]),
                    "ci_low": float(r["ci_low"]),
                    "ci_high": float(r["ci_high"]),
                }
            )
    return pd.DataFrame(rows)


def build_robustness_frame(suite: dict[str, RunArtifacts]) -> pd.DataFrame:
    """Per (model, track, task) robustness/operational metric frame."""

    rows: list[dict[str, Any]] = []
    for model, run in suite.items():
        for track_block in run.metrics.get("tracks", []):
            track = track_block["track"]
            for task, tb in track_block.get("per_task", {}).items():
                rob = tb.get("robustness") or {}
                rows.append({"model": model, "track": track, "task": task, **rob})
    return pd.DataFrame(rows)


def build_timings_frame(suite: dict[str, RunArtifacts]) -> pd.DataFrame:
    """Concatenate per-document timing records across the suite."""

    parts: list[pd.DataFrame] = []
    for model, run in suite.items():
        if run.timings.empty:
            continue
        t = run.timings.copy()
        t["model"] = model
        parts.append(t)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def build_predictions_frame(suite: dict[str, RunArtifacts]) -> pd.DataFrame:
    """Concatenate per-document prediction records across the suite.

    Only the lightweight columns are kept (``document_id``, ``task``,
    ``track``, ``raw_json_valid``, ``repair_applied``, ``is_schema_valid``)
    plus ``model``. The ``parsed`` and ``raw_output`` fields are intentionally
    dropped to keep the frame small enough for Streamlit and parquet storage.
    """

    cols = ["document_id", "task", "track", "raw_json_valid", "repair_applied", "is_schema_valid"]
    parts: list[pd.DataFrame] = []
    for model, run in suite.items():
        if run.predictions.empty:
            continue
        keep = [c for c in cols if c in run.predictions.columns]
        df = run.predictions[keep].copy()
        df["model"] = model
        parts.append(df)
    if not parts:
        return pd.DataFrame()
    return pd.concat(parts, ignore_index=True)


def build_errors_frame(suite: dict[str, RunArtifacts]) -> pd.DataFrame:
    """Concatenate per-document error records across the suite."""

    parts: list[pd.DataFrame] = []
    for model, run in suite.items():
        if run.errors.empty:
            continue
        e = run.errors.copy()
        e["model"] = model
        parts.append(e)
    if not parts:
        return pd.DataFrame(columns=["model", "document_id", "task", "track", "error"])
    return pd.concat(parts, ignore_index=True)
