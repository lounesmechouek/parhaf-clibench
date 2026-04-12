"""Parse ``errors.jsonl`` rows into a small taxonomy of failure classes.

The runner writes one row per document whenever the generated payload could
not be turned into a valid :class:`CanonicalDocument`. We classify each row
heuristically into a handful of buckets (invalid JSON, offset drift, label
OOV, negation OOV, empty records, other schema violation), so the analysis
can answer "how much of the gap is generation discipline vs. extraction
quality?" without digging through raw text.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from parhaf_clinbench.reporting.loader import RunArtifacts

_OFFSET_RE = re.compile(r"end must be >= start|span .* out of", re.IGNORECASE)
_LABEL_RE = re.compile(r"invalid .* label|label .* invalid|Label .* invalide", re.IGNORECASE)
_NEGATION_RE = re.compile(r"invalid negation", re.IGNORECASE)
_SPECIALITY_RE = re.compile(r"speciality", re.IGNORECASE)
_JSON_RE = re.compile(r"invalid json|json decode|expecting value", re.IGNORECASE)
_EMPTY_RE = re.compile(r"required|missing", re.IGNORECASE)


def classify_error(message: str) -> str:
    """Map a free-form error message to one of the taxonomy buckets."""

    if not message:
        return "unknown"
    if _JSON_RE.search(message):
        return "invalid_json"
    if _OFFSET_RE.search(message):
        return "offset_drift"
    if _LABEL_RE.search(message):
        return "label_oov"
    if _NEGATION_RE.search(message):
        return "negation_oov"
    if _SPECIALITY_RE.search(message):
        return "speciality_oov"
    if _EMPTY_RE.search(message):
        return "missing_field"
    return "other_schema"


def build_error_taxonomy(suite: dict[str, RunArtifacts]) -> pd.DataFrame:
    """Aggregate ``errors.jsonl`` into a per (model, track, task, category) frame."""

    rows: list[dict[str, Any]] = []
    for model, run in suite.items():
        if run.errors.empty:
            continue
        df = run.errors.copy()
        df["category"] = df["error"].astype(str).map(classify_error)
        grouped = (
            df.groupby(["task", "track", "category"]).size().reset_index(name="count")
        )
        grouped["model"] = model
        rows.extend(grouped.to_dict(orient="records"))
    if not rows:
        return pd.DataFrame(columns=["model", "task", "track", "category", "count"])
    return pd.DataFrame(rows)[["model", "task", "track", "category", "count"]]
