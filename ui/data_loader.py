"""Cached data accessors for the benchmark dashboard.

The Streamlit UI consumes a small, stable set of parquet and JSON artefacts
exported from a benchmark run. Centralising the loaders here keeps every page
focused on clinical interpretation instead of file-system concerns.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).parent / "data"
ASSETS_DIR = Path(__file__).parent / "assets"


def _parquet(name: str) -> pd.DataFrame:
    """Load one parquet artefact and degrade gracefully when it is absent."""

    path = DATA_DIR / f"{name}.parquet"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_scores() -> pd.DataFrame:
    """Return per-task benchmark scores."""

    return _parquet("scores")


@st.cache_data(show_spinner=False)
def load_global_scores() -> pd.DataFrame:
    """Return equal-weight global benchmark scores."""

    return _parquet("global_scores")


@st.cache_data(show_spinner=False)
def load_robustness() -> pd.DataFrame:
    """Return operational robustness metrics per model, task, and track."""

    return _parquet("robustness")


@st.cache_data(show_spinner=False)
def load_timings() -> pd.DataFrame:
    """Return per-document latency and token-usage records."""

    return _parquet("timings")


@st.cache_data(show_spinner=False)
def load_subgroups() -> pd.DataFrame:
    """Return subgroup analyses derived from the gold corpora."""

    return _parquet("subgroups")


@st.cache_data(show_spinner=False)
def load_error_taxonomy() -> pd.DataFrame:
    """Return aggregated schema and parsing failure categories."""

    return _parquet("error_taxonomy")


@st.cache_data(show_spinner=False)
def load_errors() -> pd.DataFrame:
    """Return raw per-document error rows for exploratory debugging."""

    return _parquet("errors")


@st.cache_data(show_spinner=False)
def load_audit() -> pd.DataFrame:
    """Return the independent scoring audit table."""

    return _parquet("scoring_audit")


@st.cache_data(show_spinner=False)
def load_paired_deltas() -> pd.DataFrame:
    """Return paired bootstrap deltas between model pairs."""

    return _parquet("paired_deltas")


@st.cache_data(show_spinner=False)
def load_vs_baseline() -> pd.DataFrame:
    """Return comparisons against the encoder baseline."""

    return _parquet("vs_baseline")


@st.cache_data(show_spinner=False)
def load_fewshot_vs_zeroshot() -> pd.DataFrame:
    """Return paired few-shot versus zero-shot deltas."""

    return _parquet("fewshot_vs_zeroshot")


@st.cache_data(show_spinner=False)
def load_run_metadata() -> pd.DataFrame:
    """Return per-run infrastructure and runtime metadata."""

    return _parquet("run_metadata")


@st.cache_data(show_spinner=False)
def load_manifest() -> dict[str, Any]:
    """Return the exported manifest describing the published run bundle."""

    path = DATA_DIR / "manifest.json"
    if not path.exists():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return {}
    return cast(dict[str, Any], payload)
