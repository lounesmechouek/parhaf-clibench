"""Streamlit page for inspecting schema and parsing failures.

This page turns raw benchmark execution errors into an auditable view of where
models fail operationally: invalid JSON, offset drift, unsupported labels, or
other schema violations. That separation matters because a clinical benchmark
must distinguish extraction weakness from output-contract weakness.
"""

from __future__ import annotations

import streamlit as st

from parhaf_clinbench.reporting.analysis.error_taxonomy import classify_error
from parhaf_clinbench.reporting.plots_extended import error_taxonomy_stacked_bar
from ui.data_loader import load_error_taxonomy, load_errors
from ui.theme import TRACK_LABELS


def render() -> None:
    """Render the error taxonomy chart and the raw error browser."""

    st.title("Error explorer")
    taxonomy = load_error_taxonomy()
    errors = load_errors()

    if taxonomy.empty:
        st.info("No error records found in this run.")
        return

    st.plotly_chart(error_taxonomy_stacked_bar(taxonomy), use_container_width=True)

    st.markdown("### Browse raw errors")
    col1, col2, col3 = st.columns(3)
    with col1:
        model = st.selectbox("Model", sorted(errors.model.unique()) if not errors.empty else [])
    with col2:
        task = st.selectbox("Task", sorted(errors.task.unique()) if not errors.empty else [])
    with col3:
        track = st.selectbox(
            "Track",
            sorted(errors.track.unique()) if not errors.empty else [],
            format_func=lambda t: TRACK_LABELS.get(t, t),
        )

    if errors.empty:
        st.info("No raw error records available.")
        return

    sub = errors[(errors.model == model) & (errors.task == task) & (errors.track == track)].copy()
    sub["category"] = sub["error"].astype(str).map(classify_error)
    categories = ["all", *sorted(sub["category"].unique().tolist())]
    category = st.selectbox("Category", categories)
    if category != "all":
        sub = sub[sub["category"] == category]

    st.caption(f"{len(sub)} error rows.")
    st.dataframe(
        sub[["document_id", "category", "error"]].head(200),
        use_container_width=True,
        hide_index=True,
    )
