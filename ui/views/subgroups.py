"""Streamlit page for subgroup benchmark analyses.

Subgroup slices reveal whether performance is stable across report lengths,
clinical specialities, labels, or polarities. That matters for PARHAF because
an acceptable average can still hide unsafe failures on clinically important
minority slices.
"""

from __future__ import annotations

import streamlit as st

from parhaf_clinbench.reporting.plots_extended import subgroup_small_multiples
from ui.data_loader import load_subgroups
from ui.theme import TASK_LABELS, TRACK_LABELS


def render() -> None:
    """Render the subgroup selector and the corresponding charts."""

    st.title("Subgroup analysis")
    df = load_subgroups()
    if df.empty:
        st.warning("Precomputed artefacts not found.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        task = st.selectbox(
            "Task",
            sorted(df.task.unique()),
            format_func=lambda t: TASK_LABELS.get(t, t),
        )
    with col2:
        kinds_for_task = sorted(df[df.task == task]["subgroup_kind"].unique())
        kind = st.selectbox("Subgroup axis", kinds_for_task)
    with col3:
        track = st.selectbox(
            "Track",
            sorted(df.track.unique()),
            format_func=lambda t: TRACK_LABELS.get(t, t),
        )

    st.plotly_chart(
        subgroup_small_multiples(df, task=task, subgroup_kind=kind, track=track),
        use_container_width=True,
    )

    st.markdown("### Table")
    view = df[
        (df.task == task) & (df.subgroup_kind == kind) & (df.track == track)
    ][["model", "subgroup", "precision", "recall", "f1", "n_docs"]].round(4)
    st.dataframe(view, use_container_width=True, hide_index=True)
