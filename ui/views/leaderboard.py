"""Streamlit page showing benchmark rankings across tasks and tracks.

The leaderboard is the compact decision layer for the benchmark: it exposes
which system is strongest overall, which task drives that performance, and how
uncertainty changes the interpretation of those ranks.
"""

from __future__ import annotations

import streamlit as st

from parhaf_clinbench.reporting.plots_extended import (
    forest_plot,
    global_leaderboard,
    leaderboard_bar,
)
from ui.data_loader import load_global_scores, load_scores
from ui.theme import TASK_LABELS, TRACK_LABELS


def render() -> None:
    """Render the global ranking and per-task leaderboards."""

    st.title("Leaderboard")
    st.caption("Equal-weight task-average and per-task official F1 with 95% CI.")

    scores = load_scores()
    gs = load_global_scores()
    if scores.empty or gs.empty:
        st.warning("Precomputed artefacts not found. Run `results/build_artifacts.py` first.")
        return

    include_gliner = st.toggle(
        "Include GLiNER2 (encoder baseline, zero-shot only)", value=True
    )
    if not include_gliner:
        gs = gs[gs.model != "gliner2_multi"]
        scores = scores[scores.model != "gliner2_multi"]

    st.plotly_chart(global_leaderboard(gs), use_container_width=True)

    st.markdown("### Per-task leaderboard")
    col1, col2 = st.columns(2)
    with col1:
        track = st.selectbox("Track", ["zero-shot", "few-shot"], format_func=lambda t: TRACK_LABELS.get(t, t))
    with col2:
        task = st.selectbox(
            "Task",
            ["pseudo", "infectio", "response", "scenario"],
            format_func=lambda t: TASK_LABELS.get(t, t),
        )

    st.plotly_chart(leaderboard_bar(scores, track=track, task=task), use_container_width=True)

    with st.expander("Forest plot across all tasks", expanded=False):
        st.plotly_chart(forest_plot(scores, track=track), use_container_width=True)

    with st.expander("Raw scores table", expanded=False):
        display = (
            scores[scores.metric_kind == "official"]
            .pivot_table(index=["model", "track"], columns="task", values="f1")
            .round(4)
            .reset_index()
        )
        st.dataframe(display, use_container_width=True, hide_index=True)
