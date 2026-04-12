"""Streamlit page for operational robustness and efficiency metrics.

Clinical extraction quality is only useful if the system reliably emits valid
structured outputs and does so within practical latency bounds. This page puts
those operational constraints next to the benchmark scores.
"""

from __future__ import annotations

import streamlit as st

from parhaf_clinbench.reporting.plots_extended import (
    latency_box,
    pareto_f1_vs_latency,
    robustness_heatmap,
)
from ui.data_loader import load_global_scores, load_robustness, load_timings


def render() -> None:
    """Render robustness heatmaps, latency distributions, and the Pareto front."""

    st.title("Robustness & efficiency")
    robustness = load_robustness()
    timings = load_timings()
    global_scores = load_global_scores()
    if robustness.empty:
        st.warning("Precomputed artefacts not found.")
        return

    tabs = st.tabs(["Schema conformity", "Empty-output rate", "Latency", "Pareto front"])
    with tabs[0]:
        st.plotly_chart(robustness_heatmap(robustness, metric="schema_conformity_rate"), use_container_width=True)
        st.dataframe(
            robustness.pivot_table(
                index=["model", "track"], columns="task", values="schema_conformity_rate"
            ).round(3),
            use_container_width=True,
        )
    with tabs[1]:
        st.plotly_chart(robustness_heatmap(robustness, metric="empty_output_rate"), use_container_width=True)
        st.dataframe(
            robustness.pivot_table(
                index=["model", "track"], columns="task", values="empty_output_rate"
            ).round(3),
            use_container_width=True,
        )
    with tabs[2]:
        st.plotly_chart(latency_box(timings), use_container_width=True)
        st.dataframe(
            robustness.pivot_table(
                index=["model", "track"],
                columns="task",
                values="latency_median_ms",
            ).round(0),
            use_container_width=True,
        )
    with tabs[3]:
        st.plotly_chart(
            pareto_f1_vs_latency(global_scores, robustness, track="zero-shot"),
            use_container_width=True,
        )
