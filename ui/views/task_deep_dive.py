"""Streamlit page for task-by-task score inspection.

The benchmark treats the four PARHAF-derived tasks as complementary clinical
capabilities. This page exposes how each model behaves on one task at a time
and whether few-shot prompting changes the outcome materially.
"""

from __future__ import annotations

import streamlit as st

from parhaf_clinbench.reporting.plots_extended import (
    fewshot_slopegraph,
    leaderboard_bar,
)
from ui.data_loader import load_fewshot_vs_zeroshot, load_scores
from ui.theme import TASK_LABELS


def render() -> None:
    """Render per-task score charts and few-shot lift diagnostics."""

    st.title("Task deep dive")
    scores = load_scores()
    fs_vs_zs = load_fewshot_vs_zeroshot()
    if scores.empty:
        st.warning("Precomputed artefacts not found.")
        return

    task = st.selectbox(
        "Task",
        ["pseudo", "infectio", "response", "scenario"],
        format_func=lambda t: TASK_LABELS.get(t, t),
    )

    tab_scores, tab_lift, tab_raw = st.tabs(["Scores", "Few-shot lift", "Raw table"])

    with tab_scores:
        col_zs, col_fs = st.columns(2)
        with col_zs:
            st.plotly_chart(
                leaderboard_bar(scores, track="zero-shot", task=task),
                use_container_width=True,
            )
        with col_fs:
            st.plotly_chart(
                leaderboard_bar(scores, track="few-shot", task=task),
                use_container_width=True,
            )

    with tab_lift:
        st.plotly_chart(fewshot_slopegraph(scores, task=task), use_container_width=True)
        st.markdown("**Paired bootstrap delta** : few-shot - zero-shot per model:")
        deltas = fs_vs_zs[fs_vs_zs["task_a"] == task][
            ["model_a", "delta", "ci_low", "ci_high", "n_docs", "status"]
        ].round(4).rename(columns={"model_a": "model"})
        st.dataframe(deltas, use_container_width=True, hide_index=True)
        st.caption(
            "A CI not crossing 0 means the few-shot effect is significant under the paired "
            "document-level bootstrap."
        )

    with tab_raw:
        raw = scores[(scores.task == task)][
            ["model", "track", "metric_kind", "metric_name", "precision", "recall", "f1", "ci_low", "ci_high"]
        ].round(4)
        st.dataframe(raw, use_container_width=True, hide_index=True)
