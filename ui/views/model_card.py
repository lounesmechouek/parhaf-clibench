"""Streamlit page summarising one model across benchmark dimensions.

The model card consolidates task scores, robustness metrics, timing behaviour,
and error-taxonomy counts for a single system. It is the fastest way to judge
whether a model's headline ranking is supported by clinically acceptable
operational behaviour.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.data_loader import (
    load_error_taxonomy,
    load_robustness,
    load_run_metadata,
    load_scores,
    load_timings,
)
from ui.theme import PALETTE, TASK_LABELS, TRACK_LABELS


def _radar(scores: pd.DataFrame, model: str) -> go.Figure:
    """Build the task radar chart for one model across available tracks."""

    df = scores[(scores.model == model) & (scores.metric_kind == "official")]
    tasks = ["pseudo", "infectio", "response", "scenario"]
    fig = go.Figure()
    for i, track in enumerate(sorted(df.track.unique())):
        sub = df[df.track == track].set_index("task").reindex(tasks)
        values = sub["f1"].fillna(0).tolist()
        fig.add_trace(
            go.Scatterpolar(
                r=[*values, values[0]],
                theta=[TASK_LABELS[t] for t in tasks] + [TASK_LABELS[tasks[0]]],
                fill="toself",
                name=TRACK_LABELS.get(track, track),
                line=dict(color=PALETTE[i]),
            )
        )
    fig.update_layout(
        polar=dict(radialaxis=dict(range=[0, 1], visible=True)),
        template="plotly_white",
        height=420,
        margin=dict(l=30, r=30, t=40, b=30),
    )
    return fig


def render() -> None:
    """Render the benchmark card for the selected model."""

    st.title("Model card")
    scores = load_scores()
    robustness = load_robustness()
    timings = load_timings()
    runmeta = load_run_metadata()
    taxonomy = load_error_taxonomy()
    if scores.empty:
        st.warning("Precomputed artefacts not found.")
        return

    model = st.selectbox("Model", sorted(scores.model.unique()))

    meta_row = runmeta[runmeta.model == model]
    if not meta_row.empty:
        row = meta_row.iloc[0]
        st.caption(
            f"**HF id:** `{row.get('model_hf_id', '—')}` · **Runtime:** "
            f"{row.get('runtime_name', '—')} {row.get('runtime_version', '')} · "
            f"**GPU:** {row.get('gpu_name', '—')}"
        )

    col1, col2 = st.columns([1, 1])
    with col1:
        st.plotly_chart(_radar(scores, model), use_container_width=True)
    with col2:
        table = (
            scores[(scores.model == model) & (scores.metric_kind == "official")]
            .pivot_table(index="task", columns="track", values="f1")
            .round(4)
            .reindex(["pseudo", "infectio", "response", "scenario"])
        )
        st.markdown("**Official F1 per task / track**")
        st.dataframe(table, use_container_width=True)

    st.markdown("### Robustness metrics")
    rob = robustness[robustness.model == model][
        [
            "track",
            "task",
            "raw_json_valid_rate",
            "schema_conformity_rate",
            "empty_output_rate",
            "latency_median_ms",
            "throughput_tokens_per_second",
        ]
    ].round(3)
    st.dataframe(rob, use_container_width=True, hide_index=True)

    st.markdown("### Latency distribution (all tasks, all tracks)")
    sub = timings[timings.model == model]
    if not sub.empty:
        fig = go.Figure(
            go.Histogram(x=sub["latency_ms"], nbinsx=60, marker_color=PALETTE[0])
        )
        fig.update_layout(
            template="plotly_white",
            xaxis_title="Latency (ms)",
            yaxis_title="Document count",
            height=320,
            margin=dict(l=60, r=20, t=20, b=50),
        )
        st.plotly_chart(fig, use_container_width=True)

    tax = taxonomy[taxonomy.model == model]
    if not tax.empty:
        st.markdown("### Error taxonomy")
        st.dataframe(
            tax.pivot_table(index=["task", "track"], columns="category", values="count", fill_value=0),
            use_container_width=True,
        )
