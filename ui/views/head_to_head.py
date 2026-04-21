"""Streamlit page for paired model-versus-model comparisons.

The benchmark uses a paired document-level bootstrap so users can compare two
systems on the exact same resampled clinical notes. This is the statistically
sound way to answer whether one model meaningfully beats another.
"""

from __future__ import annotations

import streamlit as st

from parhaf_clinbench.reporting.plots_extended import paired_delta_caterpillar
from ui.data_loader import load_paired_deltas, load_scores
from ui.theme import TRACK_LABELS


def render() -> None:
    """Render the paired delta view for any two selected models."""

    st.title("Head-to-head")
    st.caption("Paired document-level bootstrap deltas between any two models.")

    pairs = load_paired_deltas()
    scores = load_scores()
    if pairs.empty:
        st.warning("Precomputed artefacts not found.")
        return

    models = sorted(set(pairs["model_a"]) | set(pairs["model_b"]))
    col1, col2, col3 = st.columns(3)
    with col1:
        model_a = st.selectbox("Model A", models, index=0)
    with col2:
        default_b = 1 if len(models) > 1 else 0
        model_b = st.selectbox("Model B", models, index=default_b)
    with col3:
        track = st.selectbox(
            "Track",
            sorted(pairs["track_a"].unique()) if "track_a" in pairs.columns else sorted(pairs["track"].unique()),
            format_func=lambda t: TRACK_LABELS.get(t, t),
        )

    if model_a == model_b:
        st.info("Pick two different models.")
        return

    track_col = "track_a" if "track_a" in pairs.columns else "track"
    task_col = "task_a" if "task_a" in pairs.columns else "task"
    mask = pairs[track_col] == track
    fwd = pairs[mask & (pairs.model_a == model_a) & (pairs.model_b == model_b)]
    rev = pairs[mask & (pairs.model_a == model_b) & (pairs.model_b == model_a)]

    if fwd.empty and not rev.empty:
        deltas = rev.assign(
            delta=-rev["delta"],
            ci_low=-rev["ci_high"],
            ci_high=-rev["ci_low"],
            model_a=model_a,
            model_b=model_b,
        )
    else:
        deltas = fwd

    if deltas.empty:
        st.info("No paired record for this combination - GLiNER2 has only the zero-shot track.")
        return

    display = deltas.rename(columns={task_col: "task"})
    st.plotly_chart(paired_delta_caterpillar(display), use_container_width=True)
    st.caption(
        "Positive delta = Model A is better. A CI not crossing zero means the "
        "difference is significant under the paired bootstrap."
    )

    st.markdown("### Side-by-side F1")
    scores_off = scores[scores.metric_kind == "official"]
    table = (
        scores_off[(scores_off.model.isin([model_a, model_b])) & (scores_off.track == track)]
        .pivot_table(index="task", columns="model", values="f1")
        .round(4)
        .reindex(["pseudo", "infectio", "response", "scenario"])
    )
    st.dataframe(table, use_container_width=True)
