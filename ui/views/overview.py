"""Streamlit landing page for the benchmark publication bundle.

This overview distils the PARHAF-LM-CLINBENCH results into the questions a
clinical reviewer or benchmark consumer asks first: which system is best per
task, how close the field is to deployment, and what workload each model is
credible for.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui.data_loader import load_global_scores, load_manifest, load_scores

# ---------------------------------------------------------------------------
# Clinical framing metadata
# ---------------------------------------------------------------------------

TASK_ORDER = ["pseudo", "infectio", "response", "scenario"]

TASK_META: dict[str, dict[str, str]] = {
    "pseudo": {
        "label": "Pseudonymization",
        "emoji": "🔒",
        "clinical": (
            "Detect and locate identifying spans in a discharge report so the "
            "document can be safely shared for research or secondary use."
        ),
    },
    "infectio": {
        "label": "Infectiology extraction",
        "emoji": "🦠",
        "clinical": (
            "Extract bacteria, infections and infection sites, plus whether "
            "they are actually present or explicitly ruled out."
        ),
    },
    "response": {
        "label": "Response to treatment",
        "emoji": "💊",
        "clinical": (
            "Pull the textual justification a clinician wrote to classify a "
            "patient's response to an ongoing treatment."
        ),
    },
    "scenario": {
        "label": "Structured scenario",
        "emoji": "🩺",
        "clinical": (
            "Turn a free-text discharge summary into the follwing structured fields : age, "
            "sex, admission mode, primary diagnosis and more."
        ),
    },
}

CLINICAL_RECOMMENDATIONS: dict[str, dict[str, str]] = {
    "pseudo": {
        "tone": "amber",
        "icon": "🔒",
        "title": "Document de-identification",
        "body": (
            "Use <code>GLiNER2</code>. It reaches F1 ≈ 0.47 on exact-span "
            "pseudonymization and is the only system that emits valid "
            "character offsets for every document. You still need a human "
            "review pass before any release, the metric is strict, but the "
            "residual error is enough to leak names and dates."
        ),
    },
    "infectio": {
        "tone": "red",
        "icon": "🦠",
        "title": "Infectiology extraction",
        "body": (
            "Best system today is <code>GLiNER2</code> at F1 ≈ 0.19, with "
            "few-shot <code>ministral_8b</code> close behind at 0.15. "
            "Neither is good enough for pharmacovigilance or surveillance "
            "workflows. Treat this task as research-stage and use models "
            "only to pre-flag candidate mentions for a human reviewer."
        ),
    },
    "response": {
        "tone": "red",
        "icon": "💊",
        "title": "Treatment response",
        "body": (
            "Few-shot <code>ministral_8b</code> and <code>qwen25_7b</code> "
            "are the strongest (F1 ≈ 0.15 and 0.13). The signal is too "
            "noisy for prospective scoring, but the extractor can be used "
            "as a triage aid on retrospective cohorts."
        ),
    },
    "scenario": {
        "tone": "amber",
        "icon": "🩺",
        "title": "Structured scenario fields",
        "body": (
            "Few-shot <code>gemma2_9b</code> wins clearly (F1 ≈ 0.47) and "
            "few-shot <code>ministral_8b</code> is a lighter-weight "
            "alternative (F1 ≈ 0.37, roughly 4x faster). Viable for EHR "
            "pre-filling with a human-in-the-loop reviewer."
        ),
    },
}


# ---------------------------------------------------------------------------
# Derivations
# ---------------------------------------------------------------------------


@dataclass
class TaskWinner:
    """Best observed configuration for one benchmark task."""

    task: str
    model: str
    track: str
    f1: float
    ci_low: float
    ci_high: float


def _readiness(f1: float) -> tuple[str, str]:
    """Map an F1 score to a conservative clinical-readiness label."""

    if f1 >= 0.70:
        return ("green", "Production-ready")
    if f1 >= 0.40:
        return ("amber", "Pilot only")
    return ("red", "Research only")


def _best_per_task(off: pd.DataFrame) -> list[TaskWinner]:
    """Return the top-scoring model/track pair for each clinical task."""

    winners: list[TaskWinner] = []
    for task in TASK_ORDER:
        sub = off[off.task == task]
        if sub.empty:
            continue
        row = sub.sort_values("f1", ascending=False).iloc[0]
        ci_low = float(row["ci_low"]) if pd.notna(row["ci_low"]) else math.nan
        ci_high = float(row["ci_high"]) if pd.notna(row["ci_high"]) else math.nan
        winners.append(
            TaskWinner(
                task=task,
                model=str(row["model"]),
                track=str(row["track"]),
                f1=float(row["f1"]),
                ci_low=ci_low,
                ci_high=ci_high,
            )
        )
    return winners


def _task_card_html(winner: TaskWinner) -> str:
    """Build the HTML task card displayed on the landing page."""

    meta = TASK_META[winner.task]
    tone, label = _readiness(winner.f1)
    ci = ""
    if not math.isnan(winner.ci_low) and not math.isnan(winner.ci_high):
        ci = f"95% CI [{winner.ci_low:.2f} - {winner.ci_high:.2f}]"
    return (
        f'<div class="task-card">'
        f'<h3>{meta["emoji"]} {meta["label"]}</h3>'
        f'<div class="task-desc">{meta["clinical"]}</div>'
        f'<div class="winner-label">Best model</div>'
        f'<div class="winner-name">{winner.model}</div>'
        f'<div class="winner-track">{winner.track}</div>'
        f'<div class="score-row">'
        f'<span class="score-value">{winner.f1:.2f}</span>'
        f'<span class="score-unit">F1</span>'
        f"</div>"
        f'<div class="score-ci">{ci}</div>'
        f'<div class="readiness readiness-{tone}">{label}</div>'
        f"</div>"
    )


def _best_per_task_figure(winners: list[TaskWinner]) -> go.Figure:
    """Horizontal bar, one winner per task, colour-coded by readiness tier."""

    colors = {"green": "#1f7a3a", "amber": "#b9771c", "red": "#922b21"}
    tones = [_readiness(w.f1)[0] for w in winners]
    labels = [
        f"{TASK_META[w.task]['emoji']} {TASK_META[w.task]['label']}"
        for w in winners
    ]
    hover = [
        f"<b>{TASK_META[w.task]['label']}</b><br>"
        f"{w.model} - {w.track}<br>"
        f"F1 = {w.f1:.3f} (95% CI {w.ci_low:.2f} - {w.ci_high:.2f})"
        for w in winners
    ]
    fig = go.Figure(
        go.Bar(
            y=labels,
            x=[w.f1 for w in winners],
            orientation="h",
            marker=dict(color=[colors[t] for t in tones]),
            text=[f"{w.f1:.2f} - {w.model}" for w in winners],
            textposition="outside",
            hovertemplate="%{customdata}<extra></extra>",
            customdata=hover,
            error_x=dict(
                type="data",
                symmetric=False,
                array=[max(0.0, w.ci_high - w.f1) for w in winners],
                arrayminus=[max(0.0, w.f1 - w.ci_low) for w in winners],
                color="#666",
                thickness=1.2,
            ),
        )
    )
    fig.add_vline(x=0.70, line_dash="dot", line_color="#1f7a3a", opacity=0.45)
    fig.add_vline(x=0.40, line_dash="dot", line_color="#b9771c", opacity=0.45)
    fig.add_annotation(
        x=0.70, y=1.08, xref="x", yref="paper",
        text="Production-ready", showarrow=False,
        font=dict(size=11, color="#1f7a3a"),
    )
    fig.add_annotation(
        x=0.40, y=1.08, xref="x", yref="paper",
        text="Pilot", showarrow=False,
        font=dict(size=11, color="#b9771c"),
    )
    fig.update_xaxes(
        range=[0, 1],
        title="Official micro-F1",
        gridcolor="#eef1f5",
        zeroline=False,
    )
    fig.update_yaxes(autorange="reversed", title="")
    fig.update_layout(
        template="plotly_white",
        height=380,
        margin=dict(l=160, r=90, t=50, b=50),
        font=dict(family="Inter, Helvetica, Arial, sans-serif", size=13),
        showlegend=False,
    )
    return fig


def _system_ranking_figure(global_scores: pd.DataFrame) -> go.Figure:
    """Global ranking, keeps each system's best-performing track."""

    df = (
        global_scores.sort_values("global_f1", ascending=False)
        .groupby("model", as_index=False)
        .first()
        .sort_values("global_f1", ascending=True)
    )
    # Highlight the leader.
    max_score = df["global_f1"].max()
    colors = [
        "#1f7a3a" if abs(v - max_score) < 1e-9 else "#4C72B0"
        for v in df["global_f1"]
    ]
    fig = go.Figure(
        go.Bar(
            y=df["model"],
            x=df["global_f1"],
            orientation="h",
            marker=dict(color=colors),
            text=[
                f"{f:.2f} ({t})"
                for f, t in zip(df["global_f1"], df["track"], strict=True)
            ],
            textposition="outside",
            error_x=dict(
                type="data",
                symmetric=False,
                array=(df["ci_high"] - df["global_f1"]).clip(lower=0),
                arrayminus=(df["global_f1"] - df["ci_low"]).clip(lower=0),
                color="#888",
                thickness=1.2,
            ),
        )
    )
    fig.update_xaxes(
        range=[0, 0.38],
        title="Global F1 - average of the 4 clinical tasks",
        gridcolor="#eef1f5",
        zeroline=False,
    )
    fig.update_yaxes(title="")
    fig.update_layout(
        template="plotly_white",
        height=380,
        margin=dict(l=120, r=110, t=20, b=50),
        font=dict(family="Inter, Helvetica, Arial, sans-serif", size=13),
    )
    return fig


def _recommendation_html(meta: dict[str, str]) -> str:
    """Build the HTML callout for a task-specific clinical recommendation."""

    return (
        f'<div class="callout {meta["tone"]}">'
        f'<b>{meta["icon"]} {meta["title"]}.</b> {meta["body"]}'
        f"</div>"
    )


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


def render() -> None:
    """Render the benchmark landing page and the high-level recommendations."""

    scores = load_scores()
    global_scores = load_global_scores()
    manifest = load_manifest()

    if scores.empty or global_scores.empty:
        st.warning(
            "Precomputed artefacts not found. Run "
            "`results/build_artifacts.py` first."
        )
        return

    off = scores[scores.metric_kind == "official"]
    winners = _best_per_task(off)

    # ----- Header ------------------------------------------------------
    st.markdown(
        "<h1 style='margin-bottom:4px;'>Can small language models read "
        "clinical notes?</h1>",
        unsafe_allow_html=True,
    )
    n_models = manifest.get("n_models", 7)
    n_docs = sum(manifest.get("n_gold_docs", {}).values())
    st.markdown(
        "<div style='color:#6b7385; font-size:1.02rem; margin-bottom:22px;'>"
        f"An independent benchmark of <b>{n_models} systems</b> on <b>four "
        f"information-extraction tasks</b> carried out on more than "
        f"<b>{n_docs:,} human-authored clinical reports</b> from the PARHAF corpus, "
        "with statistical guarantees, full reproducibility and a single unified scoring "
        "pipeline."
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div style='color:#6b7385; font-size:1.02rem; margin-bottom:22px;'>"
        "The evaluated models are limited to 9B parameters to meet efficiency and flexibility requirements for "
        "potential initial deployment in secure clinical environments. They were assessed using zero-shot or, "
        "depending on the model, combined zero- and few-shot settings, without hyperparameter tuning. Further improvements in benchmark "
        "performance could likely be achieved through improved prompt engineering (e.g., reasoning or tool-augmented setups) and fine-tuning.",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="callout amber">'
        "A specialised encoder "
        "(<code>GLiNER2</code>) is still the only system we would recommend "
        "for <b>pseudonymization</b> and <b>infectiology extraction</b> on "
        "French discharge reports. For <b>structured field extraction</b> and "
        "<b>treatment-response justifications</b>, few-shot prompted 7-9 B "
        "LLMs (<code>gemma2_9b</code>, <code>ministral_8b</code>) "
        "<b>surpass</b> the encoder baseline. However none of the models "
        "reach a quality bar we would consider safe for "
        "production use."
        "</div>",
        unsafe_allow_html=True,
    )


    # ----- Task readiness cards ---------------------------------------
    st.markdown(
        '<div class="section-title">Best available model per task</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-sub">Each card shows the single best system '
        "observed across every model and every prompting regime, with a "
        "conservative clinical-readiness badge.</div>",
        unsafe_allow_html=True,
    )
    cols = st.columns(4, gap="medium")
    for col, winner in zip(cols, winners, strict=True):
        with col:
            st.markdown(_task_card_html(winner), unsafe_allow_html=True)

    # ----- Clinical readiness figure ----------------------------------
    st.markdown(
        '<div class="section-title">How close are we to clinical readiness?</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-sub">Dotted lines mark the two thresholds used '
        "on the cards above: 0.70 for production-ready (under supervision) and 0.40 for "
        "pilot-only. Whiskers are 95% bootstrap confidence intervals, "
        "resampled at the document level.</div>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        _best_per_task_figure(winners),
        use_container_width=True,
        config={"displayModeBar": False},
    )
    st.markdown(
        '<div class="callout">'
        "Read this chart as the <b>ceiling</b> of what this benchmark gives "
        "you today. For pseudonymization and structured scenarios the best "
        "score is just under the pilot threshold (≈ 0.47). For infectiology "
        "and treatment response, no system clears even the research bar "
        "the information is too sparse and too context-dependent for small "
        "LLMs in zero-shot, and the specialised encoder only partially "
        "makes up for it."
        "</div>",
        unsafe_allow_html=True,
    )

    # ----- Global ranking ---------------------------------------------
    st.markdown(
        '<div class="section-title">Which system should you trial first?</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-sub">Global F1 averages the four clinical '
        "tasks with equal weights. For each system we keep its best "
        "prompting regime (zero-shot or few-shot fixed).</div>",
        unsafe_allow_html=True,
    )
    st.plotly_chart(
        _system_ranking_figure(global_scores),
        use_container_width=True,
        config={"displayModeBar": False},
    )
    st.markdown(
        '<div class="callout">'
        "Three systems cluster around the top: <code>GLiNER2</code> "
        "(global F1 ≈ 0.21), driven by its pseudo/infectio dominance, and "
        "the two few-shot LLMs <code>ministral_8b</code> and "
        "<code>gemma2_9b</code> (≈ 0.17 each), driven by scenario and "
        "response. Everything else (<code>llama31_8b</code>, "
        "<code>lucie_7b</code>, <code>aya_8b</code>) lags by an order of "
        "magnitude and should not be considered for clinical use today."
        "</div>",
        unsafe_allow_html=True,
    )

    # ----- Recommendations --------------------------------------------
    st.markdown(
        '<div class="section-title">Recommendations by clinical use case</div>',
        unsafe_allow_html=True,
    )
    rec_col1, rec_col2 = st.columns(2, gap="large")
    with rec_col1:
        st.markdown(
            _recommendation_html(CLINICAL_RECOMMENDATIONS["pseudo"]),
            unsafe_allow_html=True,
        )
        st.markdown(
            _recommendation_html(CLINICAL_RECOMMENDATIONS["infectio"]),
            unsafe_allow_html=True,
        )
    with rec_col2:
        st.markdown(
            _recommendation_html(CLINICAL_RECOMMENDATIONS["response"]),
            unsafe_allow_html=True,
        )
        st.markdown(
            _recommendation_html(CLINICAL_RECOMMENDATIONS["scenario"]),
            unsafe_allow_html=True,
        )

    # ----- Where to go next -------------------------------------------
    st.markdown(
        '<div class="section-title">Where to go next</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "- **🏆 Leaderboard** : every system, every task, with confidence intervals.\n"
        "- **🔬 Task deep dive** : per-task comparisons and the few-shot effect.\n"
        "- **⚔️ Head-to-head** : rigorous A/B comparison with paired bootstrap.\n"
        "- **🛡️ Robustness** : can a model actually emit valid JSON? Latency and throughput too.\n"
        "- **🧪 Subgroups** : long reports, medical specialities, label rarity.\n"
        "- **📖 Methodology** : data, metrics, bootstrap and the scoring audit."
    )

    st.caption(
        "All scores shown on this page are independently re-computed from "
        "raw predictions and gold corpora before being displayed : see the "
        "Methodology tab for the scoring audit."
    )
