"""Shared visual primitives for the benchmark dashboard.

The UI intentionally keeps a consistent visual language across pages so users
can compare clinical tasks, prompting tracks, and robustness metrics without
re-learning the presentation on each tab.
"""

from __future__ import annotations

import streamlit as st

PALETTE = [
    "#4C72B0",
    "#DD8452",
    "#55A868",
    "#C44E52",
    "#8172B3",
    "#937860",
    "#DA8BC3",
    "#8C8C8C",
]

TASK_LABELS = {
    "pseudo": "Pseudonymization",
    "infectio": "Infectiology",
    "response": "Response to treatment",
    "scenario": "Structured scenario",
}
TRACK_LABELS = {"zero-shot": "Zero-shot", "few-shot": "Few-shot fixed"}


def inject_css() -> None:
    """Inject the shared CSS rules used across every Streamlit page."""

    st.markdown(
        """
        <style>
        .block-container { padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px; }

        /* ---------- metric cards ---------- */
        .metric-card {
            background: #f6f8fb;
            border: 1px solid #e1e4ea;
            border-radius: 12px;
            padding: 16px 20px;
        }
        .metric-card h4 { margin: 0 0 4px 0; font-size: 0.85rem; color: #475065; text-transform: uppercase; letter-spacing: 0.04em; }
        .metric-card .value { font-size: 1.8rem; font-weight: 700; color: #1b2433; }
        .metric-card .sub { font-size: 0.8rem; color: #6b7385; margin-top: 4px; }

        /* ---------- task readiness cards ---------- */
        .task-card {
            border-radius: 14px;
            padding: 18px 20px 16px 20px;
            border: 1px solid #e1e4ea;
            background: linear-gradient(180deg, #ffffff 0%, #f7f9fc 100%);
            height: 100%;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
        }
        .task-card h3 { margin: 0 0 2px 0; font-size: 1.05rem; color: #1b2433; font-weight: 700; }
        .task-card .task-desc { font-size: 0.82rem; color: #6b7385; margin-bottom: 14px; min-height: 34px; }
        .task-card .winner-label { font-size: 0.72rem; color: #475065; text-transform: uppercase; letter-spacing: 0.05em; }
        .task-card .winner-name { font-size: 1.05rem; font-weight: 700; color: #1b2433; margin-top: 2px; }
        .task-card .winner-track { font-size: 0.78rem; color: #6b7385; }
        .task-card .score-row { display:flex; align-items:baseline; gap:8px; margin-top: 10px; }
        .task-card .score-value { font-size: 2rem; font-weight: 800; color: #1b2433; line-height: 1; }
        .task-card .score-unit { font-size: 0.85rem; color: #6b7385; }
        .task-card .score-ci { font-size: 0.75rem; color: #6b7385; margin-top: 2px; }
        .task-card .readiness {
            display:inline-block; margin-top: 12px;
            padding: 4px 10px; border-radius: 999px;
            font-size: 0.72rem; font-weight: 700;
            text-transform: uppercase; letter-spacing: 0.04em;
        }
        .readiness-green { background: #e7f6ec; color: #1f7a3a; border: 1px solid #bce2c7; }
        .readiness-amber { background: #fff5e0; color: #8a5a00; border: 1px solid #f3d9a0; }
        .readiness-red   { background: #fde8e8; color: #922b21; border: 1px solid #f3c1bc; }

        /* ---------- callouts ---------- */
        .callout {
            border-left: 4px solid #4C72B0;
            background: #f6f8fb;
            padding: 14px 18px;
            border-radius: 8px;
            margin: 10px 0 18px 0;
            color: #1b2433;
            font-size: 0.95rem;
            line-height: 1.55;
        }
        .callout.green  { border-left-color: #1f7a3a; background: #f1faf4; }
        .callout.amber  { border-left-color: #b9771c; background: #fff8ea; }
        .callout.red    { border-left-color: #922b21; background: #fdf1f0; }

        .section-title {
            font-size: 1.35rem; font-weight: 700; color: #1b2433;
            margin: 26px 0 6px 0;
        }
        .section-sub {
            font-size: 0.92rem; color: #6b7385; margin-bottom: 14px;
        }

        .task-tag { display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.75rem; font-weight: 600; color: white; background: #4C72B0; }
        div[data-testid="stDataFrame"] { border-radius: 8px; }

        /* ---------- methodology prose ---------- */
        .method-prose { font-size: 0.96rem; line-height: 1.65; color: #2a3446; }
        .method-prose h3 { margin-top: 1.6rem; color: #1b2433; }
        .method-prose table { font-size: 0.88rem; }
        .method-prose code { background: #f1f3f7; padding: 1px 5px; border-radius: 4px; font-size: 0.85em; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_card(title: str, value: str, sub: str = "") -> str:
    """Build the HTML fragment used for compact KPI cards.

    Args:
        title: Short metric label.
        value: Main value shown in large type.
        sub: Optional explanatory subtitle.

    Returns:
        HTML markup rendered by Streamlit with ``unsafe_allow_html=True``.
    """

    return (
        f'<div class="metric-card"><h4>{title}</h4>'
        f'<div class="value">{value}</div>'
        f'<div class="sub">{sub}</div></div>'
    )
