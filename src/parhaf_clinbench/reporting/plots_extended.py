"""Plotly figure builders used by the notebook and the Streamlit app.

Every figure takes a tidy dataframe (one of the frames produced under
``reporting.analysis.frames``) and returns a configured ``plotly.graph_objects.Figure``.
Titles and axis labels are always in English. Colours follow a single
categorical palette so figures stay visually consistent across surfaces.
"""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

PALETTE = [
    "#4C72B0",  # blue
    "#DD8452",  # orange
    "#55A868",  # green
    "#C44E52",  # red
    "#8172B3",  # purple
    "#937860",  # brown
    "#DA8BC3",  # pink
    "#8C8C8C",  # grey
]

TASK_ORDER = ["pseudo", "infectio", "response", "scenario"]
TASK_LABELS = {
    "pseudo": "Pseudonymization",
    "infectio": "Infectiology",
    "response": "Response to treatment",
    "scenario": "Structured scenario",
}
TRACK_LABELS = {"zero-shot": "Zero-shot", "few-shot": "Few-shot"}


def _color_map(models: Iterable[str]) -> dict[str, str]:
    models = list(dict.fromkeys(models))
    return {m: PALETTE[i % len(PALETTE)] for i, m in enumerate(sorted(models))}


def apply_theme(fig: go.Figure, *, title: str | None = None, height: int = 450) -> go.Figure:
    """Apply a consistent theme to any figure built in this module."""

    fig.update_layout(
        template="plotly_white",
        title=title,
        height=height,
        margin=dict(l=60, r=20, t=60, b=60),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        font=dict(family="Inter, Helvetica, Arial, sans-serif", size=12),
    )
    return fig


# ---------------------------------------------------------------------------
# Leaderboard / task-level figures
# ---------------------------------------------------------------------------


def leaderboard_bar(
    scores: pd.DataFrame,
    *,
    track: str,
    task: str,
) -> go.Figure:
    """Horizontal leaderboard bar with bootstrap CI whiskers for one task/track."""

    df = scores[
        (scores["track"] == track)
        & (scores["task"] == task)
        & (scores["metric_kind"] == "official")
    ].copy()
    df = df.sort_values("f1", ascending=True)
    err_plus = (df["ci_high"] - df["f1"]).clip(lower=0)
    err_minus = (df["f1"] - df["ci_low"]).clip(lower=0)
    colors = _color_map(df["model"].tolist())
    fig = go.Figure(
        go.Bar(
            x=df["f1"],
            y=df["model"],
            orientation="h",
            marker=dict(color=[colors[m] for m in df["model"]]),
            error_x=dict(type="data", symmetric=False, array=err_plus, arrayminus=err_minus),
            text=[f"{v:.3f}" for v in df["f1"]],
            textposition="outside",
        )
    )
    fig.update_xaxes(title="Micro-F1 (official)", range=[0, max(1.0, df["ci_high"].max() * 1.05)])
    fig.update_yaxes(title="")
    return apply_theme(
        fig,
        title=f"{TASK_LABELS.get(task, task)} — {TRACK_LABELS.get(track, track)}",
        height=max(350, 40 * len(df) + 160),
    )


def global_leaderboard(global_scores: pd.DataFrame) -> go.Figure:
    """Grouped leaderboard showing the global (task-mean) F1 per track per model."""

    df = global_scores.copy()
    df["track_label"] = df["track"].map(TRACK_LABELS).fillna(df["track"])
    fig = px.bar(
        df.sort_values(["track", "global_f1"], ascending=[True, True]),
        x="global_f1",
        y="model",
        color="track_label",
        barmode="group",
        orientation="h",
        error_x=df["ci_high"] - df["global_f1"],
        error_x_minus=df["global_f1"] - df["ci_low"],
        color_discrete_sequence=PALETTE,
        labels={"global_f1": "Global F1 (mean of 4 tasks)", "model": "", "track_label": "Track"},
    )
    return apply_theme(fig, title="Global leaderboard — equal-weight task average", height=500)


def forest_plot(scores: pd.DataFrame, *, track: str) -> go.Figure:
    """Per-task forest plot of F1 ± 95% bootstrap CI across all models."""

    df = scores[(scores["track"] == track) & (scores["metric_kind"] == "official")].copy()
    df["task_label"] = df["task"].map(TASK_LABELS).fillna(df["task"])
    df["task_rank"] = df["task"].map({t: i for i, t in enumerate(TASK_ORDER)})
    df = df.sort_values(["task_rank", "f1"], ascending=[True, False])
    colors = _color_map(df["model"].unique())
    fig = go.Figure()
    for model, sub in df.groupby("model", sort=False):
        model_name = str(model)
        fig.add_trace(
            go.Scatter(
                x=sub["f1"],
                y=[f"{row.task_label} — {row.model}" for row in sub.itertuples()],
                mode="markers",
                name=model_name,
                marker=dict(color=colors[model_name], size=10),
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=(sub["ci_high"] - sub["f1"]).clip(lower=0),
                    arrayminus=(sub["f1"] - sub["ci_low"]).clip(lower=0),
                    color=colors[model_name],
                ),
            )
        )
    fig.update_xaxes(title="Official micro-F1", range=[0, 1])
    return apply_theme(
        fig,
        title=f"Per-task forest plot — {TRACK_LABELS.get(track, track)}",
        height=max(500, 30 * len(df)),
    )


# ---------------------------------------------------------------------------
# Few-shot lift
# ---------------------------------------------------------------------------


def fewshot_slopegraph(scores: pd.DataFrame, *, task: str) -> go.Figure:
    """Slopegraph of zero-shot -> few-shot for all models on one task."""

    df = scores[(scores["task"] == task) & (scores["metric_kind"] == "official")]
    pivot = df.pivot_table(index="model", columns="track", values="f1", aggfunc="first")
    if "zero-shot" not in pivot.columns or "few-shot" not in pivot.columns:
        return apply_theme(go.Figure(), title="Few-shot lift unavailable")
    colors = _color_map(pivot.index.tolist())
    fig = go.Figure()
    for model in pivot.index:
        fig.add_trace(
            go.Scatter(
                x=["Zero-shot", "Few-shot"],
                y=[pivot.loc[model, "zero-shot"], pivot.loc[model, "few-shot"]],
                mode="lines+markers+text",
                name=model,
                text=[f"{pivot.loc[model, 'zero-shot']:.2f}", f"{pivot.loc[model, 'few-shot']:.2f}"],
                textposition=["middle left", "middle right"],
                line=dict(color=colors[model], width=2),
                marker=dict(color=colors[model], size=10),
            )
        )
    fig.update_yaxes(title="Official micro-F1", range=[0, 1])
    return apply_theme(fig, title=f"Few-shot lift — {TASK_LABELS.get(task, task)}")


# ---------------------------------------------------------------------------
# Robustness and operational metrics
# ---------------------------------------------------------------------------


def robustness_heatmap(robustness: pd.DataFrame, *, metric: str = "schema_conformity_rate") -> go.Figure:
    """Model x task heatmap of a robustness metric on the zero-shot track."""

    df = robustness[robustness["track"] == "zero-shot"].copy()
    pivot = df.pivot_table(index="model", columns="task", values=metric, aggfunc="first")
    pivot = pivot[[c for c in TASK_ORDER if c in pivot.columns]]
    pretty = metric.replace("_", " ").title()
    fig = go.Figure(
        go.Heatmap(
            z=pivot.values,
            x=[TASK_LABELS.get(c, c) for c in pivot.columns],
            y=pivot.index,
            colorscale="Blues",
            zmin=0,
            zmax=1,
            colorbar=dict(title=pretty),
            text=[[f"{v:.2f}" for v in row] for row in pivot.values],
            texttemplate="%{text}",
        )
    )
    return apply_theme(fig, title=f"{pretty} — zero-shot", height=420)


def latency_box(timings: pd.DataFrame) -> go.Figure:
    """Per-model latency distribution (box plot)."""

    colors = _color_map(timings["model"].unique())
    fig = go.Figure()
    for model, sub in timings.groupby("model"):
        model_name = str(model)
        fig.add_trace(
            go.Box(
                y=sub["latency_ms"],
                name=model_name,
                marker_color=colors[model_name],
                boxmean=True,
            )
        )
    fig.update_yaxes(title="Latency (ms)", type="log")
    return apply_theme(fig, title="Per-document latency distribution (log scale)")


def pareto_f1_vs_latency(
    global_scores: pd.DataFrame,
    robustness: pd.DataFrame,
    *,
    track: str = "zero-shot",
) -> go.Figure:
    """F1 versus median latency scatter with Pareto front highlighted."""

    lat = robustness[robustness["track"] == track].groupby("model")["latency_median_ms"].mean().reset_index()
    merged = global_scores[global_scores["track"] == track].merge(lat, on="model")
    merged = merged.sort_values("latency_median_ms")
    colors = _color_map(merged["model"].tolist())
    # Pareto front — strictly higher F1 than any cheaper (lower latency) model.
    pareto_mask = []
    best_so_far = -1.0
    for _, row in merged.iterrows():
        pareto_mask.append(row["global_f1"] > best_so_far)
        best_so_far = max(best_so_far, row["global_f1"])
    merged = merged.assign(on_pareto=pareto_mask)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=merged["latency_median_ms"],
            y=merged["global_f1"],
            mode="markers+text",
            text=merged["model"],
            textposition="top center",
            marker=dict(
                color=[colors[m] for m in merged["model"]],
                size=[18 if p else 12 for p in merged["on_pareto"]],
                line=dict(width=[2 if p else 0 for p in merged["on_pareto"]], color="black"),
            ),
        )
    )
    pareto_df = merged[merged["on_pareto"]].sort_values("latency_median_ms")
    if not pareto_df.empty:
        fig.add_trace(
            go.Scatter(
                x=pareto_df["latency_median_ms"],
                y=pareto_df["global_f1"],
                mode="lines",
                name="Pareto front",
                line=dict(color="#222", dash="dot"),
            )
        )
    fig.update_xaxes(title="Median latency (ms, log)", type="log")
    fig.update_yaxes(title="Global F1", range=[0, 1])
    return apply_theme(fig, title=f"Efficiency frontier — {TRACK_LABELS.get(track, track)}")


# ---------------------------------------------------------------------------
# Subgroups and errors
# ---------------------------------------------------------------------------


def subgroup_small_multiples(
    subgroup_df: pd.DataFrame,
    *,
    task: str,
    subgroup_kind: str,
    track: str = "zero-shot",
) -> go.Figure:
    """Small multiples (facet by subgroup) of F1 across models."""

    df = subgroup_df[
        (subgroup_df["task"] == task)
        & (subgroup_df["track"] == track)
        & (subgroup_df["subgroup_kind"] == subgroup_kind)
    ].copy()
    if df.empty:
        return apply_theme(go.Figure(), title="No data")
    fig = px.bar(
        df.sort_values(["subgroup", "f1"]),
        x="f1",
        y="model",
        color="model",
        facet_col="subgroup",
        facet_col_wrap=3,
        orientation="h",
        color_discrete_sequence=PALETTE,
        labels={"f1": "F1", "model": ""},
    )
    fig.update_xaxes(range=[0, 1])
    return apply_theme(
        fig,
        title=f"{TASK_LABELS.get(task, task)} — F1 by {subgroup_kind.replace('_', ' ')} ({TRACK_LABELS.get(track, track)})",
        height=max(400, 90 * df["subgroup"].nunique()),
    )


def error_taxonomy_stacked_bar(taxonomy: pd.DataFrame) -> go.Figure:
    """Stacked bar of error categories per model (all tasks, zero-shot)."""

    df = taxonomy[taxonomy["track"] == "zero-shot"].copy()
    if df.empty:
        return apply_theme(go.Figure(), title="No error records")
    pivot = df.groupby(["model", "category"])["count"].sum().unstack(fill_value=0)
    category_order = [
        "invalid_json",
        "offset_drift",
        "label_oov",
        "negation_oov",
        "speciality_oov",
        "missing_field",
        "other_schema",
        "unknown",
    ]
    ordered_cols = [c for c in category_order if c in pivot.columns]
    pivot = pivot[ordered_cols]
    fig = go.Figure()
    for idx, cat in enumerate(ordered_cols):
        fig.add_trace(
            go.Bar(
                y=pivot.index,
                x=pivot[cat],
                name=cat,
                orientation="h",
                marker_color=PALETTE[idx % len(PALETTE)],
            )
        )
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title="Error count (zero-shot)")
    return apply_theme(fig, title="Error taxonomy — zero-shot")


def paired_delta_caterpillar(deltas: pd.DataFrame) -> go.Figure:
    """Caterpillar plot of paired bootstrap deltas (A - B) with 95% CI."""

    df = deltas.copy()
    df["label"] = df["task"] + " — " + df["model_a"] + " vs " + df["model_b"]
    df = df.sort_values("delta")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["delta"],
            y=df["label"],
            mode="markers",
            marker=dict(size=10, color=PALETTE[0]),
            error_x=dict(
                type="data",
                symmetric=False,
                array=(df["ci_high"] - df["delta"]).clip(lower=0),
                arrayminus=(df["delta"] - df["ci_low"]).clip(lower=0),
            ),
        )
    )
    fig.add_vline(x=0, line_dash="dot", line_color="#666")
    fig.update_xaxes(title="F1 delta (A - B)")
    return apply_theme(fig, title="Paired bootstrap deltas (95% CI)", height=max(400, 25 * len(df)))
