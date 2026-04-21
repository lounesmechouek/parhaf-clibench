"""Render every figure consumed by ``results/claude_analysis.md``.

Reads parquet artefacts under ``results/artifacts/`` and writes PNG exports
under ``results/figures/``. Copies the same PNGs into
``app_claude/assets/figures/`` so the Streamlit app can surface them as a
fallback when it cannot render Plotly figures interactively.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from parhaf_clinbench.reporting.plots_extended import (  # noqa: E402
    error_taxonomy_stacked_bar,
    fewshot_slopegraph,
    forest_plot,
    global_leaderboard,
    latency_box,
    leaderboard_bar,
    pareto_f1_vs_latency,
    paired_delta_caterpillar,
    robustness_heatmap,
    subgroup_small_multiples,
)


ARTIFACTS = REPO_ROOT / "results" / "artifacts"
FIG_DIR = REPO_ROOT / "results" / "figures"
APP_FIG_DIR = REPO_ROOT / "app_claude" / "assets" / "figures"


def _save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    APP_FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / f"{name}.png"
    fig.write_image(str(out), width=1100, height=int(fig.layout.height or 500), scale=2)
    shutil.copy(out, APP_FIG_DIR / out.name)
    print(f"  wrote {out.relative_to(REPO_ROOT)}")


def main() -> None:
    scores = pd.read_parquet(ARTIFACTS / "scores.parquet")
    global_scores = pd.read_parquet(ARTIFACTS / "global_scores.parquet")
    robustness = pd.read_parquet(ARTIFACTS / "robustness.parquet")
    timings = pd.read_parquet(ARTIFACTS / "timings.parquet")
    subgroups = pd.read_parquet(ARTIFACTS / "subgroups.parquet")
    taxonomy = pd.read_parquet(ARTIFACTS / "error_taxonomy.parquet")
    deltas = pd.read_parquet(ARTIFACTS / "vs_baseline.parquet")

    _save(global_leaderboard(global_scores), "global_leaderboard")
    _save(forest_plot(scores, track="zero-shot"), "forest_zero_shot")
    _save(forest_plot(scores, track="few-shot"), "forest_few_shot")
    for task in ("pseudo", "infectio", "response", "scenario"):
        _save(leaderboard_bar(scores, track="zero-shot", task=task), f"leaderboard_zs_{task}")
        _save(leaderboard_bar(scores, track="few-shot", task=task), f"leaderboard_fs_{task}")
        _save(fewshot_slopegraph(scores, task=task), f"slopegraph_{task}")

    _save(robustness_heatmap(robustness, metric="schema_conformity_rate"), "robustness_schema")
    _save(robustness_heatmap(robustness, metric="empty_output_rate"), "robustness_empty")
    _save(latency_box(timings), "latency_box")
    _save(pareto_f1_vs_latency(global_scores, robustness, track="zero-shot"), "pareto_zero_shot")

    _save(
        subgroup_small_multiples(subgroups, task="pseudo", subgroup_kind="length_quartile"),
        "subgroups_pseudo_length",
    )
    _save(
        subgroup_small_multiples(subgroups, task="infectio", subgroup_kind="negation"),
        "subgroups_infectio_negation",
    )
    _save(
        subgroup_small_multiples(subgroups, task="scenario", subgroup_kind="speciality"),
        "subgroups_scenario_speciality",
    )

    _save(error_taxonomy_stacked_bar(taxonomy), "error_taxonomy")

    # Normalise delta columns to the shape expected by the caterpillar plot.
    # vs_baseline uses ``task_a`` rather than ``task``.
    d = deltas.rename(columns={"task_a": "task"}).copy()
    _save(paired_delta_caterpillar(d), "delta_vs_gliner2")

    print("done.")


if __name__ == "__main__":
    main()
