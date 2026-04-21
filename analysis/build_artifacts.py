"""Build every parquet artefact consumed by the notebook and the Streamlit app.

Run from the repo root with the venv:

    .venv/bin/python analysis/build_artifacts.py

All outputs live under ``analysis/artifacts/`` (parquet) and ``ui/data/``
(same parquets, copied for the Streamlit app bundle).
"""

from __future__ import annotations

import json
import shutil
import sys
from itertools import combinations
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from parhaf_clinbench.reporting import load_run_suite  # noqa: E402
from parhaf_clinbench.reporting.analysis.error_taxonomy import build_error_taxonomy  # noqa: E402
from parhaf_clinbench.reporting.analysis.frames import (  # noqa: E402
    build_errors_frame,
    build_global_scores_frame,
    build_predictions_frame,
    build_robustness_frame,
    build_scores_frame,
    build_timings_frame,
)
from parhaf_clinbench.reporting.analysis.rescoring import (  # noqa: E402
    build_fewshot_vs_zeroshot_deltas,
    build_vs_baseline_deltas,
    collect_official_doc_counts,
    load_gold_examples,
    paired_bootstrap_deltas,
    rescore_scenario_verbatim,
    rescore_suite,
)
from parhaf_clinbench.reporting.analysis.subgroups import (  # noqa: E402
    score_by_label,
    score_by_length,
    score_by_negation,
    score_by_speciality,
)

RESULTS_ROOT = REPO_ROOT / "results" / "run_090426" / "results"
ARTIFACTS_DIR = REPO_ROOT / "analysis" / "artifacts"
APP_DATA_DIR = REPO_ROOT / "ui" / "data"
HF_CACHE = REPO_ROOT / "data" / "hf_cache"

BOOTSTRAP_REPS = 1000
BASELINE_MODEL = "gliner2_multi"


def _ensure_dirs() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    APP_DATA_DIR.mkdir(parents=True, exist_ok=True)


def _write_parquet(df: pd.DataFrame, name: str) -> None:
    out = ARTIFACTS_DIR / f"{name}.parquet"
    df.to_parquet(out, index=False)
    shutil.copy(out, APP_DATA_DIR / out.name)
    print(f"  wrote {out.relative_to(REPO_ROOT)} — {len(df)} rows")


def _all_pairwise_deltas(doc_counts) -> pd.DataFrame:
    """Exhaustive model-vs-model paired bootstrap on every (track, task)."""

    models = sorted({m for m, _, _ in doc_counts})
    tracks = sorted({t for _, t, _ in doc_counts})
    tasks = sorted({t for _, _, t in doc_counts})
    comparisons = []
    for track in tracks:
        for task in tasks:
            available = [m for m in models if (m, track, task) in doc_counts]
            for a, b in combinations(available, 2):
                comparisons.append(
                    {
                        "label": "pairwise",
                        "model_a": a,
                        "track_a": track,
                        "task_a": task,
                        "model_b": b,
                        "track_b": track,
                        "task_b": task,
                    }
                )
    return paired_bootstrap_deltas(
        doc_counts, comparisons=comparisons, bootstrap_reps=BOOTSTRAP_REPS
    )


def main() -> None:
    _ensure_dirs()
    print("loading run suite…")
    suite = load_run_suite(RESULTS_ROOT)
    print(f"  {len(suite)} runs loaded: {sorted(suite)}")

    print("building core frames…")
    scores = build_scores_frame(suite)
    global_scores = build_global_scores_frame(suite)
    robustness = build_robustness_frame(suite)
    timings = build_timings_frame(suite)
    predictions = build_predictions_frame(suite)
    errors_frame = build_errors_frame(suite)
    taxonomy = build_error_taxonomy(suite)

    _write_parquet(scores, "scores")
    _write_parquet(global_scores, "global_scores")
    _write_parquet(robustness, "robustness")
    _write_parquet(timings, "timings")
    _write_parquet(predictions, "predictions_index")
    if not errors_frame.empty:
        slim = errors_frame.drop(
            columns=[c for c in ["raw_output"] if c in errors_frame.columns]
        )
        _write_parquet(slim, "errors")
    _write_parquet(taxonomy, "error_taxonomy")

    print("loading gold examples from offline cache…")
    gold = load_gold_examples(HF_CACHE)
    for t, g in gold.items():
        print(f"  {t}: {len(g)} examples")

    print(f"running scoring audit (bootstrap={BOOTSTRAP_REPS})…")
    audit = rescore_suite(suite, gold, bootstrap_reps=BOOTSTRAP_REPS)
    _write_parquet(audit, "scoring_audit")
    print("  status counts:", audit["status"].value_counts().to_dict())

    print("running spec-compliant scenario re-score…")
    scenario_verbatim = rescore_scenario_verbatim(
        suite, gold["scenario"], bootstrap_reps=BOOTSTRAP_REPS
    )
    _write_parquet(scenario_verbatim, "scenario_verbatim")

    print("computing subgroup scores (length / speciality / label / negation)…")
    sg_length = score_by_length(suite, gold)
    sg_spec = score_by_speciality(suite, gold)
    sg_label = score_by_label(suite, gold)
    sg_neg = score_by_negation(suite, gold)
    subgroups = pd.concat([sg_length, sg_spec, sg_label, sg_neg], ignore_index=True)
    _write_parquet(subgroups, "subgroups")

    print("collecting per-document official doc counts for paired bootstraps…")
    doc_counts = collect_official_doc_counts(suite, gold)
    print(f"  {len(doc_counts)} score cells collected")

    print("computing pairwise paired bootstrap deltas (all model pairs)…")
    pairwise = _all_pairwise_deltas(doc_counts)
    _write_parquet(pairwise, "paired_deltas")

    print("computing few-shot − zero-shot deltas per model/task…")
    fs_vs_zs = build_fewshot_vs_zeroshot_deltas(
        doc_counts, bootstrap_reps=BOOTSTRAP_REPS
    )
    _write_parquet(fs_vs_zs, "fewshot_vs_zeroshot")

    print(f"computing model − {BASELINE_MODEL} deltas on zero-shot…")
    vs_baseline = build_vs_baseline_deltas(
        doc_counts,
        baseline_model=BASELINE_MODEL,
        baseline_track="zero-shot",
        bootstrap_reps=BOOTSTRAP_REPS,
    )
    _write_parquet(vs_baseline, "vs_baseline")

    # Run-level metadata for the app's About page.
    metadata_rows = []
    for model, run in suite.items():
        md = run.run_metadata or {}
        metadata_rows.append(
            {
                "model": model,
                "run_id": md.get("run_id"),
                "suite_id": md.get("suite_id"),
                "model_hf_id": md.get("model_hf_id"),
                "model_revision": md.get("model_revision"),
                "runtime_name": md.get("runtime_name"),
                "runtime_version": md.get("runtime_version"),
                "gpu_name": md.get("gpu_name"),
                "started_at_utc": md.get("started_at_utc"),
                "finished_at_utc": md.get("finished_at_utc"),
                "elapsed_seconds": md.get("elapsed_seconds"),
                "run_dir": str(run.run_dir.relative_to(REPO_ROOT)),
            }
        )
    _write_parquet(pd.DataFrame(metadata_rows), "run_metadata")

    manifest = {
        "bootstrap_repetitions": BOOTSTRAP_REPS,
        "n_models": len(suite),
        "n_predictions": int(len(predictions)),
        "n_errors": int(len(errors_frame)),
        "n_gold_docs": {t: len(g) for t, g in gold.items()},
        "audit": audit["status"].value_counts().to_dict(),
    }
    (ARTIFACTS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    shutil.copy(ARTIFACTS_DIR / "manifest.json", APP_DATA_DIR / "manifest.json")
    print("done.")


if __name__ == "__main__":
    main()
