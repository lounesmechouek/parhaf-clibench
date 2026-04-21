"""Streamlit page documenting the benchmark protocol in audit-ready form.

This page is the narrative companion to the score tables. It explains the
clinical tasks, the shared JSON contract, the official metrics, the bootstrap
procedure, and the independent rescoring audit that underpins published
numbers.
"""

from __future__ import annotations

import streamlit as st

from ui.data_loader import load_audit, load_manifest, load_run_metadata
from ui.theme import TASK_LABELS

# ---------------------------------------------------------------------------
# Static tables
# ---------------------------------------------------------------------------

DATASETS = [
    {
        "Task": "🔒 Pseudonymization",
        "Task id": "pseudo",
        "HuggingFace corpus": "HealthDataHub/PARHAF-pseudo-annotated",
        "Scope": "509 reports, ~7k identifying spans (names, dates, cities, addresses, phones, URLs, etc.)",
        "What is annotated": "Character-level spans with a category attribute (`span_type`).",
    },
    {
        "Task": "🦠 Infectiology",
        "Task id": "infectio",
        "HuggingFace corpus": "HealthDataHub/PARHAF-infectiology-annotated",
        "Scope": "134 reports, ~3.6k clinical mentions.",
        "What is annotated": "Spans with a clinical type (Bacteria / Bacteriemia / Infection / Site) and a polarity attribute (Absent / Indeterminate / Present).",
    },
    {
        "Task": "💊 Response to treatment",
        "Task id": "response",
        "HuggingFace corpus": "HealthDataHub/PARHAF-response_to_treatment-annotated",
        "Scope": "108 reports, one textual justification per document.",
        "What is annotated": "A single justification span per document plus a nomenclature label in {CompleteResponse, PartialResponse, StableDisease, ProgressiveDisease, NonApplicable, Undetermined}.",
    },
    {
        "Task": "🩺 Structured scenario",
        "Task id": "scenario",
        "HuggingFace corpus": "HealthDataHub/PARHAF",
        "Scope": "4,254 reports, ~13.6k structured fields across 20 medical specialities.",
        "What is annotated": "Key scenario fields (name, age, sex, admission/discharge mode, primary diagnosis, primary procedure, type of care) with their text, offsets and speciality.",
    },
]

TASK_SCORING = [
    {
        "Task": "Pseudonymization",
        "Official metric": "micro-F1 on `(start, end)` pairs",
        "Why this choice": "Masking the correct character range is enough to de-identify the document.",
        "Secondary metrics": "micro-F1 on normalized text · micro-F1 on `(start, end, label)` triples (+ recall and precision)",
    },
    {
        "Task": "Infectiology",
        "Official metric": "micro-F1 on `(norm_text, label, negation)` triples",
        "Why this choice": "A positive bacterium and a ruled-out bacterium have opposite clinical meaning.",
        "Secondary metrics": "micro-F1 on normalized text alone · micro-F1 on `(norm_text, label)` pairs",
    },
    {
        "Task": "Response to treatment",
        "Official metric": "micro-F1 on `(norm_text, label)` pairs",
        "Why this choice": "Captures both the justification span and the nomenclature class a downstream system will key on.",
        "Secondary metrics": "document-level label classification micro-F1",
    },
    {
        "Task": "Structured scenario",
        "Official metric": "micro-F1 on `(norm_text, label)` pairs",
        "Why this choice": "Measures usefulness as an EHR pre-filler: the field must be correct *and* come from the source.",
        "Secondary metrics": "document-level speciality classification micro-F1",
    },
]


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


def _section(title: str, anchor: str | None = None) -> None:
    """Render a section heading with an optional HTML anchor."""

    st.markdown(
        f"<div class='section-title' "
        f"{'id=' + repr(anchor) if anchor else ''}>{title}</div>",
        unsafe_allow_html=True,
    )


def render() -> None:
    """Render the full benchmark methodology and scoring audit."""

    st.title("Methodology")
    st.caption(
        "This page is the auditable companion of the numbers you see elsewhere "
        "in the app. It is written so a reviewer can reproduce every step."
    )

    manifest = load_manifest()
    audit = load_audit()
    runmeta = load_run_metadata()

    # ------------------------------------------------------------------
    # 1 Objective & research questions
    # ------------------------------------------------------------------
    _section("1 · Objective")
    st.markdown(
        """
        PARHAF-LM-CLINBENCH v1 measures, under a **strictly reproducible
        protocol**, how well general-purpose language models in the 7-9 B
        parameter range extract structured clinical information from
        French discharge reports, **without any supervised fine-tuning**.
        The evaluation covers four complementary tasks that together span
        the clinical IE continuum, from character-level span detection
        (pseudonymization) to document-level structuring (scenario).
        """
    )

    _section("Benchmark questions")
    st.markdown(
        """
        1. **Extraction capability.** *To what extent, measured by
           micro-F1 and its 95% bootstrap CI, can 7-9 B LLMs extract
           French clinical information in zero-shot and few-shot-fixed
           regimes?*
        2. **LLMs vs. encoder baseline.** *How does the best LLM
           configuration compare to <code>GLiNER2</code> on each task,
           and are the differences statistically significant under a
           paired document-level bootstrap?*
        3. **Few-shot lift.** *Does a fixed demonstration bank
           improve performance, by how much, and is the improvement
           consistent across tasks?*
        4. **Operational robustness.** *To what extent do SLMs guarantee
           JSON validity and schema conformity?*
        5. **Efficiency frontier.** *Who sits on the Pareto front of
           F1 versus latency?*
        6. **Failure modes.** *Subgroup breakdown (length,
           speciality, label, polarity), where do models systematically
           fail?*
        7. **Error taxonomy.** *What share of failures is schema
           errors versus genuine extraction errors?*
        """,
        unsafe_allow_html=True,
    )

    # ------------------------------------------------------------------
    # 2  Datasets
    # ------------------------------------------------------------------
    _section("2 · Datasets")
    st.markdown(
        """
        The benchmark uses four public French clinical corpora released
        under the **PARHAF** umbrella by the Health Data Hub on
        HuggingFace.

        *PARHAF is an open French corpus of human-authored clinical reports of fictional patients. It was created to support the development and evaluation of clinical NLP systems under strict health-data protection constraints. Each patient is each documented with structured clinical information (diagnosis, procedures, care pathway, discharge data).*
        """
    )
    st.dataframe(DATASETS, use_container_width=True, hide_index=True)
    if manifest.get("n_gold_docs"):
        footprint = [
            {"Task": TASK_LABELS.get(task, task), "Documents": n}
            for task, n in manifest["n_gold_docs"].items()
        ]
        st.markdown("**Actual gold footprint used by this run:**")
        st.dataframe(footprint, use_container_width=True, hide_index=True)

    st.caption(
        "For each task both splits (`train` and `dev`) are combined and "
        "used jointly for evaluation. There is no held-out set because "
        "no supervised training happens in this benchmark."
    )

    # ------------------------------------------------------------------
    # 3  Unified task formulation
    # ------------------------------------------------------------------
    _section("3 · Unified task formulation")
    st.markdown(
        """
        Every task is framed as a transformation of a discharge report
        into a canonical JSON envelope. Using a single schema means one
        parser, one validator and one scorer, so the only way for a
        model to look different in the results is to actually produce
        different extractions.
        """
    )
    st.code(
        """{
  "document_id": "CARDIOLOGIE-00054_CRH",
  "task": "pseudo | infectio | response | scenario",
  "speciality": "CARDIOLOGIE" | null,
  "records": [
    {
      "label": "FIRST_NAME",
      "text": "Arun",
      "start": 83,
      "end": 87,
      "attributes": {"role": "PATIENT"}
    }
  ]
}""",
        language="json",
    )
    st.markdown(
        """
        Which record fields are required depends on the task:

        - **pseudo**: `label`, `text`, `start`, `end` (offsets are
          evaluated on the raw, *non-normalised* text). Optional
          `attributes.role` for person names.
        - **infectio**: `label`, `text`, `attributes.negation` in
          `{Absent, Indeterminate, Present}`.
        - **response**: `label` in the six-class nomenclature, `text`
          for the justification span.
        - **scenario**: `speciality` at document level, then `label` +
          `text` for each extracted field (age, sex, admission mode…).

        A Pydantic schema validates every generation *before* scoring.
        Invalid records are rejected (and counted in the error taxonomy),
        they never contribute to the score.
        """
    )

    # ------------------------------------------------------------------
    # 4  Prompting regimes
    # ------------------------------------------------------------------
    _section("4 · Prompting regimes")
    st.markdown(
        """
        The benchmark defines two prompting regimes:

        - **Zero-shot.** The model receives the task instruction, the
          canonical schema, and the source text.
        - **Few-shot fixed.** Same prompt plus a small, **frozen** bank
          of synthetically generated demonstrations. The bank is built **outside**
          the evaluation corpus so every public report remains a valid
          test document, and it is identical for every model on a given
          task.

        All prompts are **versioned by hash**, any change to the
        instruction text, demonstration bank, JSON schema, decoding
        parameters or parsing rules creates a new benchmark version.
        Every run artefact stores the prompt hash so a reader can verify
        the exact instruction that produced a given number.

        Decoding parameters are held constant across models and tracks
        (temperature 0, top-p 1, seed 42, max 512 output tokens) so the
        paired bootstrap between any two models is a clean measurement
        of model differences rather than decoding noise.
        """
    )

    # ------------------------------------------------------------------
    # 5  Normalization
    # ------------------------------------------------------------------
    _section("5 · Text normalization before scoring")
    st.markdown(
        """
        To avoid penalising superficial string differences, every text
        compared in a metric goes through the following pipeline:

        1. **NFC Unicode normalization**, canonical composition of
           accented characters.
        2. **Strip** leading/trailing whitespace.
        3. **Collapse** internal whitespace to a single space.
        4. **Lowercase** (for case-insensitive comparison).
        5. **Strip trailing punctuation** `[. , ; : ! ?]`.

        **Character offsets are never normalised**, they are compared
        byte-for-byte against the raw source text, because the whole
        point of the pseudonymization task is to mask the exact
        characters an annotator flagged as sensitive.

        Enumerated fields (`label`, `speciality`, `attributes.negation`)
        are compared by **exact match**. The canonical schema enforces
        the allowed value set via Pydantic, so a model that hallucinates
        a new label produces a schema-invalid record that lands in the
        error taxonomy rather than silently scoring zero.
        """
    )

    # ------------------------------------------------------------------
    # 6  Scoring
    # ------------------------------------------------------------------
    _section("6 · Scoring")
    st.markdown(
        """
        Every task produces a per-document multiset of *elementary
        units*. The shape of these units is the only thing that differs
        between tasks:

        | Task | Elementary unit (official) |
        |---|---|
        | Pseudonymization | `(start, end)` character-offset pairs |
        | Infectiology | `(normalized_text, label, negation)` triples |
        | Response to treatment | `(normalized_text, label)` pairs |
        | Structured scenario | `(normalized_text, label)` pairs |

        From those multisets we compute, per document, the number of
        true positives (`TP`), false positives (`FP`) and false
        negatives (`FN`) using a multiset intersection. Counts are then
        **micro-aggregated** over the whole corpus:
        """
    )
    st.latex(
        r"\text{Precision} = \frac{TP}{TP + FP}, \quad"
        r"\text{Recall} = \frac{TP}{TP + FN},   "
        r"F_1 = \frac{2\,PR}{P+R}"
    )
    st.markdown("**Per-task details:**")
    st.dataframe(TASK_SCORING, use_container_width=True, hide_index=True)

    st.info(
        "**Why micro-F1 and not macro?** Micro-F1 weights every record "
        "equally, which is the right objective if you care about how "
        "many correct extractions the system produces on the whole "
        "corpus. Macro-F1 would over-weight rare categories and mask "
        "failures on the frequent ones."
    )

    # ------------------------------------------------------------------
    # 7 Aggregation
    # ------------------------------------------------------------------
    _section("7 · Cross-task aggregation")
    st.markdown(
        """
        We propose a per-track **global score** as the *equal-weight arithmetic
        mean* of the four task F1s. Equal weights is the deliberate
        choice: the scenario corpus has ~30x more documents than the
        response corpus, so a document-weighted average would reduce
        the benchmark to the scenario task alone.

        A cross-track aggregation (zero-shot ↔ few-shot) is reported
        only as a secondary indicator, the two tracks are treated as
        separate leaderboards so a reader can see which regime was
        responsible for a given ranking.

        We acknowledge that micro-F1 scores are computed using task-specific criteria.
        As a result, aggregated scores should be interpreted as high-level trend indicators for comparing models overall.
        For more granular insights, we also provide task-level and subgroup analyses.
        """
    )

    # ------------------------------------------------------------------
    # 8  Uncertainty quantification
    # ------------------------------------------------------------------
    _section("8 · Document-level bootstrap for uncertainty")
    bootstrap_repetitions = manifest.get("bootstrap_repetitions", 1000)
    st.markdown(
        f"""
        Point estimates are only part of the story. Every official
        metric is accompanied by a **non-parametric bootstrap
        confidence interval**, computed at the **document level**,
        that is, we resample whole documents with replacement, recompute
        `TP/FP/FN` on the resampled corpus, and recompute the metric.

        - **Replications:** `B = {bootstrap_repetitions}`.
        - **Seed:** `42` (deterministic across runs of this notebook).
        - **CI type:** percentile 95%, lower bound at the 2.5th
          percentile, upper bound at the 97.5th percentile of the
          bootstrap distribution.
        - **Paired comparisons:** when comparing two models on the
          same task we share the document indices between both models
          so they see the same resampled corpus on every replication.
          This removes a large chunk of variance and makes the
          confidence interval on the delta much tighter than an
          unpaired bootstrap would.

        **Why document-level and not record-level?** Records inside the
        same document are not independent, a single noisy discharge
        report can contain dozens of mentions. Resampling records would
        over-estimate the effective sample size and give artificially
        narrow confidence intervals. Document-level resampling
        preserves the natural clustering of the data and reports
        honest uncertainty.

        Operational metrics (schema conformity, latency, throughput)
        are reported as observed values without bootstrap, they are
        already population-level rates, not sample estimates.
        """
    )

    # ------------------------------------------------------------------
    # 9 Scoring audit
    # ------------------------------------------------------------------
    _section("9 · Scoring audit")
    st.markdown(
        """
        Before a single number is shown in the app, the shipped
        `metrics.json` is independently audited. The audit pipeline:

        1. Rebuilds canonical predictions by reloading every
           `predictions.jsonl` row and re-parsing it through the
           canonical schema.
        2. Reloads the gold examples from the offline Arrow cache
        3. Re-invokes the shipped scoring modules on the reloaded gold and
           predictions.
        4. Compares the re-scored F1 to the value persisted in
           `metrics.json` at a tolerance of `1e-6`.

        Any cell with a mismatch is surfaced here as a flagged row.
        """
    )
    if not audit.empty:
        ok = int((audit.status == "ok").sum())
        total = len(audit)
        col1, col2 = st.columns([1, 3])
        with col1:
            st.metric("Cells matching", f"{ok} / {total}")
        with col2:
            st.progress(ok / total if total else 0.0)
        mismatches = audit[audit.status != "ok"]
        if mismatches.empty:
            st.success(
                "All cells reproduce the shipped metrics to six decimal "
                "places. The scoring pipeline is byte-faithful to the "
                "specification."
            )
        else:
            st.warning(
                f"{len(mismatches)} cells disagree with the shipped "
                "metrics. See the table below."
            )
            st.dataframe(mismatches, use_container_width=True, hide_index=True)
        with st.expander("Show the full audit table", expanded=False):
            st.dataframe(
                audit[
                    [
                        "model", "track", "task", "shipped_f1",
                        "rescored_f1", "rescored_precision",
                        "rescored_recall", "ci_low", "ci_high",
                        "n_docs", "status",
                    ]
                ].round(6),
                use_container_width=True,
                hide_index=True,
            )

    st.markdown(
        """
        > **Note on the scenario verbatim filter.** We only materialise
        > records whose text can be character-aligned in the source
        > (via `_align_span`). We verified on the full corpus that
        > **every gold scenario record is already verbatim**, so the
        > spec-compliant re-scored F1 equals the shipped F1 across the
        > 13 `(model, track)` scenario cells.
        """
    )

    # ------------------------------------------------------------------
    # 10 Models
    # ------------------------------------------------------------------
    _section("10 · Systems evaluated")
    st.markdown(
        """
        Seven systems were evaluated in this run. Six are
        general-purpose 7-9 B parameter instruction-tuned LLMs served
        through a shared **vLLM** backend with structured-outputs JSON
        schema enforcement. The seventh is the **GLiNER2** encoder
        baseline, non-autoregressive, specialised for information
        extraction, with the same output contract applied
        **post-inference**.
        """
    )
    if not runmeta.empty:
        show = runmeta[
            ["model", "model_hf_id", "runtime_name", "runtime_version", "gpu_name", "elapsed_seconds"]
        ].copy()
        show.columns = [
            "Short id", "HuggingFace id", "Runtime", "Runtime version",
            "GPU", "Elapsed (s)",
        ]
        st.dataframe(show, use_container_width=True, hide_index=True)
    st.caption(
        "GLiNER2 is the only system without a few-shot track, its "
        "entity descriptors are fixed at inference time, so a "
        "demonstration bank would not apply."
    )

    # ------------------------------------------------------------------
    # 11 Limitations
    # ------------------------------------------------------------------
    _section("11 · Limitations")
    st.markdown(
        """
        - **Fixed few-shot bank.** We cannot tell apart "model
          benefits from examples" from "model benefits from *this*
          particular example bank". A sensitivity study over
          alternative banks could be future work.
        - **French only, one institutional source.** All corpora come
          from the Health Data Hub and are in French. Generalisation
          to other languages or other hospitals is not measured here.
        - **No fine-tuning.** We only measure zero-shot and few-shot
          fixed performance, not the potential of these models after supervised fine-tuning on the task.
          This could also be part of future work.
        - **No advanced prompting, reasoning or tool use.** We only evaluate a basic prompting format with a single instruction and a fixed demonstration bank.
            That's because this version is focused on the core extraction capability of the models.
            Future works could explore chain-of-thought, self-consistency, retrieval, tooling, etc.
        """
    )

    # ------------------------------------------------------------------
    # 12 Reproducibility
    # ------------------------------------------------------------------
    _section("12 · Reproducibility")
    st.markdown(
        """
        Instructions for running this benchmark and reproduce the results can be found at : https://github.com/lounesmechouek/parhaf-clibench

        The package has been thought to be expandable and reusable. Feel free to test different model architectures and sizes for reference.


        """
    )
    if manifest:
        with st.expander("Manifest", expanded=False):
            st.json(manifest)

    # ------------------------------------------------------------------
    # 13 Citation
    # ------------------------------------------------------------------
    _section("13 · Citation")
    st.markdown(
        """
        If this work was useful to you, please consider citing it.
        Two entries are provided: one for the **benchmark codebase** and
        one for the **analyses and results** published through this report.
        """
    )

    st.markdown("**Benchmark codebase**")
    st.code(
        """\
@software{parhaf_clibench,
  author       = {Mechouek, Lounes},
  title        = {parhaf-clibench: A reproducible benchmark for clinical information extraction with language models on PARHAF},
  year         = {2026},
  url          = {https://github.com/lounesmechouek/parhaf-clibench},
  version      = {0.1.0}
}""",
        language="bibtex",
    )

    st.markdown("**Analyses and results**")
    st.code(
        """\
@techreport{mechouek2026Dparhaf,
  author       = {Mechouek, Lounes},
  title        = {Benchmarking Small Language Models on French Clinical
                  Information Extraction: {PARHAF-LM-CLINBENCH}},
  year         = {2026},
  type         = {Technical Report},
  url          = {https://github.com/lounesmechouek/parhaf-clibench},
  note         = {Covers 7 systems (6 LLMs + GLiNER2), 4 tasks,
                  zero-shot and few-shot fixed tracks, 65k predictions.}
}""",
        language="bibtex",
    )

    st.caption(
        "Please also cite the original PARHAF corpora (HealthDataHub) "
        "if you use the datasets directly."
    )
