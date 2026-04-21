# Robustness metrics

A benchmark that reports only accuracy tells half the story. A model
that returns correct extractions 40% of the time and invalid outputs
60% of the time is a broken pipeline, not a 0.40 F1 system. The
robustness pane captures what the accuracy number hides.

Source: [`parhaf_clinbench.reporting.analysis`](../reference/parhaf_clinbench/reporting/analysis/index.md).

## What the package measures

| Metric              | Definition                                                                 |
|---------------------|----------------------------------------------------------------------------|
| Schema conformity   | Share of documents for which the JSON response parses against the schema. |
| Empty output rate   | Share of documents for which the model returned zero records.             |
| JSON validity rate  | Share of documents for which the raw response is valid JSON before schema checks. |
| Median latency      | Per-document latency at the median.                                        |
| p95 latency         | Per-document latency at the 95th percentile.                               |
| Throughput          | Documents per second across the campaign.                                  |

Every metric is computed per `(model, track, task)` cell and
aggregated into the suite-level `robustness.parquet` table consumed
by the Streamlit UI.

## The error taxonomy

On top of raw counts, the package classifies every document-level
error into a small set of categories:

- **Invalid JSON**: the response does not parse as JSON.
- **Schema violation**: JSON parses but fails Pydantic validation.
- **Offset drift**: records look valid but character offsets are
  wrong.
- **Empty output**: no records returned.
- **Content error**: records parse and align but do not match gold.

The taxonomy lives in
[`parhaf_clinbench.reporting.analysis.error_taxonomy`](../reference/parhaf_clinbench/reporting/analysis/error_taxonomy.md)
and drives the error-explorer page of the UI. It is often the first
thing to look at when a model scores unexpectedly low: a collapse in
schema conformity is very different from a genuine drop in
extraction quality.

## Why this matters

The two symptoms look identical at the F1 level but call for very
different fixes:

- Low schema conformity with otherwise correct content: invest in
  guided decoding, better prompts, or a model with stronger output
  discipline.
- High schema conformity with low content F1: invest in fine-tuning,
  better few-shot examples, or a different model family.

Reading the robustness numbers alongside F1 is the shortest path
between a benchmark result and an actionable next step.
