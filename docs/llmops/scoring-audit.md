# Scoring audit

The scoring audit is the procedure that checks a run's reported
numbers against the raw predictions. It is the single best defense
against silent regressions in the scoring code, against corrupt
artifacts, and against editorial mistakes in a report.

## How it works

The audit is a pure recomputation:

1. Reload every `predictions.jsonl` under the run directory.
2. Reparse each record through the canonical Pydantic schema.
3. Reload the gold records for each task from the local dataset
   cache.
4. Re-invoke the scorer for each cell.
5. Compare the recomputed F1 and bootstrap interval to the values
   stored in `scores.json`.

The comparison tolerance is six decimal places. A cell is marked
matching when every shipped number agrees with the recomputed
number within that tolerance.

## Running the audit

### For one cell

```bash
uv run parhaf-clinbench score \
  --predictions results/<run_id>/<model>/<task>/<track>/predictions.jsonl \
  --gold results/<run_id>/gold/<task>.jsonl \
  --task pseudo
```

Compare the printed JSON to the `scores.json` of the cell. If both
agree to six decimals, the cell passes.

### For a full run

The reporting layer exposes a batch entry point that walks the run
and reports matching cells versus total cells. It lives in
[`parhaf_clinbench.reporting.analysis.rescoring`](../reference/parhaf_clinbench/reporting/analysis/rescoring.md).

## When to run it

- Before publishing a report or a blog post built from a run.
- After any change to the scoring code.
- After any change to the parser, the normalizer, or the aligner.
- When migrating a run directory between machines.

## What it does not catch

The audit recomputes scores from the predictions. It does not
re-infer. A corrupt predictions file that agrees with a corrupt
scoring pass will still reproduce. That is fine: the audit's job is
to catch scoring and reporting bugs, not inference bugs. Inference
reproducibility is the job of [Determinism](determinism.md) and
[Versioning](versioning.md).
