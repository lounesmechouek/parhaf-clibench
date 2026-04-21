# Streamlit UI

The Streamlit UI lives under [`ui/`](https://github.com/lounesmechouek/parhaf-clinbench/tree/main/ui)
and loads directly from a run directory. It is designed as an
artifact viewer: nothing in the UI runs inference or mutates a run.

![Streamlit UI](../assets/streamlit.png)

## Launching

Install the `ui` extra if you have not already:

```bash
uv sync --extra ui
```

Then launch the app:

```bash
uv run --extra ui streamlit run ui/app.py
```

Point the app at the run directory from the sidebar. The data loader
under `ui/data_loader.py` reads the parquet tables described in
[Reading the results](reading-results.md) through Streamlit's
`@st.cache_data` decorator, so navigation between pages stays
responsive even for large runs.

## Pages

| Page               | Shows                                                                       |
|--------------------|-----------------------------------------------------------------------------|
| Overview           | A one-screen summary of the run and its headline numbers.                   |
| Leaderboard        | Global F1 per model with bootstrap bars, per track.                         |
| Task deep dive     | Per-task leaderboards, forest plots, subgroup slices.                        |
| Model card         | All scores, robustness, and latency for a single model.                     |
| Head-to-head       | Paired delta with a baseline model and significance from the paired bootstrap.|
| Robustness         | Schema conformity, empty rate, validity, latency per cell.                  |
| Subgroups          | F1 by document-length quartile, specialty, negation polarity.               |
| Error explorer     | Document-level drill-down into the error taxonomy.                          |
| Methodology        | Embedded documentation of the metrics and the run contract.                 |

## Reading the Overview

The Overview page answers the first three questions a reader has:

- What was evaluated (suite, models, tracks, dataset revisions).
- What is the headline global F1 per model and how wide are the
  confidence intervals.
- How much of that F1 is due to output-format failures rather than
  to the model itself.

From there, the Task deep dive and the Error explorer are the two
pages that generate the most insight per minute.

## Extending the UI

The UI is intentionally thin. Every page is a small Streamlit view
that calls one loader function and one or two Plotly figure builders
from [`parhaf_clinbench.reporting`](../reference/parhaf_clinbench/reporting/index.md).
Adding a page is a matter of copying an existing one and pointing it
at a different parquet file.

## Shipping the UI

The UI is not published as a hosted service today. Anyone with read
access to a run directory can launch it locally. If you need a
shared dashboard for a team, the simplest path is to run it inside
the Docker image (see [Docker](../llmops/docker.md)) behind a
reverse proxy.
