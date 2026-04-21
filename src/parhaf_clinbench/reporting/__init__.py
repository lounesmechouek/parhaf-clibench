"""PARHAF-CLINBENCH reporting layer.

The public surface is split in two halves:

1. The **runner side** (``export``, ``markdown``, ``plots``, ``tables``) — used
   at benchmark time to write ``metrics.json`` / ``metrics.csv`` / ``report.md``.
2. The **analysis side** (``loader``, ``analysis.*``, ``plots_extended``) —
   used after the fact by the notebook and the Streamlit app to rehydrate
   runs, re-score, build tidy frames and render figures.
"""

from parhaf_clinbench.reporting.loader import RunArtifacts, load_run, load_run_suite

__all__ = ["RunArtifacts", "load_run", "load_run_suite"]
