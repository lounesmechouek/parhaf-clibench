from __future__ import annotations

from parhaf_clinbench.scoring.bootstrap import bootstrap_global_score
from parhaf_clinbench.scoring.common import DocCounts


def test_bootstrap_global_score_full_matches_mean_task_scores() -> None:
    per_task = {
        "pseudo": [DocCounts(tp=2, fp=0, fn=0)],
        "infectio": [DocCounts(tp=1, fp=1, fn=0)],
    }
    boot = bootstrap_global_score(per_task_doc_counts=per_task, repetitions=100, seed=7)
    assert 0.0 <= boot.ci_low <= boot.ci_high <= 1.0
    # pseudo=1.0, infectio=2/3 -> mean=0.8333...
    assert abs(boot.score_full - ((1.0 + (2.0 / 3.0)) / 2.0)) < 1e-6
