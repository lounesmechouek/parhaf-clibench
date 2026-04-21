from __future__ import annotations

from pathlib import Path

from parhaf_clinbench.orchestration.runner import run_campaign


def test_run_campaign_smoke(tmp_path: Path) -> None:
    runs = run_campaign(
        suite_path=Path("configs/suites/v1_smoke.yaml"),
        task_selection="all",
        track_selection="all",
        model_selection="all",
        output_dir=tmp_path,
    )
    assert runs
    run_dir = runs[0]
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "report.md").exists()
    assert (run_dir / "run_metadata.json").exists()
