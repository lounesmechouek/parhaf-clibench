from __future__ import annotations

from parhaf_clinbench.core.enums import TaskId, TrackId
from parhaf_clinbench.data.manifests import build_manifest


def test_build_manifest_cartesian_product() -> None:
    manifest = build_manifest(
        tasks=[TaskId.PSEUDO, TaskId.RESPONSE],
        tracks=[TrackId.ZEROSHOT],
        models=["m1", "m2"],
    )
    assert len(manifest) == 4
    assert manifest[0].task == TaskId.PSEUDO
    assert manifest[0].track == TrackId.ZEROSHOT
    assert manifest[0].model == "m1"
    assert {item.model for item in manifest} == {"m1", "m2"}
