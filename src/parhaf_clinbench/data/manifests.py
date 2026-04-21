"""Experiment manifest construction helpers."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from parhaf_clinbench.core.enums import TaskId, TrackId


class ManifestItem(BaseModel):
    """Single experiment unit `(task, track, model)`."""

    model_config = ConfigDict(extra="forbid")

    task: TaskId
    track: TrackId
    model: str



def build_manifest(tasks: list[TaskId], tracks: list[TrackId], models: list[str]) -> list[ManifestItem]:
    """Build the Cartesian product of tasks, tracks, and models.

    Args:
        tasks: Selected tasks.
        tracks: Selected tracks.
        models: Selected model identifiers.

    Returns:
        Flattened list of manifest entries.

    Examples:
        >>> items = build_manifest([TaskId.PSEUDO], [TrackId.ZEROSHOT], ["m1"])
        >>> len(items)
        1
    """

    items: list[ManifestItem] = []
    for task in tasks:
        for track in tracks:
            for model in models:
                items.append(ManifestItem(task=task, track=track, model=model))
    return items
