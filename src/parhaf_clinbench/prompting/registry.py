"""Prompt template registry."""

from __future__ import annotations

from pathlib import Path

from parhaf_clinbench.core.enums import TaskId, TrackId


def prompt_template_path(task: TaskId, track: TrackId) -> Path:
    """Return the template path for a `(task, track)` pair.

    Args:
        task: Task identifier.
        track: Track identifier.

    Returns:
        Path to the selected template file.
    """

    if track == TrackId.ZEROSHOT:
        filename = "zeroshot.jinja2"
    else:
        filename = "fewshot_fixed.jinja2"
    return Path("prompts") / task.value / filename
