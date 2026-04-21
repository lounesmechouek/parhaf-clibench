"""Offline reconstruction of gold examples from the local HF cache.

This module is intentionally network-free: it reads Arrow files materialized in
``data/hf_cache`` and reuses the canonical task-specific converters already
implemented in ``parhaf_clinbench.data.hf_loaders``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from parhaf_clinbench.core.enums import TaskId
from parhaf_clinbench.core.models import DocumentExample
from parhaf_clinbench.data.hf_loaders import (
    _load_hf_infectio,
    _load_hf_pseudo,
    _load_hf_response,
    _load_hf_scenario,
)


@dataclass(frozen=True)
class CacheDatasetSpec:
    """Local cache location metadata for one benchmark task."""

    task: TaskId
    cache_dataset: str
    revision: str
    configs: tuple[str, ...]
    arrow_prefix: str


TASK_DATASETS: dict[str, CacheDatasetSpec] = {
    "pseudo": CacheDatasetSpec(
        task=TaskId.PSEUDO,
        cache_dataset="HealthDataHub___parhaf-pseudo-annotated",
        revision="4d866f075a4d91c4e5ce0058feedd6da1d8e879a",
        configs=("document_metadata", "spans"),
        arrow_prefix="parhaf-pseudo-annotated",
    ),
    "infectio": CacheDatasetSpec(
        task=TaskId.INFECTIO,
        cache_dataset="HealthDataHub___parhaf-infectiology-annotated",
        revision="452cfeeec0987b8088a6b9294f00dd4f735a5980",
        configs=("document_metadata", "spans"),
        arrow_prefix="parhaf-infectiology-annotated",
    ),
    "response": CacheDatasetSpec(
        task=TaskId.RESPONSE,
        cache_dataset="HealthDataHub___parhaf-response_to_treatment-annotated",
        revision="a100ab2742614821898bc6880c0bff0c9a5bdccd",
        configs=("document_metadata", "spans"),
        arrow_prefix="parhaf-response_to_treatment-annotated",
    ),
    "scenario": CacheDatasetSpec(
        task=TaskId.SCENARIO,
        cache_dataset="HealthDataHub___parhaf",
        revision="0a88093ff16168494fdb17aa40a040107171ed9d",
        configs=("default",),
        arrow_prefix="parhaf",
    ),
}

_SPLIT_RANK = {"train": 0, "dev": 1, "validation": 2, "test": 3}


def _split_from_arrow_name(path: Path) -> str:
    stem = path.stem
    if "-" not in stem:
        return "train"
    return stem.rsplit("-", 1)[-1]


def _split_sort_key(path: Path) -> tuple[int, str]:
    split = _split_from_arrow_name(path)
    return (_SPLIT_RANK.get(split, 99), path.name)


def _read_arrow_rows(path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow as pa
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "pyarrow is required for offline gold reconstruction. "
            "Install it in the analysis environment."
        ) from exc

    # Hugging Face ``datasets`` writes Arrow files using the IPC *stream*
    # format, not the *file* format — so ``open_stream`` is the correct reader.
    with pa.memory_map(str(path), "r") as source:
        reader = pa.ipc.open_stream(source)
        table = reader.read_all()
    rows = table.to_pylist()
    return [dict(row) for row in rows]


def _resolve_arrow_files(
    *,
    cache_dir: Path,
    spec: CacheDatasetSpec,
    config: str,
) -> list[Path]:
    root = cache_dir / spec.cache_dataset / config / "0.0.0" / spec.revision
    if not root.exists():
        raise FileNotFoundError(
            f"Missing local cache directory for task={spec.task.value} at {root}"
        )
    files = sorted(root.glob(f"{spec.arrow_prefix}-*.arrow"), key=_split_sort_key)
    if not files:
        files = sorted(root.glob("*.arrow"), key=_split_sort_key)
    if not files:
        raise FileNotFoundError(
            f"No Arrow files found for task={spec.task.value} under {root}"
        )
    return files


def load_task_examples_offline(task: TaskId, cache_dir: Path) -> list[DocumentExample]:
    """Load one task gold set strictly from local Arrow cache."""

    spec = TASK_DATASETS[task.value]
    rows: list[tuple[str, dict[str, Any]]] = []
    for config in spec.configs:
        files = _resolve_arrow_files(cache_dir=cache_dir, spec=spec, config=config)
        for path in files:
            split = _split_from_arrow_name(path)
            source = split if task == TaskId.SCENARIO else f"{config}:{split}"
            for row in _read_arrow_rows(path):
                rows.append((source, row))

    if task == TaskId.PSEUDO:
        return _load_hf_pseudo(rows)
    if task == TaskId.INFECTIO:
        return _load_hf_infectio(rows)
    if task == TaskId.RESPONSE:
        return _load_hf_response(rows)
    if task == TaskId.SCENARIO:
        return _load_hf_scenario(rows)
    raise ValueError(f"Unsupported task: {task}")


def load_gold_examples_offline(cache_dir: Path) -> dict[str, list[DocumentExample]]:
    """Load every benchmark task gold set from local cache."""

    out: dict[str, list[DocumentExample]] = {}
    for task_name in ("pseudo", "infectio", "response", "scenario"):
        out[task_name] = load_task_examples_offline(TaskId(task_name), cache_dir)
    return out


def dataset_spec_table() -> list[dict[str, str]]:
    """Expose task-to-dataset mapping for notebook methodology sections."""

    rows: list[dict[str, str]] = []
    for task_name in ("pseudo", "infectio", "response", "scenario"):
        spec = TASK_DATASETS[task_name]
        rows.append(
            {
                "task": task_name,
                "cache_dataset": spec.cache_dataset,
                "revision": spec.revision,
                "configs": ",".join(spec.configs),
            }
        )
    return rows
