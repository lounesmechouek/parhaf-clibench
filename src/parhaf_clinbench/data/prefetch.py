"""Idempotent Hugging Face dataset prefetch into persistent cache."""

from __future__ import annotations

import errno
import json
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict

_SAFE_CHARS_RE = re.compile(r"[^a-zA-Z0-9._-]+")


class DatasetPrefetchResult(BaseModel):
    """Result payload for a dataset prefetch operation."""

    model_config = ConfigDict(extra="forbid")

    dataset_name: str
    revision: str
    local_path: str
    cache_hit: bool


def _safe_segment(value: str) -> str:
    """Return a filesystem-safe path segment."""

    return _SAFE_CHARS_RE.sub("-", value.strip("/"))


def resolve_local_dataset_path(cache_root: Path, dataset_name: str, revision: str) -> Path:
    """Compute the local cache path for one dataset revision.

    Args:
        cache_root: Dataset cache root directory.
        dataset_name: Hugging Face dataset identifier.
        revision: Dataset revision.

    Returns:
        Deterministic local cache path.
    """

    return cache_root / _safe_segment(dataset_name) / _safe_segment(revision)


def _has_materialized_payload(target_dir: Path, marker_name: str) -> bool:
    """Check whether dataset files exist beyond the marker file."""

    for path in target_dir.rglob("*"):
        if path.is_file() and path.name != marker_name:
            return True
    return False


def _build_prefetch_error_message(
    *,
    dataset_name: str,
    revision: str,
    target_dir: Path,
    phase: str,
    exc: Exception,
) -> str:
    """Build a detailed prefetch error message."""

    details = [
        f"Préchargement dataset impossible pendant `{phase}`.",
        f"dataset={dataset_name}",
        f"revision={revision}",
        f"target_dir={target_dir}",
        f"détail={exc}",
    ]
    if isinstance(exc, OSError) and exc.errno == errno.ENOSPC:
        details.append("Cause probable: espace disque insuffisant sur le volume.")
    else:
        details.append("Vérifie accès HF_TOKEN, existence du dataset/révision et permissions disque.")
    return " ".join(details)


def prefetch_hf_dataset(
    *,
    dataset_name: str,
    revision: str,
    cache_root: Path,
    hf_token: str | None = None,
    configs: list[str] | None = None,
) -> DatasetPrefetchResult:
    """Prefetch a Hugging Face dataset into `cache_root` when needed.

    The dataset cache is stored in a dedicated `(dataset, revision)` folder to
    guarantee deterministic reuse across process restarts.

    Args:
        dataset_name: Hugging Face dataset identifier.
        revision: Dataset revision.
        cache_root: Cache root directory.
        hf_token: Optional Hugging Face token.
        configs: Optional configuration list to prefetch.

    Returns:
        Dataset prefetch result containing local path and cache-hit flag.

    Examples:
        >>> result = prefetch_hf_dataset(
        ...     dataset_name="my-org/my-dataset",
        ...     revision="main",
        ...     cache_root=Path("/tmp/datasets"),
        ... )
        >>> isinstance(result.cache_hit, bool)
        True
    """

    target_dir = resolve_local_dataset_path(
        cache_root=cache_root,
        dataset_name=dataset_name,
        revision=revision,
    )
    marker = target_dir / ".prefetch_dataset.json"
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        raise RuntimeError(
            _build_prefetch_error_message(
                dataset_name=dataset_name,
                revision=revision,
                target_dir=target_dir,
                phase="mkdir",
                exc=exc,
            )
        ) from exc

    if marker.exists():
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        if (
            payload.get("dataset_name") == dataset_name
            and payload.get("revision") == revision
            and _has_materialized_payload(target_dir, marker.name)
        ):
            return DatasetPrefetchResult(
                dataset_name=dataset_name,
                revision=revision,
                local_path=str(target_dir),
                cache_hit=True,
            )

    try:
        from datasets import (  # type: ignore[import-untyped]
            get_dataset_config_names,
            load_dataset,
        )
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Le package `datasets` est requis pour précharger les datasets."
        ) from exc

    config_names = list(configs) if configs is not None else list(
        get_dataset_config_names(dataset_name, revision=revision)
    )
    try:
        if config_names:
            for config_name in config_names:
                load_dataset(
                    dataset_name,
                    config_name,
                    revision=revision,
                    cache_dir=str(target_dir),
                    token=hf_token,
                )
        else:
            load_dataset(
                dataset_name,
                revision=revision,
                cache_dir=str(target_dir),
                token=hf_token,
            )
    except Exception as exc:
        raise RuntimeError(
            _build_prefetch_error_message(
                dataset_name=dataset_name,
                revision=revision,
                target_dir=target_dir,
                phase="load_dataset",
                exc=exc,
            )
        ) from exc
    try:
        marker.write_text(
            json.dumps(
                {
                    "dataset_name": dataset_name,
                    "revision": revision,
                    "configs": config_names,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    except Exception as exc:
        raise RuntimeError(
            _build_prefetch_error_message(
                dataset_name=dataset_name,
                revision=revision,
                target_dir=target_dir,
                phase="write_marker",
                exc=exc,
            )
        ) from exc
    return DatasetPrefetchResult(
        dataset_name=dataset_name,
        revision=revision,
        local_path=str(target_dir),
        cache_hit=False,
    )
