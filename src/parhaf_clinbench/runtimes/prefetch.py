"""Idempotent Hugging Face model prefetch into persistent cache."""

from __future__ import annotations

import errno
import json
import re
from pathlib import Path

from pydantic import BaseModel, ConfigDict

_SAFE_CHARS_RE = re.compile(r"[^a-zA-Z0-9._-]+")


class PrefetchResult(BaseModel):
    """Result payload for a model prefetch operation."""

    model_config = ConfigDict(extra="forbid")

    hf_id: str
    revision: str
    local_path: str
    cache_hit: bool


def _safe_segment(value: str) -> str:
    """Return a filesystem-safe path segment."""

    return _SAFE_CHARS_RE.sub("-", value.strip("/"))


def resolve_local_model_path(cache_root: Path, hf_id: str, revision: str) -> Path:
    """Compute the local cache path for one model revision.

    Args:
        cache_root: Model cache root directory.
        hf_id: Hugging Face model repository id.
        revision: Model revision.

    Returns:
        Deterministic local cache path.
    """

    return cache_root / _safe_segment(hf_id) / _safe_segment(revision)


def _has_materialized_payload(target_dir: Path, marker_name: str) -> bool:
    """Check whether model files exist beyond the marker file."""

    for path in target_dir.rglob("*"):
        if path.is_file() and path.name != marker_name:
            return True
    return False


def _build_prefetch_error_message(
    *,
    hf_id: str,
    revision: str,
    target_dir: Path,
    phase: str,
    exc: Exception,
) -> str:
    """Build a detailed prefetch error message."""

    details = [
        f"Préchargement modèle impossible pendant `{phase}`.",
        f"hf_id={hf_id}",
        f"revision={revision}",
        f"target_dir={target_dir}",
        f"détail={exc}",
    ]
    if isinstance(exc, OSError) and exc.errno == errno.ENOSPC:
        details.append("Cause probable: espace disque insuffisant sur le volume.")
    else:
        details.append("Vérifie accès HF_TOKEN, existence du repo/révision et permissions disque.")
    return " ".join(details)


def prefetch_hf_model(
    *,
    hf_id: str,
    revision: str,
    cache_root: Path,
    hf_token: str | None = None,
) -> PrefetchResult:
    """Prefetch a Hugging Face model into `cache_root` when needed.

    Args:
        hf_id: Hugging Face model repository id.
        revision: Model revision.
        cache_root: Cache root directory.
        hf_token: Optional Hugging Face token.

    Returns:
        Model prefetch result containing local path and cache-hit flag.

    Examples:
        >>> result = prefetch_hf_model(
        ...     hf_id="Qwen/Qwen2.5-7B-Instruct",
        ...     revision="main",
        ...     cache_root=Path("/tmp/models"),
        ... )
        >>> isinstance(result.local_path, str)
        True
    """

    target_dir = resolve_local_model_path(cache_root=cache_root, hf_id=hf_id, revision=revision)
    marker = target_dir / ".prefetch.json"
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        raise RuntimeError(
            _build_prefetch_error_message(
                hf_id=hf_id,
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
            payload.get("hf_id") == hf_id
            and payload.get("revision") == revision
            and _has_materialized_payload(target_dir, marker.name)
        ):
            return PrefetchResult(
                hf_id=hf_id,
                revision=revision,
                local_path=str(target_dir),
                cache_hit=True,
            )

    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Le package `huggingface_hub` est requis pour précharger les modèles."
        ) from exc

    try:
        snapshot_download(
            repo_id=hf_id,
            revision=revision,
            token=hf_token,
            local_dir=str(target_dir),
        )
    except Exception as exc:
        raise RuntimeError(
            _build_prefetch_error_message(
                hf_id=hf_id,
                revision=revision,
                target_dir=target_dir,
                phase="snapshot_download",
                exc=exc,
            )
        ) from exc
    try:
        marker.write_text(
            json.dumps({"hf_id": hf_id, "revision": revision}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except Exception as exc:
        raise RuntimeError(
            _build_prefetch_error_message(
                hf_id=hf_id,
                revision=revision,
                target_dir=target_dir,
                phase="write_marker",
                exc=exc,
            )
        ) from exc
    return PrefetchResult(
        hf_id=hf_id,
        revision=revision,
        local_path=str(target_dir),
        cache_hit=False,
    )
