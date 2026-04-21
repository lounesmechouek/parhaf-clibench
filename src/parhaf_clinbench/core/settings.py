"""Centralized environment-based application settings."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _resolve_env_files() -> tuple[str, ...]:
    """Return candidate `.env` file paths in deterministic precedence order."""

    candidates: list[Path] = []
    override = os.environ.get("PARHAF_ENV_FILE")
    if override:
        candidates.append(Path(override).expanduser())

    cwd = Path.cwd()
    candidates.extend(
        [
            cwd / ".env",
            cwd / "infra/.env",
            _PROJECT_ROOT / ".env",
            _PROJECT_ROOT / "infra/.env",
        ]
    )

    seen: set[str] = set()
    ordered: list[str] = []
    for candidate in candidates:
        resolved = str(candidate.resolve())
        if resolved in seen:
            continue
        seen.add(resolved)
        ordered.append(resolved)
    return tuple(ordered)


class AppSettings(BaseSettings):
    """Runtime settings shared by CLI commands, runtimes, and ops tooling."""

    model_config = SettingsConfigDict(
        env_file=_resolve_env_files(),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # NOTE: Runtime mode and output defaults.
    parhaf_suite: str = "configs/suites/v1_full.yaml"
    parhaf_output_dir: str = "/workspace/results/default"

    # NOTE: Hugging Face credentials and cache locations.
    hf_token: str | None = None
    hf_home: Path = Path("/workspace/.cache/huggingface")
    huggingface_hub_cache: Path = Path("/workspace/.cache/huggingface/hub")
    transformers_cache: Path = Path("/workspace/.cache/huggingface/transformers")
    model_cache_root: Path = Path("/workspace/models")
    dataset_cache_root: Path = Path("/workspace/datasets")

    # NOTE: vLLM endpoint defaults.
    vllm_host: str = "127.0.0.1"
    vllm_port: int = 8000

    # NOTE: Export and archive paths.
    results_dir: Path = Path("/workspace/results")
    export_dir: Path = Path("/workspace/exports")
    final_archive_path: Path = Path("/workspace/results.tar.zst")

    # NOTE: RunPod API configuration.
    runpod_api_key: str | None = None
    runpod_api_base: str = "https://rest.runpod.io/v1"
    runpod_template_id: str | None = None
    runpod_pod_id: str | None = None

    # NOTE: Optional metadata propagated to artifacts.
    parhaf_image_digest: str | None = None
    runpod_gpu_name: str | None = None
    runpod_gpu_count: int | None = None
    runpod_vram_gb: int | None = None
    runpod_container_disk_gb: int | None = None
    runpod_volume_gb: int | None = None
    parhaf_export_mode: str = "local"
    parhaf_export_destination: str | None = None
    runpod_template_name: str | None = None

    def apply_hf_cache_env(self) -> None:
        """Export Hugging Face cache environment variables for subprocesses.

        Examples:
            >>> settings = AppSettings()
            >>> settings.apply_hf_cache_env()
        """

        os.environ.setdefault("HF_HOME", str(self.hf_home))
        os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(self.huggingface_hub_cache))
        os.environ.setdefault("TRANSFORMERS_CACHE", str(self.transformers_cache))
        if self.hf_token:
            os.environ.setdefault("HF_TOKEN", self.hf_token)


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return cached application settings loaded once per process.

    Returns:
        Cached `AppSettings` instance.

    Examples:
        >>> get_settings() is get_settings()
        True
    """

    return AppSettings()
