"""Load and resolve benchmark experiment plans."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field

from parhaf_clinbench.core.enums import RuntimeName, TaskId, TrackId


class ConfigModel(BaseModel):
    """Strict Pydantic base model for configuration files."""

    model_config = ConfigDict(extra="forbid")


class SuiteConfig(ConfigModel):
    """Benchmark suite configuration."""

    suite_id: str
    benchmark_version: str = "v1"
    tracks: list[TrackId]
    tasks: list[TaskId]
    models: list[str]
    runtime_default: RuntimeName
    runtime_overrides: dict[str, RuntimeName] = Field(default_factory=dict)
    parameters: dict[str, Any] = Field(default_factory=dict)
    smoke_dataset: str | None = None


class ModelConfig(ConfigModel):
    """Benchmark model configuration."""

    model_id: str
    hf_id: str
    revision: str = "main"
    tokenizer_revision: str = "main"
    family: str = "llm"
    max_context_tokens: int = 131072  # NOTE: Default 128K when omitted in YAML.


class RuntimeConfig(ConfigModel):
    """Runtime backend configuration."""

    runtime_id: RuntimeName
    payload: dict[str, Any]


class TaskConfig(ConfigModel):
    """Benchmark task configuration."""

    task_id: TaskId
    dataset: str
    dataset_revision: str = "main"
    payload: dict[str, Any]



def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file and enforce top-level dictionary shape."""

    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Invalid config format: {path}")
    return value



def load_suite(path: Path) -> SuiteConfig:
    """Load and validate a suite YAML file.

    Args:
        path: Path to suite configuration.

    Returns:
        Validated suite configuration.
    """

    raw = _load_yaml(path)
    return SuiteConfig.model_validate(raw)



def load_model(model_id: str) -> ModelConfig:
    """Load a model configuration and fail if missing.

    Args:
        model_id: Model identifier mapped to `configs/models/<model_id>.yaml`.

    Returns:
        Validated model configuration.
    """

    path = Path("configs/models") / f"{model_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Model config not found: {model_id} ({path})")
    raw = _load_yaml(path)
    return ModelConfig.model_validate(raw)



def load_runtime(runtime_id: RuntimeName) -> RuntimeConfig:
    """Load a runtime configuration.

    Args:
        runtime_id: Runtime identifier.

    Returns:
        Validated runtime configuration.
    """

    path = Path("configs/runtimes") / f"{runtime_id.value}.yaml"
    raw = _load_yaml(path)
    runtime_name = RuntimeName(str(raw.get("runtime_id", runtime_id.value)))
    return RuntimeConfig(runtime_id=runtime_name, payload=raw)



def load_task(task_id: TaskId) -> TaskConfig:
    """Load a task configuration.

    Args:
        task_id: Task identifier.

    Returns:
        Validated task configuration.
    """

    path = Path("configs/tasks") / f"{task_id.value}.yaml"
    raw = _load_yaml(path)
    return TaskConfig(
        task_id=TaskId(str(raw["task_id"])),
        dataset=str(raw["dataset"]),
        dataset_revision=str(raw.get("dataset_revision", "main")),
        payload=raw,
    )



def resolve_tasks(selection: str, available: list[TaskId]) -> list[TaskId]:
    """Resolve task selection coming from CLI arguments.

    Args:
        selection: CLI selection (`all` or one task id).
        available: Available tasks from suite config.

    Returns:
        Resolved task list.
    """

    if selection == "all":
        return list(available)
    return [TaskId(selection)]



def resolve_tracks(selection: str, available: list[TrackId]) -> list[TrackId]:
    """Resolve track selection coming from CLI arguments.

    Args:
        selection: CLI selection (`all`, `zeroshot`, `fewshot`).
        available: Available tracks from suite config.

    Returns:
        Resolved track list.
    """

    if selection == "all":
        return list(available)
    if selection == "zeroshot":
        picked = TrackId.ZEROSHOT
    else:
        picked = TrackId.FEWSHOT
    return [picked]



def resolve_models(selection: str, available: list[str]) -> list[str]:
    """Resolve model selection coming from CLI arguments.

    Args:
        selection: CLI selection (`all` or one model id).
        available: Available model ids from suite config.

    Returns:
        Resolved model list.
    """

    if selection == "all":
        return list(available)
    return [selection]
