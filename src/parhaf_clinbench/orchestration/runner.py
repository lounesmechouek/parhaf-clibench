"""Main orchestrator for benchmark execution."""

from __future__ import annotations

import json
import logging
import math
import os
import platform
import signal
import statistics
import subprocess
import time
from contextlib import contextmanager, nullcontext
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

from parhaf_clinbench.core.enums import RuntimeName, TaskId, TrackId
from parhaf_clinbench.core.hashing import stable_sha256_text
from parhaf_clinbench.core.ids import make_run_id
from parhaf_clinbench.core.models import (
    BootstrapInterval,
    CanonicalDocument,
    DocumentExample,
    InferenceRequest,
    PredictionOutcome,
    RunMetadata,
    TaskMetrics,
    TrackReport,
)
from parhaf_clinbench.core.settings import get_settings
from parhaf_clinbench.data.canonicalize import canonical_to_dict
from parhaf_clinbench.data.hf_loaders import (
    examples_fingerprint,
    load_hf_examples,
    load_smoke_examples,
)
from parhaf_clinbench.data.prefetch import (
    DatasetPrefetchResult,
    prefetch_hf_dataset,
    resolve_local_dataset_path,
)
from parhaf_clinbench.orchestration.artifact_store import ArtifactStore
from parhaf_clinbench.orchestration.experiment_plan import (
    ModelConfig,
    RuntimeConfig,
    SuiteConfig,
    TaskConfig,
    load_model,
    load_runtime,
    load_suite,
    load_task,
    resolve_models,
    resolve_tasks,
    resolve_tracks,
)
from parhaf_clinbench.orchestration.healthchecks import run_healthcheck
from parhaf_clinbench.parsing.align import align_offsets
from parhaf_clinbench.parsing.validate import validate_and_parse
from parhaf_clinbench.prompting.render import render_prompt
from parhaf_clinbench.reporting.export import export_reports
from parhaf_clinbench.runtimes.gliner import GlinerRuntime
from parhaf_clinbench.runtimes.mock import MockRuntime
from parhaf_clinbench.runtimes.prefetch import PrefetchResult, prefetch_hf_model
from parhaf_clinbench.runtimes.vllm import VllmRuntime
from parhaf_clinbench.scoring.bootstrap import bootstrap_global_score, bootstrap_official_score
from parhaf_clinbench.scoring.common import DocCounts, ScoreComputation
from parhaf_clinbench.scoring.infectio import compute_infectio_metrics
from parhaf_clinbench.scoring.pseudo import compute_pseudo_metrics
from parhaf_clinbench.scoring.response import compute_response_metrics
from parhaf_clinbench.scoring.scenario import compute_scenario_metrics


def _pct95(values: list[float]) -> float:
    """Return the empirical 95th percentile for a list of values."""

    if not values:
        return 0.0
    sorted_values = sorted(values)
    idx = math.ceil(0.95 * len(sorted_values)) - 1
    idx = max(0, min(idx, len(sorted_values) - 1))
    return float(sorted_values[idx])


def _token_count(text: str) -> int:
    """Approximate token count using whitespace splitting."""

    return len(text.split())


def _configure_run_logger(run_dir: Path) -> logging.Logger:
    """Configure and return the run-scoped logger."""

    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"parhaf_clinbench.run.{run_dir.name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    logger.handlers = []

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_handler = logging.FileHandler(log_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    return logger


def _log_event(logger: logging.Logger, event: str, **payload: Any) -> None:
    """Write one structured JSON event to the run log."""

    logger.info(json.dumps({"event": event, **payload}, ensure_ascii=False))


def _empty_prediction(example_doc_id: str, task: TaskId, speciality: str | None) -> CanonicalDocument:
    """Build an empty canonical prediction used for fail-fast invalid outputs."""

    if task == TaskId.SCENARIO and speciality is None:
        raise ValueError(
            "Cannot generate an empty prediction for `scenario` without `speciality`."
        )
    return CanonicalDocument(
        document_id=example_doc_id,
        task=task,
        speciality=speciality,
        records=[],
    )


def _robustness_metrics(
    outcomes: list[PredictionOutcome],
    input_tokens: list[int],
    output_tokens: list[int],
) -> dict[str, float]:
    """Compute robustness and latency statistics over prediction outcomes."""

    n = len(outcomes)
    if n == 0:
        return {
            "raw_json_valid_rate": 0.0,
            "repair_applied_rate": 0.0,
            "schema_conformity_rate": 0.0,
            "empty_output_rate": 0.0,
            "latency_mean_ms": 0.0,
            "latency_median_ms": 0.0,
            "latency_p95_ms": 0.0,
            "input_tokens_mean": 0.0,
            "output_tokens_mean": 0.0,
            "throughput_tokens_per_second": 0.0,
        }

    latencies = [item.latency_ms for item in outcomes]
    raw_json_valid = sum(1 for item in outcomes if item.raw_json_valid)
    repair_applied = sum(1 for item in outcomes if item.repair_applied)
    schema_valid = sum(1 for item in outcomes if item.is_schema_valid)
    empty_outputs = sum(1 for item in outcomes if item.parsed is None or len(item.parsed.records) == 0)

    total_output_tokens = sum(output_tokens)
    total_latency_seconds = sum(latencies) / 1000.0
    throughput = (total_output_tokens / total_latency_seconds) if total_latency_seconds > 0 else 0.0

    return {
        "raw_json_valid_rate": raw_json_valid / n,
        "repair_applied_rate": repair_applied / n,
        "schema_conformity_rate": schema_valid / n,
        "empty_output_rate": empty_outputs / n,
        "latency_mean_ms": float(statistics.mean(latencies)),
        "latency_median_ms": float(statistics.median(latencies)),
        "latency_p95_ms": _pct95(latencies),
        "input_tokens_mean": float(statistics.mean(input_tokens)) if input_tokens else 0.0,
        "output_tokens_mean": float(statistics.mean(output_tokens)) if output_tokens else 0.0,
        "throughput_tokens_per_second": throughput,
    }


def _build_runtime(
    runtime_name: RuntimeName,
    model_reference: str,
    runtime_payload: dict[str, Any],
    *,
    hf_token: str | None,
    tokenizer_revision: str = "main",
    max_context_tokens: int = 131072,
) -> MockRuntime | GlinerRuntime | VllmRuntime:
    """Instantiate the runtime backend selected for the current model."""

    if runtime_name == RuntimeName.MOCK:
        return MockRuntime()
    if runtime_name == RuntimeName.GLINER:
        return GlinerRuntime(
            model_reference=model_reference,
            hf_token=hf_token,
            device=str(runtime_payload.get("device", "auto")),
            threshold=float(runtime_payload.get("threshold", 0.5)),
            flat_ner=bool(runtime_payload.get("flat_ner", True)),
            multi_label=bool(runtime_payload.get("multi_label", False)),
            batch_size=int(runtime_payload.get("batch_size", 8)),
            negation_window_chars=int(runtime_payload.get("negation_window_chars", 48)),
            max_context_tokens=max_context_tokens,
            tokenizer_revision=tokenizer_revision,
        )
    if runtime_name == RuntimeName.VLLM:
        api_base = str(runtime_payload.get("api_base", "http://127.0.0.1:8000/v1"))
        timeout_seconds = int(runtime_payload.get("timeout_seconds", 120))
        max_tokens_raw = runtime_payload.get("max_tokens")
        seed_raw = runtime_payload.get("seed")
        max_tokens = int(max_tokens_raw) if max_tokens_raw is not None else None
        seed = int(seed_raw) if seed_raw is not None else None
        return VllmRuntime(
            api_base=api_base,
            model_hf_id=model_reference,
            timeout_seconds=timeout_seconds,
            temperature=float(runtime_payload.get("temperature", 0.0)),
            top_p=float(runtime_payload.get("top_p", 1.0)),
            max_tokens=max_tokens,
            seed=seed,
            max_context_tokens=max_context_tokens,
            tokenizer_revision=tokenizer_revision,
        )
    raise ValueError(f"Unsupported runtime: {runtime_name}")


def _resolve_vllm_runtime_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Resolve vLLM endpoints from settings for all execution contexts."""

    settings = get_settings()
    resolved = dict(payload)
    resolved["api_base"] = f"http://{settings.vllm_host}:{settings.vllm_port}/v1"
    resolved["healthcheck_url"] = f"http://{settings.vllm_host}:{settings.vllm_port}/health"
    return resolved


def _apply_suite_generation_params_to_vllm_payload(
    payload: dict[str, Any],
    suite_parameters: dict[str, Any],
) -> dict[str, Any]:
    """Inject suite generation parameters into vLLM runtime payload."""

    resolved = dict(payload)
    for key in ("temperature", "top_p", "max_tokens", "seed"):
        value = suite_parameters.get(key)
        if value is None:
            continue
        resolved[key] = value
    return resolved


def _order_models_for_execution(
    *,
    selected_models: list[str],
    suite: SuiteConfig,
    model_selection: str,
) -> list[str]:
    """Order models for execution, prioritizing GLiNER when selecting `all`."""

    if model_selection != "all":
        return list(selected_models)
    gliner_first: list[str] = []
    others: list[str] = []
    for model_id in selected_models:
        runtime_name = suite.runtime_overrides.get(model_id, suite.runtime_default)
        if runtime_name == RuntimeName.GLINER:
            gliner_first.append(model_id)
        else:
            others.append(model_id)
    return gliner_first + others


def _tracks_for_runtime(
    *,
    runtime_name: RuntimeName,
    selected_tracks: list[TrackId],
    track_selection: str,
) -> list[TrackId]:
    """Resolve effective tracks for a runtime with backend-specific constraints."""

    if runtime_name != RuntimeName.GLINER:
        return list(selected_tracks)
    if track_selection == "fewshot":
        raise ValueError("GLiNER only supports the `zero-shot` track.")
    if TrackId.ZEROSHOT in selected_tracks:
        return [TrackId.ZEROSHOT]
    raise ValueError("GLiNER requires the `zero-shot` track.")


def _wait_http_ready(url: str, timeout_seconds: int) -> None:
    """Wait until an HTTP URL responds with a 2xx status."""

    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            with urlopen(url, timeout=2) as response:
                status = getattr(response, "status", 200)
                if 200 <= int(status) < 300:
                    return
        except (URLError, TimeoutError, ValueError):
            pass
        time.sleep(1)
    raise TimeoutError(f"Timeout healthcheck HTTP: {url}")


@contextmanager
def _managed_vllm_server(
    *,
    model_reference: str,
    runtime_payload: dict[str, Any],
    logger: logging.Logger,
    log_path: Path,
) -> Any:
    """Start and stop a managed local vLLM server for a model."""

    health_url = str(runtime_payload.get("healthcheck_url", "http://127.0.0.1:8000/health"))
    host = str(runtime_payload.get("api_base", "http://127.0.0.1:8000/v1")).split("://", 1)[-1].split("/", 1)[0]
    timeout_seconds = int(runtime_payload.get("startup_timeout_seconds", 180))
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            [
                "vllm",
                "serve",
                model_reference,
                "--host",
                str(get_settings().vllm_host),
                "--port",
                str(get_settings().vllm_port),
            ],
            stdout=handle,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        _log_event(
            logger,
            "vllm_server_start",
            model_reference=model_reference,
            pid=process.pid,
            bind=host,
            health_url=health_url,
            log_path=str(log_path),
        )
        try:
            _wait_http_ready(health_url, timeout_seconds=timeout_seconds)
            _log_event(logger, "vllm_server_ready", model_reference=model_reference)
            yield
        finally:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    process.wait(timeout=10)
            _log_event(logger, "vllm_server_stopped", model_reference=model_reference, returncode=process.returncode)


def _compute_task_metrics(
    task: TaskId,
    predictions: list[CanonicalDocument],
    references: list[CanonicalDocument],
    robustness: dict[str, float],
) -> ScoreComputation:
    """Dispatch to the task-specific scoring implementation."""

    if task == TaskId.PSEUDO:
        return compute_pseudo_metrics(predictions=predictions, references=references, robustness=robustness)
    if task == TaskId.INFECTIO:
        return compute_infectio_metrics(predictions=predictions, references=references, robustness=robustness)
    if task == TaskId.RESPONSE:
        return compute_response_metrics(predictions=predictions, references=references, robustness=robustness)
    return compute_scenario_metrics(predictions=predictions, references=references, robustness=robustness)


def _load_examples(task: TaskId, task_cfg: TaskConfig, suite: SuiteConfig) -> list[DocumentExample]:
    """Load either smoke or Hugging Face examples for one task."""

    settings = get_settings()
    if suite.smoke_dataset:
        return load_smoke_examples(Path(suite.smoke_dataset), task=task)
    dataset_cache_dir = resolve_local_dataset_path(
        settings.dataset_cache_root,
        task_cfg.dataset,
        task_cfg.dataset_revision,
    )
    return load_hf_examples(
        task=task,
        dataset_name=task_cfg.dataset,
        dataset_revision=task_cfg.dataset_revision,
        cache_dir=dataset_cache_dir,
    )


def _dataset_prefetch_configs_for_task(task: TaskId) -> list[str] | None:
    """Return dataset configurations required for task-level prefetch."""

    if task == TaskId.SCENARIO:
        return None
    return ["document_metadata", "spans"]


def _git_sha() -> str:
    """Return current git SHA, or `'unknown'` when unavailable."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
        )
        value = result.stdout.strip()
        if result.returncode == 0 and value:
            return value
    except Exception:
        pass
    return "unknown"


def _load_fewshot(task: TaskId) -> str:
    """Load few-shot examples for a task, returning empty string if missing."""

    path = Path("assets/fewshot") / f"{task.value}_examples.txt"
    if path.exists():
        return path.read_text(encoding="utf-8")
    return ""


def _dump_yaml(payload: dict[str, Any]) -> str:
    """Serialize a dictionary to YAML, with JSON fallback."""

    try:
        import yaml

        return yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
    except Exception:
        return json.dumps(payload, ensure_ascii=False, indent=2)


def _resolve_model_reference(
    *,
    model_cfg: ModelConfig,
    runtime_name: RuntimeName,
) -> tuple[str, PrefetchResult | None]:
    """Resolve model reference and prefetch local artifacts when required."""

    settings = get_settings()
    settings.apply_hf_cache_env()
    if runtime_name not in {RuntimeName.VLLM, RuntimeName.GLINER}:
        return model_cfg.hf_id, None
    prefetch = prefetch_hf_model(
        hf_id=model_cfg.hf_id,
        revision=model_cfg.revision,
        cache_root=settings.model_cache_root,
        hf_token=settings.hf_token,
    )
    return prefetch.local_path, prefetch


def run_campaign(
    *,
    suite_path: Path,
    task_selection: str,
    track_selection: str,
    model_selection: str,
    output_dir: Path,
) -> list[Path]:
    """Execute a benchmark campaign and return produced run directories.

    Args:
        suite_path: Path to suite YAML configuration.
        task_selection: Task selector (`all` or one task id).
        track_selection: Track selector (`all`, `zeroshot`, `fewshot`).
        model_selection: Model selector (`all` or one model id).
        output_dir: Base output directory where run artifacts are written.

    Returns:
        List of produced run directories.

    Examples:
        >>> runs = run_campaign(
        ...     suite_path=Path("configs/suites/v1_smoke.yaml"),
        ...     task_selection="all",
        ...     track_selection="all",
        ...     model_selection="all",
        ...     output_dir=Path("results/smoke"),
        ... )
        >>> isinstance(runs, list)
        True
    """

    suite = load_suite(suite_path)
    tasks = resolve_tasks(task_selection, suite.tasks)
    selected_tracks = resolve_tracks(track_selection, suite.tracks)
    selected_models = resolve_models(model_selection, suite.models)
    execution_models = _order_models_for_execution(
        selected_models=selected_models,
        suite=suite,
        model_selection=model_selection,
    )
    settings = get_settings()

    output_dir.mkdir(parents=True, exist_ok=True)
    produced_runs: list[Path] = []

    for model_index, model_id in enumerate(execution_models, start=1):
        model_cfg = load_model(model_id)
        runtime_name = suite.runtime_overrides.get(model_id, suite.runtime_default)
        tracks = _tracks_for_runtime(
            runtime_name=runtime_name,
            selected_tracks=selected_tracks,
            track_selection=track_selection,
        )
        runtime_cfg = load_runtime(runtime_name)
        runtime_payload = dict(runtime_cfg.payload)
        if runtime_name == RuntimeName.VLLM:
            runtime_payload = _resolve_vllm_runtime_payload(runtime_payload)
            runtime_payload = _apply_suite_generation_params_to_vllm_payload(
                runtime_payload,
                suite.parameters,
            )

        run_id = make_run_id(prefix=model_id)
        run_dir = output_dir / run_id
        store = ArtifactStore(run_dir)
        logger = _configure_run_logger(run_dir)
        produced_runs.append(run_dir)

        _log_event(
            logger,
            "run_started",
            run_id=run_id,
            model_id=model_id,
            runtime=runtime_name.value,
            execution_index=model_index,
            execution_total=len(execution_models),
            execution_order=execution_models,
            tracks_requested=[track.value for track in selected_tracks],
            tracks_resolved=[track.value for track in tracks],
        )
        if runtime_name == RuntimeName.GLINER and tracks != selected_tracks:
            _log_event(
                logger,
                "gliner_track_override",
                tracks_requested=[track.value for track in selected_tracks],
                tracks_resolved=[track.value for track in tracks],
            )
        try:
            model_reference, prefetch_result = _resolve_model_reference(
                model_cfg=model_cfg,
                runtime_name=runtime_name,
            )
        except Exception as exc:
            _log_event(
                logger,
                "model_prefetch_failed",
                model_id=model_id,
                hf_id=model_cfg.hf_id,
                revision=model_cfg.revision,
                error=str(exc),
            )
            raise RuntimeError(
                f"Model prefetch failed for `{model_id}` ({model_cfg.hf_id}@{model_cfg.revision}). "
                "Benchmark aborted."
            ) from exc
        if prefetch_result is not None:
            _log_event(
                logger,
                "model_prefetch",
                hf_id=prefetch_result.hf_id,
                revision=prefetch_result.revision,
                local_path=prefetch_result.local_path,
                cache_hit=prefetch_result.cache_hit,
            )
        server_context: Any = nullcontext()
        if runtime_name == RuntimeName.VLLM:
            server_context = _managed_vllm_server(
                model_reference=model_reference,
                runtime_payload=runtime_payload,
                logger=logger,
                log_path=run_dir / "logs" / "vllm_server.log",
            )

        with server_context:
            resolved_runtime_cfg = RuntimeConfig(runtime_id=runtime_name, payload=runtime_payload)
            run_healthcheck(runtime_name, resolved_runtime_cfg)
            runtime = _build_runtime(
                runtime_name,
                model_reference,
                runtime_payload,
                hf_token=settings.hf_token,
            )

            metadata = RunMetadata.start(
                run_id=run_id,
                suite_id=suite.suite_id,
                task_ids=[task.value for task in tasks],
                track_ids=[track.value for track in tracks],
                model_id=model_cfg.model_id,
                model_hf_id=model_cfg.hf_id,
                model_revision=model_cfg.revision,
                tokenizer_revision=model_cfg.tokenizer_revision,
                runtime_name=runtime.name,
                runtime_version=runtime.version,
            )
            metadata.runtime_server_args = dict(runtime_payload)
            metadata.structured_outputs_config = {
                "response_format": {
                    "type": "json_schema",
                    "strict": True,
                    "mode": "task_specific",
                }
            }
            metadata.image_digest = settings.parhaf_image_digest
            metadata.runpod_pod_id = settings.runpod_pod_id
            metadata.runpod_template_id = settings.runpod_template_id
            metadata.gpu_name = settings.runpod_gpu_name
            metadata.gpu_count = settings.runpod_gpu_count
            metadata.vram_gb = settings.runpod_vram_gb
            metadata.container_disk_gb = settings.runpod_container_disk_gb
            metadata.volume_gb = settings.runpod_volume_gb
            metadata.export_mode = settings.parhaf_export_mode
            metadata.export_destination = settings.parhaf_export_destination
            metadata.model_local_path = prefetch_result.local_path if prefetch_result is not None else None
            metadata.no_download_needed = prefetch_result.cache_hit if prefetch_result is not None else None
            metadata.model_execution_order = list(execution_models)
            metadata.model_execution_index = model_index
            metadata.model_execution_total = len(execution_models)

            track_reports: list[TrackReport] = []
            store.write_text("predictions.jsonl", "")
            store.write_text("errors.jsonl", "")
            store.write_text("timings.jsonl", "")
            prompt_snapshots: dict[str, dict[str, str]] = {}

            task_examples: dict[TaskId, list[DocumentExample]] = {}
            task_fingerprints: dict[TaskId, str] = {}

            for task in tasks:
                task_cfg = load_task(task)
                metadata.dataset_revisions[task.value] = task_cfg.dataset_revision
                dataset_prefetch: DatasetPrefetchResult | None = None
                if not suite.smoke_dataset:
                    try:
                        dataset_prefetch = prefetch_hf_dataset(
                            dataset_name=task_cfg.dataset,
                            revision=task_cfg.dataset_revision,
                            cache_root=settings.dataset_cache_root,
                            hf_token=settings.hf_token,
                            configs=_dataset_prefetch_configs_for_task(task),
                        )
                    except Exception as exc:
                        _log_event(
                            logger,
                            "dataset_prefetch_failed",
                            task=task.value,
                            dataset=task_cfg.dataset,
                            revision=task_cfg.dataset_revision,
                            error=str(exc),
                        )
                        raise RuntimeError(
                            f"Dataset prefetch failed for `{task_cfg.dataset}` "
                            f"({task_cfg.dataset_revision}) on task `{task.value}`. "
                            "Benchmark aborted."
                        ) from exc
                    metadata.dataset_cache_hits[task.value] = dataset_prefetch.cache_hit
                    _log_event(
                        logger,
                        "dataset_prefetch",
                        task=task.value,
                        dataset=task_cfg.dataset,
                        revision=task_cfg.dataset_revision,
                        local_path=dataset_prefetch.local_path,
                        cache_hit=dataset_prefetch.cache_hit,
                    )
                examples = _load_examples(task, task_cfg, suite)
                task_examples[task] = examples
                task_fingerprints[task] = examples_fingerprint(
                    dataset_name=task_cfg.dataset,
                    examples=examples,
                )

            metadata.dataset_fingerprint = stable_sha256_text(
                "|".join(
                    f"{task.value}:{task_fingerprints[task]}"
                    for task in sorted(task_fingerprints, key=lambda value: value.value)
                )
            )

            started = time.perf_counter()
            for track in tracks:
                per_task_metrics: dict[str, TaskMetrics] = {}
                per_task_bootstrap: dict[str, BootstrapInterval] = {}
                per_task_doc_counts: dict[str, list[DocCounts]] = {}

                for task in tasks:
                    examples = task_examples[task]
                    fewshot_examples = _load_fewshot(task) if track == TrackId.FEWSHOT else ""
                    probe_speciality_metadata = "__SPECIALITY_METADATA__" if task == TaskId.SCENARIO else None
                    probe_prompt = render_prompt(
                        task=task,
                        track=track,
                        document_id="hash-probe",
                        text="probe",
                        fewshot_examples=fewshot_examples,
                        speciality_metadata=probe_speciality_metadata,
                    )
                    prompt_hash = stable_sha256_text(probe_prompt)
                    key = f"{task.value}:{track.value}"
                    metadata.prompt_hashes[key] = prompt_hash
                    prompt_snapshots[key] = {"hash": prompt_hash, "prompt": probe_prompt}
                    if track == TrackId.FEWSHOT:
                        metadata.fewshot_hash = stable_sha256_text(fewshot_examples)

                    outcomes: list[PredictionOutcome] = []
                    predictions: list[CanonicalDocument] = []
                    references: list[CanonicalDocument] = []
                    input_tokens: list[int] = []
                    output_tokens: list[int] = []

                    for example in examples:
                        speciality_metadata = example.speciality if task == TaskId.SCENARIO else None
                        prompt = render_prompt(
                            task=task,
                            track=track,
                            document_id=example.document_id,
                            text=example.text,
                            fewshot_examples=fewshot_examples,
                            speciality_metadata=speciality_metadata,
                        )
                        request = InferenceRequest(
                            document_id=example.document_id,
                            task=task,
                            track=track,
                            prompt=prompt,
                            text=example.text,
                            gold=example.gold,
                        )

                        t0 = time.perf_counter()
                        error: str | None = None
                        try:
                            raw_output = runtime.infer(request)
                        except Exception as exc:
                            raw_output = ""
                            parsed_doc = None
                            raw_json_valid = False
                            repair_applied = False
                            schema_valid = False
                            error = f"Runtime error: {exc}"
                        else:
                            parsed_doc, raw_json_valid, repair_applied, schema_valid, error = validate_and_parse(
                                raw_output,
                                task,
                            )
                            if parsed_doc is not None:
                                parsed_doc = parsed_doc.model_copy(
                                    update={
                                        "records": [
                                            align_offsets(rec, example.text)
                                            for rec in parsed_doc.records
                                        ]
                                    }
                                )

                        latency_ms = (time.perf_counter() - t0) * 1000.0

                        outcome = PredictionOutcome(
                            document_id=example.document_id,
                            task=task,
                            raw_output=raw_output,
                            parsed=parsed_doc,
                            raw_json_valid=raw_json_valid,
                            repair_applied=repair_applied,
                            is_schema_valid=schema_valid,
                            error=error,
                            latency_ms=latency_ms,
                        )
                        outcomes.append(outcome)

                        input_tok = _token_count(prompt)
                        output_tok = _token_count(raw_output)
                        input_tokens.append(input_tok)
                        output_tokens.append(output_tok)
                        store.append_jsonl(
                            "timings.jsonl",
                            {
                                "document_id": example.document_id,
                                "task": task.value,
                                "track": track.value,
                                "latency_ms": latency_ms,
                                "input_tokens": input_tok,
                                "output_tokens": output_tok,
                            },
                        )

                        # NOTE: Fail-fast policy: no regeneration attempt.
                        # NOTE: Invalid outputs (JSON/schema/contract) become
                        # NOTE: empty predictions and count in robustness metrics.
                        final_doc = (
                            parsed_doc
                            if parsed_doc is not None and schema_valid
                            else _empty_prediction(example.document_id, task, example.speciality)
                        )
                        predictions.append(final_doc)
                        references.append(example.gold)

                        store.append_jsonl(
                            "predictions.jsonl",
                            {
                                "document_id": example.document_id,
                                "task": task.value,
                                "track": track.value,
                                "raw_output": raw_output,
                                "parsed": canonical_to_dict(final_doc),
                                "raw_json_valid": raw_json_valid,
                                "repair_applied": repair_applied,
                                "is_schema_valid": schema_valid,
                            },
                        )
                        if error is not None:
                            store.append_jsonl(
                                "errors.jsonl",
                                {
                                    "document_id": example.document_id,
                                    "task": task.value,
                                    "track": track.value,
                                    "error": error,
                                    "raw_output": raw_output,
                                },
                            )

                    robustness = _robustness_metrics(outcomes, input_tokens, output_tokens)
                    scoring = _compute_task_metrics(task, predictions, references, robustness)
                    per_task_metrics[task.value] = scoring.metrics
                    per_task_doc_counts[task.value] = scoring.official_doc_counts
                    per_task_bootstrap[task.value] = bootstrap_official_score(
                        doc_counts=scoring.official_doc_counts,
                        repetitions=1000,
                        seed=42,
                    )

                global_bootstrap = bootstrap_global_score(
                    per_task_doc_counts=per_task_doc_counts,
                    repetitions=1000,
                    seed=42,
                )
                track_reports.append(
                    TrackReport(
                        track=track,
                        per_task=per_task_metrics,
                        per_task_bootstrap=per_task_bootstrap,
                        global_score=global_bootstrap.score_full,
                        global_bootstrap=global_bootstrap,
                    )
                )

            elapsed = time.perf_counter() - started
            metadata.finished_at_utc = datetime.now(tz=UTC).isoformat()
            metadata.elapsed_seconds = elapsed
            metadata.run_status = "success"

            export_reports(run_dir=run_dir, run_id=run_id, reports=track_reports)
            store.write_json("run_metadata.json", metadata.model_dump(mode="json"))
            store.write_text(
                "resolved_config.yaml",
                _dump_yaml(
                    {
                        "suite": suite.suite_id,
                        "benchmark_version": suite.benchmark_version,
                        "tasks": [task.value for task in tasks],
                        "tracks": [track.value for track in tracks],
                        "model": model_cfg.model_id,
                        "runtime": runtime_name.value,
                        "parameters": suite.parameters,
                    }
                ),
            )
            store.write_text("git_sha.txt", _git_sha() + "\n")
            store.write_json(
                "environment.json",
                {
                    "python_version": platform.python_version(),
                    "platform": platform.platform(),
                    "runtime_name": runtime.name,
                    "runtime_version": runtime.version,
                },
            )
            store.write_text("docker_image.txt", (metadata.image_digest or "unknown") + "\n")
            store.write_json("server_args.json", runtime_payload)
            store.write_json("dataset_fingerprint.json", {"fingerprint": metadata.dataset_fingerprint})
            store.write_json("prompt_hashes.json", metadata.prompt_hashes)
            store.write_json(
                "run_status.json",
                {
                    "run_id": run_id,
                    "status": metadata.run_status,
                    "started_at_utc": metadata.started_at_utc,
                    "finished_at_utc": metadata.finished_at_utc,
                    "elapsed_seconds": metadata.elapsed_seconds,
                },
            )
            store.write_json(
                "export_manifest.json",
                {
                    "export_mode": metadata.export_mode,
                    "destination": metadata.export_destination or str(run_dir),
                },
            )
            store.write_json(
                "pod_info.json",
                {
                    "runpod_pod_id": metadata.runpod_pod_id,
                    "runpod_template_id": metadata.runpod_template_id,
                    "gpu_name": metadata.gpu_name,
                    "gpu_count": metadata.gpu_count,
                    "vram_gb": metadata.vram_gb,
                },
            )

            prompt_lines: list[str] = []
            for key in sorted(prompt_snapshots):
                prompt_lines.append(f"## {key}")
                prompt_lines.append(f"HASH: {prompt_snapshots[key]['hash']}")
                prompt_lines.append(prompt_snapshots[key]["prompt"])
                prompt_lines.append("")
            store.write_text("prompt_snapshot.txt", "\n".join(prompt_lines).rstrip() + "\n")
            runtime.close()
            _log_event(logger, "run_finished", run_id=run_id, elapsed_seconds=elapsed)

    return produced_runs


def score_from_jsonl(*, task: TaskId, predictions_path: Path, gold_path: Path) -> dict[str, Any]:
    """Compute offline metrics from canonical prediction/gold JSONL files.

    Args:
        task: Task identifier used for consistency checks.
        predictions_path: Path to prediction JSONL file.
        gold_path: Path to gold/reference JSONL file.

    Returns:
        Dictionary containing official and secondary score values.

    Examples:
        >>> metrics = score_from_jsonl(
        ...     task=TaskId.PSEUDO,
        ...     predictions_path=Path("predictions.jsonl"),
        ...     gold_path=Path("gold.jsonl"),
        ... )
        >>> "f1" in metrics
        True
    """

    from parhaf_clinbench.data.canonicalize import dict_to_canonical_document

    refs_by_doc_id: dict[str, CanonicalDocument] = {}
    with predictions_path.open("r", encoding="utf-8") as pred_handle:
        pred_rows = [json.loads(line) for line in pred_handle if line.strip()]

    with gold_path.open("r", encoding="utf-8") as gold_handle:
        for line in gold_handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            document = dict_to_canonical_document(payload)
            if document.task != task:
                raise ValueError(
                    f"Unexpected task in gold for document `{document.document_id}`: "
                    f"{document.task.value} (expected: {task.value})"
                )
            if document.document_id in refs_by_doc_id:
                raise ValueError(f"Duplicate document in gold: `{document.document_id}`")
            refs_by_doc_id[document.document_id] = document

    preds_by_doc_id: dict[str, CanonicalDocument] = {}
    for row in pred_rows:
        payload = row.get("parsed", row)
        if not isinstance(payload, dict):
            raise ValueError("Each prediction line must contain a JSON object")
        document = dict_to_canonical_document(payload)
        if document.task != task:
            raise ValueError(
                f"Unexpected task in prediction for document `{document.document_id}`: "
                f"{document.task.value} (expected: {task.value})"
            )
        if document.document_id in preds_by_doc_id:
            raise ValueError(f"Duplicate document in predictions: `{document.document_id}`")
        preds_by_doc_id[document.document_id] = document

    missing_predictions = sorted(set(refs_by_doc_id) - set(preds_by_doc_id))
    unexpected_predictions = sorted(set(preds_by_doc_id) - set(refs_by_doc_id))
    if missing_predictions or unexpected_predictions:
        raise ValueError(
            "Predictions/references not aligned by document_id. "
            f"missing={missing_predictions[:10]} unexpected={unexpected_predictions[:10]}"
        )

    refs: list[CanonicalDocument] = []
    fixed_preds: list[CanonicalDocument] = []
    for document_id in refs_by_doc_id:
        refs.append(refs_by_doc_id[document_id])
        fixed_preds.append(preds_by_doc_id[document_id])

    robustness = {
        "raw_json_valid_rate": 1.0,
        "repair_applied_rate": 0.0,
        "schema_conformity_rate": 1.0,
        "empty_output_rate": 0.0,
        "latency_mean_ms": 0.0,
        "latency_median_ms": 0.0,
        "latency_p95_ms": 0.0,
        "input_tokens_mean": 0.0,
        "output_tokens_mean": 0.0,
        "throughput_tokens_per_second": 0.0,
    }
    scoring = _compute_task_metrics(task, fixed_preds, refs, robustness)
    boot = bootstrap_official_score(doc_counts=scoring.official_doc_counts, repetitions=1000, seed=42)

    return {
        "task": task.value,
        "official_metric": scoring.metrics.official_name,
        "precision": scoring.metrics.official.precision,
        "recall": scoring.metrics.official.recall,
        "f1": scoring.metrics.official.f1,
        "ci_low": boot.ci_low,
        "ci_high": boot.ci_high,
    }
