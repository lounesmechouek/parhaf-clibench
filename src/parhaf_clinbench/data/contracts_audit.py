"""Audit Hugging Face dataset contracts (labels, attributes, fields)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from parhaf_clinbench.core.enums import TaskId
from parhaf_clinbench.core.models import (
    INFECTIO_LABELS,
    INFECTIO_NEGATIONS,
    PSEUDO_LABELS,
    RESPONSE_LABELS,
    SCENARIO_FIELDS,
    SCENARIO_SPECIALITIES,
)
from parhaf_clinbench.data.prefetch import resolve_local_dataset_path
from parhaf_clinbench.orchestration.experiment_plan import load_suite, load_task


class TaskContractAudit(BaseModel):
    """Audit result for one task dataset contract."""

    model_config = ConfigDict(extra="forbid")

    task: TaskId
    dataset: str
    revision: str
    observed: dict[str, Any] = Field(default_factory=dict)
    expected: dict[str, Any] = Field(default_factory=dict)
    mismatches: list[str] = Field(default_factory=list)
    error: str | None = None


class ContractAuditReport(BaseModel):
    """Audit report for a full benchmark suite."""

    model_config = ConfigDict(extra="forbid")

    suite_id: str
    all_ok: bool
    tasks: list[TaskContractAudit]


def _sorted_set(values: set[str]) -> list[str]:
    """Return sorted values from a set."""

    return sorted(values)


def _extract_first_row_keys(dataset: Any) -> set[str]:
    """Collect keys observed in first rows across dataset splits."""

    keys: set[str] = set()
    for split in dataset:
        rows = dataset[split]
        if len(rows) == 0:
            continue
        first = rows[0]
        if isinstance(first, dict):
            keys.update(str(key) for key in first.keys())
    return keys


def _load_dataset(
    dataset_name: str,
    *,
    revision: str,
    cache_dir: Path,
    name: str | None = None,
    hf_token: str | None = None,
) -> Any:
    """Load a dataset/configuration pair with a fixed cache directory."""

    from datasets import load_dataset  # type: ignore[import-untyped]

    if name is None:
        return load_dataset(
            dataset_name,
            revision=revision,
            cache_dir=str(cache_dir),
            token=hf_token,
        )
    return load_dataset(
        dataset_name,
        name,
        revision=revision,
        cache_dir=str(cache_dir),
        token=hf_token,
    )


def _audit_pseudo(*, dataset: str, revision: str, cache_dir: Path, hf_token: str | None) -> TaskContractAudit:
    """Audit pseudo task labels, roles, and required fields."""

    audit = TaskContractAudit(task=TaskId.PSEUDO, dataset=dataset, revision=revision)
    spans = _load_dataset(dataset, name="spans", revision=revision, cache_dir=cache_dir, hf_token=hf_token)
    metadata = _load_dataset(
        dataset,
        name="document_metadata",
        revision=revision,
        cache_dir=cache_dir,
        hf_token=hf_token,
    )

    labels: set[str] = set()
    roles: set[str] = set()
    for split in spans:
        for row in spans[split]:
            label = row.get("attribute_Categorie")
            if label is not None and str(label).strip():
                labels.add(str(label))
            role = row.get("attribute_RolePER")
            if role is not None and str(role).strip():
                roles.add(str(role))

    spans_keys = _extract_first_row_keys(spans)
    metadata_keys = _extract_first_row_keys(metadata)
    observed_labels = _sorted_set(labels)
    observed_roles = _sorted_set(roles)

    audit.observed = {
        "labels": observed_labels,
        "attribute_RolePER_values": observed_roles,
        "spans_keys": sorted(spans_keys),
        "document_metadata_keys": sorted(metadata_keys),
    }
    audit.expected = {
        "labels": _sorted_set(set(PSEUDO_LABELS)),
        "required_spans_keys": ["begin", "end", "report", "span_text"],
        "required_document_metadata_keys": ["full_text", "report"],
    }

    if set(observed_labels) != set(PSEUDO_LABELS):
        audit.mismatches.append("Pseudo observed labels do not match expected labels.")
    if not {"begin", "end", "report", "span_text"}.issubset(spans_keys):
        audit.mismatches.append("Pseudo spans: missing required keys.")
    if "attribute_Categorie" not in spans_keys:
        audit.mismatches.append("Pseudo spans: label key `attribute_Categorie` is missing.")
    if not {"full_text", "report"}.issubset(metadata_keys):
        audit.mismatches.append("Pseudo document_metadata: missing required keys.")
    return audit


def _audit_infectio(*, dataset: str, revision: str, cache_dir: Path, hf_token: str | None) -> TaskContractAudit:
    """Audit infectio task labels, negations, and required fields."""

    audit = TaskContractAudit(task=TaskId.INFECTIO, dataset=dataset, revision=revision)
    spans = _load_dataset(dataset, name="spans", revision=revision, cache_dir=cache_dir, hf_token=hf_token)

    labels: set[str] = set()
    negations: set[str] = set()
    for split in spans:
        for row in spans[split]:
            label = row.get("attribute_LABEL") or row.get("label")
            if label is not None and str(label).strip():
                labels.add(str(label))
            neg = row.get("attribute_NEGATION")
            if neg is not None and str(neg).strip():
                negations.add(str(neg))

    spans_keys = _extract_first_row_keys(spans)
    observed_labels = _sorted_set(labels)
    observed_negations = _sorted_set(negations)

    audit.observed = {
        "labels": observed_labels,
        "attribute_NEGATION_values": observed_negations,
        "spans_keys": sorted(spans_keys),
    }
    audit.expected = {
        "labels": _sorted_set(set(INFECTIO_LABELS)),
        "attribute_NEGATION_values": _sorted_set(set(INFECTIO_NEGATIONS)),
        "required_spans_keys": ["attribute_LABEL", "attribute_NEGATION", "span_text"],
    }

    if set(observed_labels) != set(INFECTIO_LABELS):
        audit.mismatches.append("Infectio observed labels do not match expected labels.")
    if set(observed_negations) != set(INFECTIO_NEGATIONS):
        audit.mismatches.append("Infectio observed negations do not match expected values.")
    if not {"span_text", "attribute_LABEL", "attribute_NEGATION"}.issubset(spans_keys):
        audit.mismatches.append("Infectio spans: missing required keys.")
    return audit


def _audit_response(*, dataset: str, revision: str, cache_dir: Path, hf_token: str | None) -> TaskContractAudit:
    """Audit response task labels and required metadata/span fields."""

    audit = TaskContractAudit(task=TaskId.RESPONSE, dataset=dataset, revision=revision)
    metadata = _load_dataset(
        dataset,
        name="document_metadata",
        revision=revision,
        cache_dir=cache_dir,
        hf_token=hf_token,
    )
    spans = _load_dataset(dataset, name="spans", revision=revision, cache_dir=cache_dir, hf_token=hf_token)

    labels: set[str] = set()
    for split in metadata:
        for row in metadata[split]:
            label = row.get("attribute_Nomenclature") or row.get("label")
            if label is not None and str(label).strip():
                labels.add(str(label))
    for split in spans:
        for row in spans[split]:
            label = row.get("attribute_Nomenclature") or row.get("label")
            if label is not None and str(label).strip():
                labels.add(str(label))

    metadata_keys = _extract_first_row_keys(metadata)
    spans_keys = _extract_first_row_keys(spans)
    observed_labels = _sorted_set(labels)

    audit.observed = {
        "labels_union": observed_labels,
        "document_metadata_keys": sorted(metadata_keys),
        "spans_keys": sorted(spans_keys),
    }
    audit.expected = {
        "labels_union": _sorted_set(set(RESPONSE_LABELS)),
        "required_document_metadata_keys": ["attribute_Nomenclature", "full_text", "report"],
        "required_spans_keys": ["attribute_Justification", "span_text"],
    }

    if set(observed_labels) != set(RESPONSE_LABELS):
        audit.mismatches.append("Response observed labels do not match expected labels.")
    if not {"attribute_Nomenclature", "full_text", "report"}.issubset(metadata_keys):
        audit.mismatches.append("Response document_metadata: missing required keys.")
    if not {"attribute_Justification", "span_text"}.issubset(spans_keys):
        audit.mismatches.append("Response spans: missing required keys.")
    return audit


def _audit_scenario(*, dataset: str, revision: str, cache_dir: Path, hf_token: str | None) -> TaskContractAudit:
    """Audit scenario task specialities, fields, and required columns."""

    audit = TaskContractAudit(task=TaskId.SCENARIO, dataset=dataset, revision=revision)
    rows = _load_dataset(dataset, revision=revision, cache_dir=cache_dir, hf_token=hf_token)

    specialities: set[str] = set()
    fields: set[str] = set()
    row_keys = _extract_first_row_keys(rows)
    for split in rows:
        for row in rows[split]:
            speciality = row.get("speciality") or row.get("specialty")
            if speciality is not None and str(speciality).strip():
                specialities.add(str(speciality))
            suggested = row.get("suggested_scenario")
            if isinstance(suggested, dict):
                fields.update(str(key) for key in suggested.keys())

    observed_specialities = _sorted_set(specialities)
    observed_fields = _sorted_set(fields)

    audit.observed = {
        "specialities": observed_specialities,
        "suggested_scenario_fields": observed_fields,
        "dataset_keys": sorted(row_keys),
    }
    audit.expected = {
        "specialities": _sorted_set(set(SCENARIO_SPECIALITIES)),
        "suggested_scenario_fields": _sorted_set(set(SCENARIO_FIELDS)),
        "required_dataset_keys": ["documents", "suggested_scenario"],
    }

    if set(observed_specialities) != set(SCENARIO_SPECIALITIES):
        audit.mismatches.append("Scenario observed specialities do not match expected values.")
    if set(observed_fields) != set(SCENARIO_FIELDS):
        audit.mismatches.append("Scenario observed fields do not match expected fields.")
    if not {"documents", "suggested_scenario"}.issubset(row_keys):
        audit.mismatches.append("Scenario dataset: missing required keys.")
    return audit


def audit_suite_contracts(
    *,
    suite_path: Path,
    dataset_cache_root: Path,
    hf_token: str | None = None,
) -> ContractAuditReport:
    """Load and audit all task datasets declared in a suite.

    Args:
        suite_path: Path to the suite YAML configuration.
        dataset_cache_root: Root path used to resolve local dataset caches.
        hf_token: Optional Hugging Face token.

    Returns:
        Contract audit report with per-task details.

    Examples:
        >>> report = audit_suite_contracts(
        ...     suite_path=Path("configs/suites/v1_full.yaml"),
        ...     dataset_cache_root=Path("/tmp/datasets"),
        ... )
        >>> isinstance(report.all_ok, bool)
        True
    """

    suite = load_suite(suite_path)
    task_reports: list[TaskContractAudit] = []

    for task in suite.tasks:
        task_cfg = load_task(task)
        cache_dir = resolve_local_dataset_path(dataset_cache_root, task_cfg.dataset, task_cfg.dataset_revision)
        try:
            if task == TaskId.PSEUDO:
                report = _audit_pseudo(
                    dataset=task_cfg.dataset,
                    revision=task_cfg.dataset_revision,
                    cache_dir=cache_dir,
                    hf_token=hf_token,
                )
            elif task == TaskId.INFECTIO:
                report = _audit_infectio(
                    dataset=task_cfg.dataset,
                    revision=task_cfg.dataset_revision,
                    cache_dir=cache_dir,
                    hf_token=hf_token,
                )
            elif task == TaskId.RESPONSE:
                report = _audit_response(
                    dataset=task_cfg.dataset,
                    revision=task_cfg.dataset_revision,
                    cache_dir=cache_dir,
                    hf_token=hf_token,
                )
            else:
                report = _audit_scenario(
                    dataset=task_cfg.dataset,
                    revision=task_cfg.dataset_revision,
                    cache_dir=cache_dir,
                    hf_token=hf_token,
                )
        except Exception as exc:
            report = TaskContractAudit(
                task=task,
                dataset=task_cfg.dataset,
                revision=task_cfg.dataset_revision,
                error=str(exc),
            )
            report.mismatches.append("Audit failed: dataset loading error.")
        task_reports.append(report)

    all_ok = all(not report.mismatches and report.error is None for report in task_reports)
    return ContractAuditReport(suite_id=suite.suite_id, all_ok=all_ok, tasks=task_reports)
