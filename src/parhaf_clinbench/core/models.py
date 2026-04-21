"""Core benchmark data models implemented with Pydantic."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from parhaf_clinbench.core.enums import TaskId, TrackId

PSEUDO_LABELS = {
    "ADDRESS",
    "CITY",
    "COUNTRY",
    "FAMILY_STATUS",
    "FIRST_NAME",
    "IDENTIFYING_DATE",
    "LAST_NAME",
    "PATIENT_BIRTHDATE",
    "PATIENT_NATIONALITY",
    "PATIENT_SOCIAL_IDENTITY",
    "PHONE_NUMBER",
    "UNIDENTIFYING_DATE",
    "URL",
}
INFECTIO_LABELS = {"Bacterie", "Bacteriemie", "Infection", "Site"}
INFECTIO_NEGATIONS = {"Absent", "Indetermine", "Present"}
RESPONSE_LABELS = {
    "ReponsePartielle",
    "ReponseComplete",
    "MaladieStable",
    "MaladieProgressive",
    "NonApplicable",
    "NonDetermine",
}
SCENARIO_FIELDS = {
    "name",
    "age",
    "sex",
    "admission_mode",
    "discharge_mode",
    "primary_procedure",
    "primary_diagnosis",
    "type_of_care",
}
SCENARIO_SPECIALITIES = {
    "ANATOMOPATHOLOGIE",
    "CANCERO ADULTE",
    "CARDIOLOGIE",
    "CHIR ORTHO ET TRAUMATO",
    "CHIR.CARDIO-VASC.",
    "CHIRURGIE VISCERALE",
    "GYNECOLOGIE",
    "HEMATOLOGIE CLINIQUE",
    "HEPATO-GASTRO-ENTERO",
    "MALADIES INFECTIEUSES",
    "MEDECINE GERIATRIQUE",
    "MEDECINE INTER-SPECIALITES",
    "MEDECINE INTERNE",
    "MEDECINE PEDIATRIQUE",
    "NEPHROLOGIE",
    "NEUROLOGIE",
    "OBSTETRIQUE",
    "PNEUMOLOGIE",
    "REANIMATION",
    "UROLOGIE",
}


class BenchModel(BaseModel):
    """Shared strict Pydantic base model."""

    model_config = ConfigDict(extra="forbid")


class Record(BenchModel):
    """Canonical extracted record."""

    label: str
    text: str | None = None
    start: int | None = None
    end: int | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)

    @field_validator("label")
    @classmethod
    def _label_not_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("label cannot be empty")
        return cleaned

    @model_validator(mode="after")
    def _validate_span_bounds(self) -> Record:
        if (self.start is None) ^ (self.end is None):
            raise ValueError("start and end must be provided together")
        if self.start is not None and self.end is not None and self.end < self.start:
            raise ValueError("end must be >= start")
        return self


class CanonicalDocument(BenchModel):
    """Unified canonical output format for every task.

    Examples:
        >>> CanonicalDocument(document_id="doc-1", task=TaskId.RESPONSE, records=[])
        CanonicalDocument(document_id='doc-1', task=<TaskId.RESPONSE: 'response'>, speciality=None, records=[])
    """

    document_id: str
    task: TaskId
    speciality: str | None = None
    records: list[Record] = Field(default_factory=list)

    @field_validator("document_id")
    @classmethod
    def _document_id_not_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("document_id cannot be empty")
        return cleaned

    @model_validator(mode="after")
    def _check_task_specific_constraints(self) -> CanonicalDocument:
        if self.task == TaskId.PSEUDO:
            for record in self.records:
                if record.label not in PSEUDO_LABELS:
                    raise ValueError(f"Label pseudo invalide: {record.label}")
                if record.text is None:
                    raise ValueError("pseudo task: text is required for each record")
                if record.start is None or record.end is None:
                    raise ValueError("pseudo task: start/end are required for each record")
                unknown_keys = set(record.attributes) - {"role"}
                if unknown_keys:
                    raise ValueError(
                        f"unsupported pseudo attribute(s): {sorted(unknown_keys)}"
                    )

        if self.task == TaskId.INFECTIO:
            for record in self.records:
                if record.label not in INFECTIO_LABELS:
                    raise ValueError(f"invalid infectio label: {record.label}")
                if record.text is None:
                    raise ValueError("infectio task: text is required")
                unknown_keys = set(record.attributes) - {"negation"}
                if unknown_keys:
                    raise ValueError(
                        f"unsupported infectio attribute(s): {sorted(unknown_keys)}"
                    )
                negation = record.attributes.get("negation")
                if negation is not None and str(negation) not in INFECTIO_NEGATIONS:
                    raise ValueError(f"invalid negation value: {negation}")

        if self.task == TaskId.RESPONSE:
            for record in self.records:
                if record.label not in RESPONSE_LABELS:
                    raise ValueError(f"invalid response label: {record.label}")
                if record.text is None:
                    raise ValueError("response task: text is required")

        if self.task == TaskId.SCENARIO:
            if self.speciality is None:
                raise ValueError("scenario task: speciality is required")
            if self.speciality not in SCENARIO_SPECIALITIES:
                raise ValueError(f"Speciality scenario invalide: {self.speciality}")
            for record in self.records:
                if record.label not in SCENARIO_FIELDS:
                    raise ValueError(f"Label scenario invalide: {record.label}")
        return self


class DocumentExample(BenchModel):
    """Evaluation example containing source text and reference annotation."""

    document_id: str
    task: TaskId
    speciality: str | None = None
    text: str
    gold: CanonicalDocument


class InferenceRequest(BenchModel):
    """Request payload passed to runtime backends."""

    document_id: str
    task: TaskId
    track: TrackId
    prompt: str
    text: str
    gold: CanonicalDocument | None = None


class PredictionOutcome(BenchModel):
    """Inference result after parsing and schema validation."""

    document_id: str
    task: TaskId
    raw_output: str
    parsed: CanonicalDocument | None
    raw_json_valid: bool
    repair_applied: bool
    is_schema_valid: bool
    error: str | None
    latency_ms: float


class RunMetadata(BenchModel):
    """Run-level metadata persisted for auditability."""

    run_id: str
    suite_id: str
    task_ids: list[str]
    track_ids: list[str]
    model_id: str
    model_hf_id: str
    model_revision: str
    tokenizer_revision: str
    runtime_name: str
    runtime_version: str
    runtime_server_args: dict[str, Any] = Field(default_factory=dict)
    structured_outputs_config: dict[str, Any] = Field(default_factory=dict)
    dataset_fingerprint: str = ""
    dataset_revisions: dict[str, str] = Field(default_factory=dict)
    dataset_cache_hits: dict[str, bool] = Field(default_factory=dict)
    prompt_hashes: dict[str, str] = Field(default_factory=dict)
    fewshot_hash: str | None = None
    image_digest: str | None = None
    runpod_pod_id: str | None = None
    runpod_template_id: str | None = None
    gpu_name: str | None = None
    gpu_count: int | None = None
    vram_gb: int | None = None
    container_disk_gb: int | None = None
    volume_gb: int | None = None
    export_mode: str = "local"
    export_destination: str | None = None
    run_status: str = "running"
    started_at_utc: str
    finished_at_utc: str | None = None
    elapsed_seconds: float | None = None
    model_local_path: str | None = None
    no_download_needed: bool | None = None
    model_execution_order: list[str] = Field(default_factory=list)
    model_execution_index: int | None = None
    model_execution_total: int | None = None

    @classmethod
    def start(
        cls,
        *,
        run_id: str,
        suite_id: str,
        task_ids: list[str],
        track_ids: list[str],
        model_id: str,
        model_hf_id: str,
        model_revision: str,
        tokenizer_revision: str,
        runtime_name: str,
        runtime_version: str,
    ) -> RunMetadata:
        """Build initial metadata at run start."""

        return cls(
            run_id=run_id,
            suite_id=suite_id,
            task_ids=task_ids,
            track_ids=track_ids,
            model_id=model_id,
            model_hf_id=model_hf_id,
            model_revision=model_revision,
            tokenizer_revision=tokenizer_revision,
            runtime_name=runtime_name,
            runtime_version=runtime_version,
            started_at_utc=datetime.now(tz=UTC).isoformat(),
        )


class ScoreTriplet(BenchModel):
    """Precision/recall/F1 triplet."""

    precision: float
    recall: float
    f1: float


class TaskMetrics(BenchModel):
    """Per-task metric bundle."""

    task: TaskId
    official: ScoreTriplet
    official_name: str
    secondary: dict[str, ScoreTriplet] = Field(default_factory=dict)
    robustness: dict[str, float] = Field(default_factory=dict)


class BootstrapInterval(BenchModel):
    """Percentile bootstrap confidence interval."""

    score_full: float
    ci_low: float
    ci_high: float
    repetitions: int


class TrackReport(BenchModel):
    """Aggregated report at track level."""

    track: TrackId
    per_task: dict[str, TaskMetrics]
    per_task_bootstrap: dict[str, BootstrapInterval]
    global_score: float
    global_bootstrap: BootstrapInterval
