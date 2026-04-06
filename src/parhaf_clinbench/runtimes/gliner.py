"""GLiNER2 runtime adapter."""

from __future__ import annotations

import gc
import json
import logging
import re
import unicodedata
from typing import Any

from parhaf_clinbench.chunking.merger import merge_canonical_documents
from parhaf_clinbench.chunking.splitter import make_chunks
from parhaf_clinbench.core.enums import TaskId, TrackId
from parhaf_clinbench.core.models import (
    INFECTIO_LABELS,
    PSEUDO_LABELS,
    RESPONSE_LABELS,
    SCENARIO_FIELDS,
    CanonicalDocument,
    InferenceRequest,
    Record,
)
from parhaf_clinbench.runtimes.base import RuntimeBackend

_LOG = logging.getLogger(__name__)

_CHUNK_THRESHOLD = 0.9
_CHUNK_TARGET = 0.85


class _WordCountTokenizer:
    """Minimal tokenizer fallback: one token per whitespace-separated word."""

    def encode(self, text: str, *, add_special_tokens: bool = True) -> list[str]:
        return text.split()

_TASK_LABELS: dict[TaskId, list[str]] = {
    TaskId.PSEUDO: sorted(PSEUDO_LABELS),
    TaskId.INFECTIO: sorted(INFECTIO_LABELS),
    TaskId.RESPONSE: sorted(RESPONSE_LABELS),
    TaskId.SCENARIO: sorted(SCENARIO_FIELDS),
}
_TASK_LABEL_DESCRIPTIONS: dict[TaskId, dict[str, str]] = {
    TaskId.PSEUDO: {
        "LAST_NAME": "Nom de famille d'une personne",
        "FIRST_NAME": "Prénom d'une personne",
        "PATIENT_BIRTHDATE": "Date de naissance du patient",
        "IDENTIFYING_DATE": "Date potentiellement identifiante",
        "UNIDENTIFYING_DATE": "Date non identifiante",
        "ADDRESS": "Adresse postale",
        "CITY": "Ville",
        "COUNTRY": "Pays",
        "PHONE_NUMBER": "Numéro de téléphone",
        "PATIENT_NATIONALITY": "Nationalité du patient",
        "PATIENT_SOCIAL_IDENTITY": "Identité sociale du patient",
        "FAMILY_STATUS": "Statut familial",
        "URL": "URL ou lien web",
    },
    TaskId.INFECTIO: {
        "Bacterie": "Nom de bactérie",
        "Bacteriemie": "Mention de bactériémie",
        "Infection": "Mention d'infection",
        "Site": "Site anatomique ou foyer infectieux",
    },
    TaskId.RESPONSE: {
        "ReponsePartielle": "Réponse partielle au traitement",
        "ReponseComplete": "Réponse complète au traitement",
        "MaladieStable": "Maladie stable",
        "MaladieProgressive": "Progression de la maladie",
        "NonApplicable": "Réponse non applicable",
        "NonDetermine": "Réponse non déterminée",
    },
    TaskId.SCENARIO: {
        "name": "Nom de personne",
        "age": "Age du patient",
        "sex": "Sexe biologique",
        "admission_mode": "Mode d'admission",
        "discharge_mode": "Mode de sortie",
        "primary_procedure": "Procédure principale",
        "primary_diagnosis": "Diagnostic principal",
        "type_of_care": "Type de prise en charge",
    },
}

_ABSENT_CUES = (
    "absence de",
    "absent",
    "aucun",
    "aucune",
    "exclu",
    "exclue",
    "infirmé",
    "infirmee",
    "negatif",
    "négatif",
    "pas de",
    "pas d",
    "sans",
)
_INDETERMINATE_CUES = (
    "doute",
    "evoque",
    "évoque",
    "incertain",
    "indetermine",
    "indéterminé",
    "possible",
    "probable",
    "suspicion",
    "suspect",
)
_CLAUSE_DELIMITERS = (".", ",", ";", "!", "?", "\n")


def _strip_accents(text: str) -> str:
    """Remove diacritics from text while preserving base characters."""

    normalized = unicodedata.normalize("NFKD", text)
    return "".join(char for char in normalized if not unicodedata.combining(char))


def _normalize_label_key(label: str) -> str:
    """Normalize labels into lowercase underscore keys for alias mapping."""

    value = _strip_accents(label).lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_")


def _normalize_context(text: str) -> str:
    """Normalize context text for cue matching in negation heuristics."""

    value = _strip_accents(text).lower()
    value = value.replace("\u2019", "'")
    value = re.sub(r"[^a-z0-9']+", " ", value)
    return re.sub(r"\s+", " ", value).strip()


def _camel_to_words(value: str) -> str:
    """Convert `CamelCase` identifiers into spaced lowercase words."""

    return re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", value)


def _build_label_alias_map(task: TaskId) -> dict[str, str]:
    """Build normalized alias-to-canonical label mapping for a task."""

    aliases: dict[str, str] = {}
    for canonical in _TASK_LABELS[task]:
        candidates = {
            canonical,
            canonical.lower(),
            canonical.replace("_", " "),
            canonical.replace("_", ""),
            _camel_to_words(canonical),
            _camel_to_words(canonical).replace(" ", "_"),
        }
        for candidate in candidates:
            aliases[_normalize_label_key(candidate)] = canonical

    if task == TaskId.RESPONSE:
        aliases.update(
            {
                "progression": "MaladieProgressive",
                "stable": "MaladieStable",
                "complete_response": "ReponseComplete",
                "partial_response": "ReponsePartielle",
                "non_determine": "NonDetermine",
                "not_determined": "NonDetermine",
                "not_applicable": "NonApplicable",
            }
        )
    return aliases


def _extract_clause(text: str, start: int, end: int) -> str:
    """Extract a local clause around `[start:end]` using punctuation delimiters."""

    left = -1
    for delimiter in _CLAUSE_DELIMITERS:
        idx = text.rfind(delimiter, 0, start)
        if idx > left:
            left = idx
    right = len(text)
    for delimiter in _CLAUSE_DELIMITERS:
        idx = text.find(delimiter, end)
        if idx != -1 and idx < right:
            right = idx
    return text[left + 1 : right]


class GlinerRuntime(RuntimeBackend):
    """GLiNER runtime adapted to benchmark canonical contract."""

    def __init__(
        self,
        *,
        model_reference: str,
        hf_token: str | None,
        device: str = "auto",
        threshold: float = 0.5,
        flat_ner: bool = True,
        multi_label: bool = False,
        batch_size: int = 8,
        negation_window_chars: int = 48,
        max_context_tokens: int = 512,
        tokenizer_revision: str = "main",
    ) -> None:
        self._model_reference = model_reference
        self._hf_token = hf_token
        self._device = device
        self._threshold = threshold
        # NOTE: `flat_ner`/`multi_label`/`batch_size` are accepted for YAML
        # NOTE: compatibility but not forwarded to GLiNER2 (unsupported API).
        self._negation_window_chars = negation_window_chars
        self._max_context_tokens = max_context_tokens
        self._tokenizer_revision = tokenizer_revision
        self._tokenizer: Any = None  # NOTE: Lazy tokenizer initialization.
        self._model: Any | None = None
        self._library_version = "unknown"
        self._label_aliases: dict[TaskId, dict[str, str]] = {
            task: _build_label_alias_map(task) for task in _TASK_LABELS
        }

    @property
    def name(self) -> str:
        return "gliner"

    @property
    def version(self) -> str:
        return f"gliner2-{self._library_version}"

    def _resolve_map_location(self) -> str:
        if self._device != "auto":
            return self._device
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        try:
            import gliner2 as gliner2_module  # type: ignore[import-untyped]
            from gliner2 import GLiNER2
        except Exception as exc:
            raise RuntimeError(
                "The `gliner2` package is required for the GLiNER2 runtime. "
                "Install it and re-run the benchmark."
            ) from exc

        map_location = self._resolve_map_location()
        try:
            model = GLiNER2.from_pretrained(
                self._model_reference,
                local_files_only=True,
                token=self._hf_token,
                map_location=map_location,
            )
            if hasattr(model, "to"):
                model = model.to(map_location)
            self._model = model
        except Exception as exc:
            raise RuntimeError(
                "GLiNER2 failed to load from local cache. "
                f"model_reference={self._model_reference} device={map_location} detail={exc}"
            ) from exc
        _LOG.info("GLiNER2 loaded on device=%s", map_location)

        self._library_version = str(getattr(gliner2_module, "__version__", "unknown"))
        return self._model

    def _map_label(self, task: TaskId, label: str) -> str | None:
        return self._label_aliases[task].get(_normalize_label_key(label))

    def _resolve_span(self, *, text: str, entity: dict[str, Any]) -> tuple[int, int, str] | None:
        start_raw = entity.get("start")
        end_raw = entity.get("end")
        if isinstance(start_raw, int) and isinstance(end_raw, int):
            start = max(0, start_raw)
            end = min(len(text), end_raw)
        else:
            start = -1
            end = -1

        entity_text = entity.get("text")
        span_text = entity_text if isinstance(entity_text, str) else ""
        if (start < 0 or end <= start) and span_text:
            idx = text.find(span_text)
            if idx != -1:
                start = idx
                end = idx + len(span_text)

        if start < 0 or end <= start:
            return None
        if not span_text:
            span_text = text[start:end]
        if not span_text:
            return None
        return start, end, span_text

    def _negation_for_entity(self, *, text: str, start: int, end: int) -> str:
        clause = _normalize_context(_extract_clause(text, start, end))
        prefix = _normalize_context(text[max(0, start - self._negation_window_chars) : start])
        suffix = _normalize_context(text[end : min(len(text), end + self._negation_window_chars)])
        scope = " ".join(value for value in (prefix, clause, suffix) if value)

        if any(cue in scope for cue in _ABSENT_CUES):
            return "Absent"
        if any(cue in scope for cue in _INDETERMINATE_CUES):
            return "Indetermine"
        return "Present"

    def _to_records(self, *, request: InferenceRequest, entities: list[dict[str, Any]]) -> list[Record]:
        records: list[Record] = []
        seen: set[tuple[str, str, int, int, tuple[tuple[str, str], ...]]] = set()
        for entity in entities:
            label_raw = entity.get("label")
            if not isinstance(label_raw, str):
                continue
            label = self._map_label(request.task, label_raw)
            if label is None:
                continue
            span = self._resolve_span(text=request.text, entity=entity)
            if span is None:
                continue
            start, end, span_text = span
            attributes: dict[str, Any] = {}
            if request.task == TaskId.INFECTIO:
                attributes["negation"] = self._negation_for_entity(
                    text=request.text,
                    start=start,
                    end=end,
                )
            dedupe_key = (
                label,
                span_text,
                start,
                end,
                tuple(sorted((key, str(value)) for key, value in attributes.items())),
            )
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            records.append(
                Record(
                    label=label,
                    text=span_text,
                    start=start,
                    end=end,
                    attributes=attributes,
                )
            )

        records.sort(
            key=lambda item: (
                item.start if item.start is not None else 10**9,
                item.end if item.end is not None else 10**9,
                item.label,
                item.text or "",
            )
        )
        return records

    def _extract_entities_payload(self, request: InferenceRequest, model: Any) -> list[dict[str, Any]]:
        schema = _TASK_LABEL_DESCRIPTIONS[request.task]
        kwargs: dict[str, Any] = {
            "include_spans": True,
            "include_confidence": True,
        }
        try:
            raw = model.extract_entities(
                request.text,
                schema,
                threshold=self._threshold,
                **kwargs,
            )
        except TypeError:
            raw = model.extract_entities(
                request.text,
                schema,
                **kwargs,
            )

        if not isinstance(raw, dict):
            raise RuntimeError("Unexpected GLiNER2 response: expected a JSON object.")
        entities_node = raw.get("entities")
        if not isinstance(entities_node, dict):
            raise RuntimeError("Unexpected GLiNER2 response: invalid `entities` field.")

        flattened: list[dict[str, Any]] = []
        for raw_label, values in entities_node.items():
            if not isinstance(raw_label, str):
                continue
            if not isinstance(values, list):
                continue
            for value in values:
                if isinstance(value, dict):
                    item = dict(value)
                    item["label"] = raw_label
                    flattened.append(item)
                elif isinstance(value, str):
                    flattened.append({"label": raw_label, "text": value})
        return flattened

    def _get_tokenizer(self) -> Any:
        """Load tokenizer lazily and fallback to word-based counting."""

        if self._tokenizer is None:
            try:
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(  # type: ignore[no-untyped-call]
                    self._model_reference,
                    revision=self._tokenizer_revision,
                    trust_remote_code=True,
                )
            except Exception as exc:
                _LOG.warning(
                    "Failed to load tokenizer for %s: %s. "
                    "Falling back to word-count approximation.",
                    self._model_reference,
                    exc,
                )
                self._tokenizer = _WordCountTokenizer()
        return self._tokenizer

    def _count_tokens(self, text: str) -> int:
        tok = self._get_tokenizer()
        return len(tok.encode(text, add_special_tokens=False))

    def infer(self, request: InferenceRequest) -> str:
        """Run single-pass inference, enabling chunking near context limit."""

        text_tokens = self._count_tokens(request.text)
        if text_tokens >= _CHUNK_THRESHOLD * self._max_context_tokens:
            _LOG.info(
                "Chunking enabled: text=%d tokens >= %.0f%% of %d "
                "(document_id=%s, task=%s)",
                text_tokens,
                _CHUNK_THRESHOLD * 100,
                self._max_context_tokens,
                request.document_id,
                request.task.value,
            )
            return self._chunk_infer(request)
        return self._infer_single(request)

    def _infer_single(self, request: InferenceRequest) -> str:
        if request.track != TrackId.ZEROSHOT:
            raise RuntimeError("Le runtime GLiNER2 supporte uniquement la track `zero-shot`.")

        model = self._ensure_model()
        parsed_entities = self._extract_entities_payload(request, model)
        records = self._to_records(request=request, entities=parsed_entities)

        payload: dict[str, Any] = {
            "document_id": request.document_id,
            "task": request.task.value,
            "records": [record.model_dump(mode="json") for record in records],
        }
        if request.task == TaskId.SCENARIO:
            speciality = request.gold.speciality if request.gold is not None else None
            payload["speciality"] = speciality or "unknown"

        return json.dumps(payload, ensure_ascii=False)

    def _chunk_infer(self, request: InferenceRequest) -> str:
        """Chunk source text, infer per chunk, and merge canonical outputs."""
        tokenizer = self._get_tokenizer()
        text_budget = int(_CHUNK_TARGET * self._max_context_tokens)
        chunks = make_chunks(request.text, tokenizer, text_budget)
        chunk_results: list[tuple[CanonicalDocument, int]] = []
        for chunk in chunks:
            sub = request.model_copy(update={"text": chunk.text})
            raw = self._infer_single(sub)
            try:
                doc = CanonicalDocument.model_validate(json.loads(raw))
                chunk_results.append((doc, chunk.start_char))
            except Exception as exc:
                _LOG.warning(
                    "Chunk skipped (parse error) for document_id=%s: %s",
                    request.document_id,
                    exc,
                )

        if not chunk_results:
            return ""

        merged = merge_canonical_documents(chunk_results)
        return json.dumps(merged.model_dump(mode="json"), ensure_ascii=False)

    def close(self) -> None:
        self._model = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            return None
