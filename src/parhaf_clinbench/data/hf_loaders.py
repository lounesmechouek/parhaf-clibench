"""Load benchmark datasets and convert rows to canonical examples."""

from __future__ import annotations

import importlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from parhaf_clinbench.core.enums import TaskId
from parhaf_clinbench.core.hashing import stable_sha256_text
from parhaf_clinbench.core.models import CanonicalDocument, DocumentExample, Record
from parhaf_clinbench.data.canonicalize import dict_to_canonical_document


def _pick_first_str(row: dict[str, Any], *keys: str) -> str | None:
    """Return the first non-empty string value for candidate keys."""

    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _pick_first_int(row: dict[str, Any], *keys: str) -> int | None:
    """Return the first integer value for candidate keys."""

    for key in keys:
        value = row.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, str) and value.strip().isdigit():
            return int(value.strip())
    return None


def _extract_document_id(row: dict[str, Any]) -> str | None:
    """Extract document identifier from known dataset key aliases."""

    return _pick_first_str(row, "document_id", "report", "report_id", "doc_id", "id")


def _extract_text(row: dict[str, Any]) -> str:
    """Extract source text from heterogeneous dataset payload structures."""

    full = _pick_first_str(row, "full_text", "text")
    if full is not None:
        return full
    documents = row.get("documents")
    if isinstance(documents, dict):
        text = documents.get("text")
        if isinstance(text, str):
            return text
        if isinstance(text, list):
            text_parts: list[str] = []
            for item in text:
                if isinstance(item, str):
                    text_parts.append(item)
                elif isinstance(item, dict):
                    chunk = item.get("text")
                    if isinstance(chunk, str):
                        text_parts.append(chunk)
            if text_parts:
                return "\n".join(text_parts)
    if isinstance(documents, list):
        parts: list[str] = []
        for item in documents:
            if isinstance(item, dict):
                chunk = item.get("text")
                if isinstance(chunk, str):
                    parts.append(chunk)
            elif isinstance(item, str):
                parts.append(item)
        if parts:
            return "\n".join(parts)
    return ""


def _iter_dataset_rows(
    dataset: Any,
    *,
    source_prefix: str | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    """Iterate all rows across dataset splits with split-prefixed source tags."""

    rows: list[tuple[str, dict[str, Any]]] = []
    for split_name in list(dataset.keys()):
        split = dataset[split_name]
        source = str(split_name)
        if source_prefix:
            source = f"{source_prefix}:{split_name}"
        for row in split:
            rows.append((source, dict(row)))
    return rows


def _iter_standard_splits(
    dataset: Any,
    *,
    source_prefix: str | None = None,
) -> list[tuple[str, dict[str, Any]]]:
    """Iterate canonical train/dev/validation/test splits when present."""

    rows: list[tuple[str, dict[str, Any]]] = []
    for split_name in ("train", "dev", "validation", "test"):
        if split_name in dataset:
            source = split_name
            if source_prefix:
                source = f"{source_prefix}:{split_name}"
            for row in dataset[split_name]:
                rows.append((source, dict(row)))
    return rows


def _has_local_cache_materialized(cache_dir: Path | None) -> bool:
    """Return whether a cache directory contains at least one file."""

    if cache_dir is None or not cache_dir.exists():
        return False
    return any(path.is_file() for path in cache_dir.rglob("*"))


def _load_dataset_with_cache_fallback(
    *,
    load_dataset_fn: Any,
    dataset_name: str,
    config_name: str | None,
    revision: str,
    cache_dir: Path | None,
) -> Any:
    """Load dataset from cache first, then fallback to standard network loading."""

    def _make_local_only_download_config() -> Any | None:
        """Build a `datasets.DownloadConfig` instance when available."""

        for module_name in ("datasets", "datasets.download.download_config"):
            try:
                module = importlib.import_module(module_name)
            except ImportError:
                continue

            download_config_cls = getattr(module, "DownloadConfig", None)
            if download_config_cls is None:
                continue

            return download_config_cls(local_files_only=True, max_retries=0)

        return None

    cache_dir_str = str(cache_dir) if cache_dir is not None else None
    if _has_local_cache_materialized(cache_dir):
        try:
            download_config = _make_local_only_download_config()
            if config_name is None:
                return load_dataset_fn(
                    dataset_name,
                    revision=revision,
                    cache_dir=cache_dir_str,
                    download_config=download_config,
                )
            return load_dataset_fn(
                dataset_name,
                config_name,
                revision=revision,
                cache_dir=cache_dir_str,
                download_config=download_config,
            )
        except Exception:
            # NOTE: Corrupted or partial cache: fallback to regular loading.
            pass

    if config_name is None:
        return load_dataset_fn(
            dataset_name,
            revision=revision,
            cache_dir=cache_dir_str,
        )
    return load_dataset_fn(
        dataset_name,
        config_name,
        revision=revision,
        cache_dir=cache_dir_str,
    )


def _extract_speciality(row: dict[str, Any]) -> str | None:
    """Extract speciality from direct fields or nested scenario payload."""

    speciality = _pick_first_str(row, "speciality", "specialty")
    if speciality is not None:
        return speciality
    scenario = row.get("suggested_scenario")
    if isinstance(scenario, dict):
        value = scenario.get("speciality")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _align_span(text: str, span_text: str) -> tuple[int | None, int | None]:
    """Find first occurrence of `span_text` in `text` and return offsets."""

    if not span_text:
        return None, None
    idx = text.find(span_text)
    if idx < 0:
        return None, None
    return idx, idx + len(span_text)


def _scenario_value_to_text(value: Any) -> str | None:
    """Convert heterogeneous scenario values into a normalized text string."""

    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        parts = [_scenario_value_to_text(item) for item in value]
        filtered = [part for part in parts if part]
        if not filtered:
            return None
        return ", ".join(filtered)
    if isinstance(value, dict):
        description = _scenario_value_to_text(value.get("description"))
        if description:
            return description
        raw_value = _scenario_value_to_text(value.get("value"))
        if raw_value:
            unit = _scenario_value_to_text(value.get("unit"))
            if unit:
                return f"{raw_value} {unit}".strip()
            return raw_value
        code = _scenario_value_to_text(value.get("code"))
        if code:
            return code
        for item in value.values():
            maybe = _scenario_value_to_text(item)
            if maybe:
                return maybe
        return None
    text = str(value).strip()
    return text or None


def _canonical_from_records(
    *,
    task: TaskId,
    document_id: str,
    speciality: str | None,
    records: list[Record],
) -> CanonicalDocument:
    """Build a canonical document from normalized records."""

    return CanonicalDocument(
        document_id=document_id,
        task=task,
        speciality=speciality,
        records=records,
    )


def _load_hf_pseudo(rows: list[tuple[str, dict[str, Any]]]) -> list[DocumentExample]:
    """Convert pseudo rows into benchmark `DocumentExample` objects."""

    by_doc: dict[str, dict[str, Any]] = defaultdict(lambda: {"text": "", "speciality": None, "records": []})
    for _split, row in rows:
        document_id = _extract_document_id(row)
        if document_id is None:
            continue
        text = _extract_text(row)
        if text:
            by_doc[document_id]["text"] = text
        speciality = _extract_speciality(row)
        if speciality is not None:
            by_doc[document_id]["speciality"] = speciality

        label = _pick_first_str(row, "attribute_Categorie", "span_type")
        span_text = _pick_first_str(row, "span_text")
        if label is None or span_text is None:
            continue
        start = _pick_first_int(row, "begin", "start")
        end = _pick_first_int(row, "end")
        if start is None or end is None:
            start, end = _align_span(by_doc[document_id]["text"], span_text)
        if start is None or end is None:
            continue
        attributes: dict[str, Any] = {}
        role = _pick_first_str(row, "attribute_RolePER")
        if role is not None:
            attributes["role"] = role
        by_doc[document_id]["records"].append(
            Record(label=label, text=span_text, start=start, end=end, attributes=attributes)
        )

    examples: list[DocumentExample] = []
    for document_id, item in by_doc.items():
        text = str(item["text"])
        if not text:
            continue
        gold = _canonical_from_records(
            task=TaskId.PSEUDO,
            document_id=document_id,
            speciality=item["speciality"],
            records=list(item["records"]),
        )
        examples.append(
            DocumentExample(
                document_id=document_id,
                task=TaskId.PSEUDO,
                speciality=item["speciality"],
                text=text,
                gold=gold,
            )
        )
    return examples


def _load_hf_infectio(rows: list[tuple[str, dict[str, Any]]]) -> list[DocumentExample]:
    """Convert infectio rows into benchmark `DocumentExample` objects."""

    by_doc: dict[str, dict[str, Any]] = defaultdict(lambda: {"text": "", "speciality": None, "records": []})
    for _split, row in rows:
        document_id = _extract_document_id(row)
        if document_id is None:
            continue
        text = _extract_text(row)
        if text:
            by_doc[document_id]["text"] = text
        speciality = _extract_speciality(row)
        if speciality is not None:
            by_doc[document_id]["speciality"] = speciality

        label = _pick_first_str(row, "attribute_LABEL", "label")
        span_text = _pick_first_str(row, "span_text")
        if label is None or span_text is None:
            continue
        start = _pick_first_int(row, "begin", "start")
        end = _pick_first_int(row, "end")
        if start is None or end is None:
            start, end = _align_span(by_doc[document_id]["text"], span_text)
        negation = _pick_first_str(row, "attribute_NEGATION")
        attributes: dict[str, Any] = {}
        if negation is not None:
            attributes["negation"] = negation
        by_doc[document_id]["records"].append(
            Record(label=label, text=span_text, start=start, end=end, attributes=attributes)
        )

    examples: list[DocumentExample] = []
    for document_id, item in by_doc.items():
        text = str(item["text"])
        if not text:
            continue
        gold = _canonical_from_records(
            task=TaskId.INFECTIO,
            document_id=document_id,
            speciality=item["speciality"],
            records=list(item["records"]),
        )
        examples.append(
            DocumentExample(
                document_id=document_id,
                task=TaskId.INFECTIO,
                speciality=item["speciality"],
                text=text,
                gold=gold,
            )
        )
    return examples


def _load_hf_response(rows: list[tuple[str, dict[str, Any]]]) -> list[DocumentExample]:
    """Convert response rows into benchmark `DocumentExample` objects."""

    by_doc: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"text": "", "speciality": None, "records": [], "doc_label": None}
    )

    for split, row in rows:
        document_id = _extract_document_id(row)
        if document_id is None:
            continue
        text = _extract_text(row)
        if text:
            by_doc[document_id]["text"] = text
        speciality = _extract_speciality(row)
        if speciality is not None:
            by_doc[document_id]["speciality"] = speciality

        if split.startswith("document_metadata:"):
            doc_label = _pick_first_str(row, "attribute_Nomenclature", "label")
            if doc_label is not None:
                by_doc[document_id]["doc_label"] = doc_label

        span_text = _pick_first_str(row, "span_text", "attribute_Justification")
        label = _pick_first_str(row, "attribute_Nomenclature", "label")
        if span_text is None:
            continue
        if label is None:
            label = by_doc[document_id]["doc_label"]
        if label is None:
            continue
        start = _pick_first_int(row, "begin", "start")
        end = _pick_first_int(row, "end")
        if start is None or end is None:
            start, end = _align_span(by_doc[document_id]["text"], span_text)
        by_doc[document_id]["records"].append(
            Record(label=label, text=span_text, start=start, end=end, attributes={})
        )

    examples: list[DocumentExample] = []
    for document_id, item in by_doc.items():
        text = str(item["text"])
        if not text:
            continue
        records: list[Record] = list(item["records"])
        doc_label = item["doc_label"]
        if not records and isinstance(doc_label, str) and doc_label.strip():
            # NOTE: Keep document-level class even without a justification span.
            # NOTE: Official `(text,label)` scoring ignores this empty-text record.
            records.append(Record(label=doc_label, text="", start=None, end=None, attributes={}))
        gold = _canonical_from_records(
            task=TaskId.RESPONSE,
            document_id=document_id,
            speciality=item["speciality"],
            records=records,
        )
        examples.append(
            DocumentExample(
                document_id=document_id,
                task=TaskId.RESPONSE,
                speciality=item["speciality"],
                text=text,
                gold=gold,
            )
        )
    return examples


def _load_hf_scenario(rows: list[tuple[str, dict[str, Any]]]) -> list[DocumentExample]:
    """Convert scenario rows into benchmark `DocumentExample` objects."""

    scenario_fields = [
        "name",
        "age",
        "sex",
        "admission_mode",
        "discharge_mode",
        "primary_procedure",
        "primary_diagnosis",
        "type_of_care",
    ]
    examples: list[DocumentExample] = []
    for _split, row in rows:
        document_id = _extract_document_id(row)
        if document_id is None:
            continue
        text = _extract_text(row)
        if not text:
            continue
        suggested = row.get("suggested_scenario")
        if not isinstance(suggested, dict):
            continue
        speciality = _extract_speciality(row)
        if speciality is None:
            continue
        records: list[Record] = []
        for label in scenario_fields:
            value = suggested.get(label)
            text_value = _scenario_value_to_text(value)
            if text_value is None:
                continue
            start, end = _align_span(text, text_value)
            if start is None or end is None:
                # NOTE: v1 methodology keeps only values verbatim-located in source text.
                continue
            records.append(
                Record(
                    label=label,
                    text=text_value,
                    start=start,
                    end=end,
                    attributes={},
                )
            )

        gold = _canonical_from_records(
            task=TaskId.SCENARIO,
            document_id=document_id,
            speciality=speciality,
            records=records,
        )
        examples.append(
            DocumentExample(
                document_id=document_id,
                task=TaskId.SCENARIO,
                speciality=speciality,
                text=text,
                gold=gold,
            )
        )
    return examples


def load_smoke_examples(path: Path, task: TaskId | None = None) -> list[DocumentExample]:
    """Load smoke examples from a local JSONL file.

    Args:
        path: Path to a JSONL file containing examples.
        task: Optional task filter.

    Returns:
        List of canonicalized document examples.

    Examples:
        >>> examples = load_smoke_examples(Path("data/smoke/examples.jsonl"))
        >>> isinstance(examples, list)
        True
    """

    examples: list[DocumentExample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            row_task = TaskId(row["task"])
            if task is not None and row_task != task:
                continue
            gold = dict_to_canonical_document(dict(row["gold"]))
            examples.append(
                DocumentExample(
                    document_id=str(row["document_id"]),
                    task=row_task,
                    speciality=row.get("speciality"),
                    text=str(row["text"]),
                    gold=gold,
                )
            )
    return examples


def load_hf_examples(
    task: TaskId,
    dataset_name: str,
    dataset_revision: str = "main",
    cache_dir: Path | None = None,
) -> list[DocumentExample]:
    """Load a Hugging Face dataset and build task-specific canonical gold.

    Args:
        task: Target benchmark task.
        dataset_name: Hugging Face dataset identifier.
        dataset_revision: Dataset revision/branch/tag.
        cache_dir: Optional cache directory.

    Returns:
        Canonical document examples ready for inference/scoring.

    Examples:
        >>> examples = load_hf_examples(TaskId.PSEUDO, "my-org/my-dataset")
        >>> isinstance(examples, list)
        True
    """

    try:
        datasets_module = importlib.import_module("datasets")
        load_dataset = datasets_module.load_dataset
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "The `datasets` package is required to load Hugging Face datasets. "
            "Install it or use a local smoke set."
        ) from exc

    if task == TaskId.SCENARIO:
        dataset = _load_dataset_with_cache_fallback(
            load_dataset_fn=load_dataset,
            dataset_name=dataset_name,
            config_name=None,
            revision=dataset_revision,
            cache_dir=cache_dir,
        )
        rows = _iter_standard_splits(dataset)
    else:
        metadata_dataset = _load_dataset_with_cache_fallback(
            load_dataset_fn=load_dataset,
            dataset_name=dataset_name,
            config_name="document_metadata",
            revision=dataset_revision,
            cache_dir=cache_dir,
        )
        spans_dataset = _load_dataset_with_cache_fallback(
            load_dataset_fn=load_dataset,
            dataset_name=dataset_name,
            config_name="spans",
            revision=dataset_revision,
            cache_dir=cache_dir,
        )
        rows = _iter_dataset_rows(metadata_dataset, source_prefix="document_metadata")
        rows.extend(_iter_dataset_rows(spans_dataset, source_prefix="spans"))
    if task == TaskId.PSEUDO:
        return _load_hf_pseudo(rows)
    if task == TaskId.INFECTIO:
        return _load_hf_infectio(rows)
    if task == TaskId.RESPONSE:
        return _load_hf_response(rows)
    return _load_hf_scenario(rows)


def examples_fingerprint(*, dataset_name: str, examples: list[DocumentExample]) -> str:
    """Compute a deterministic fingerprint for loaded examples.

    Args:
        dataset_name: Dataset identifier included in the fingerprint.
        examples: Canonicalized document examples.

    Returns:
        Stable SHA-256 fingerprint.

    Examples:
        >>> fingerprint = examples_fingerprint(dataset_name="demo", examples=[])
        >>> len(fingerprint)
        64
    """

    entries: list[str] = [dataset_name, str(len(examples))]
    for example in examples:
        entries.append(example.document_id)
        entries.append(stable_sha256_text(example.text))
        entries.append(stable_sha256_text(example.gold.model_dump_json()))
    return stable_sha256_text("|".join(entries))
