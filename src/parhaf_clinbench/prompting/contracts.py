"""Dynamic prompt context injected into Jinja templates."""

from __future__ import annotations

from typing import Any

from parhaf_clinbench.core.enums import TaskId
from parhaf_clinbench.core.models import (
    INFECTIO_LABELS,
    INFECTIO_NEGATIONS,
    PSEUDO_LABELS,
    RESPONSE_LABELS,
    SCENARIO_FIELDS,
    SCENARIO_SPECIALITIES,
)

_CANONICAL_SCHEMA = """{
  "document_id": "string",
  "task": "pseudo | infectio | response | scenario",
  "speciality": "string | null",
  "records": [
    {
      "label": "string",
      "text": "string | null",
      "start": "integer | null",
      "end": "integer | null",
      "attributes": {}
    }
  ]
}"""

_INFECTIO_LABEL_DEFINITIONS = {
    "Bacterie": "nom d'une bactérie ou d'un agent bactérien explicitement mentionné.",
    "Bacteriemie": "mention explicite d'une bactériémie.",
    "Infection": "mention explicite d'une infection ou d'un diagnostic infectieux.",
    "Site": "localisation anatomique ou site infectieux explicitement mentionné.",
}

_INFECTIO_NEGATION_DEFINITIONS = {
    "Present": "l'entité ou l'événement infectieux est affirmé.",
    "Absent": "l'entité ou l'événement infectieux est explicitement nié ou exclu.",
    "Indetermine": "l'information est suspectée, possible, incertaine ou non tranchée.",
}

_SCENARIO_FIELDS_PREFERRED_ORDER = [
    "name",
    "age",
    "sex",
    "admission_mode",
    "discharge_mode",
    "primary_procedure",
    "primary_diagnosis",
    "type_of_care",
]


def _ordered_with_preference(values: set[str], preferred_order: list[str]) -> list[str]:
    """Return `values` ordered by preferred items, then alphabetical remainder."""

    preferred = [item for item in preferred_order if item in values]
    remainder = sorted(item for item in values if item not in set(preferred_order))
    return preferred + remainder


def prompt_dynamic_context(*, task: TaskId, speciality_metadata: str | None = None) -> dict[str, Any]:
    """Build dynamic context consumed by prompt templates.

    Args:
        task: Benchmark task.
        speciality_metadata: Optional metadata block injected as-is.

    Returns:
        Template context dictionary with schema and task-specific definitions.

    Examples:
        >>> ctx = prompt_dynamic_context(task=TaskId.PSEUDO)
        >>> "canonical_schema_json" in ctx
        True
    """

    context: dict[str, Any] = {
        "canonical_schema_json": _CANONICAL_SCHEMA,
        "offset_policy": "start=offset caractère 0-based, end=offset caractère exclusif.",
        "speciality_metadata": speciality_metadata or "",
    }

    if task == TaskId.PSEUDO:
        context["pseudo_labels"] = sorted(PSEUDO_LABELS)
        return context

    if task == TaskId.INFECTIO:
        infectio_labels = sorted(INFECTIO_LABELS)
        infectio_negations = sorted(INFECTIO_NEGATIONS)
        context["infectio_labels"] = infectio_labels
        context["infectio_negations"] = infectio_negations
        context["infectio_label_definitions"] = [
            {
                "name": label,
                "definition": _INFECTIO_LABEL_DEFINITIONS.get(label, ""),
            }
            for label in infectio_labels
        ]
        context["infectio_negation_definitions"] = [
            {
                "name": negation,
                "definition": _INFECTIO_NEGATION_DEFINITIONS.get(negation, ""),
            }
            for negation in infectio_negations
        ]
        return context

    if task == TaskId.RESPONSE:
        context["response_labels"] = sorted(RESPONSE_LABELS)
        return context

    context["scenario_fields"] = _ordered_with_preference(SCENARIO_FIELDS, _SCENARIO_FIELDS_PREFERRED_ORDER)
    context["scenario_specialities"] = sorted(SCENARIO_SPECIALITIES)
    return context
