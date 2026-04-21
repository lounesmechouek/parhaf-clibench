"""JSON parsing and canonical-schema validation with Pydantic."""

from __future__ import annotations

import json

from pydantic import ValidationError

from parhaf_clinbench.core.enums import TaskId
from parhaf_clinbench.core.models import CanonicalDocument
from parhaf_clinbench.parsing.repair import extract_json_block


def _validate_expected_task(
    document: CanonicalDocument,
    expected_task: TaskId,
) -> tuple[bool, str | None]:
    """Ensure the parsed document task matches the expected task."""

    if document.task != expected_task:
        return False, "The `task` field does not match the expected task."
    return True, None


def validate_and_parse(
    raw_output: str,
    expected_task: TaskId,
) -> tuple[CanonicalDocument | None, bool, bool, bool, str | None]:
    """Parse and validate a runtime output against canonical schema.

    Args:
        raw_output: Raw text produced by a runtime.
        expected_task: Task that the output must target.

    Returns:
        Tuple in the form:
        `(parsed, raw_json_valid, repair_applied, schema_valid, error)`.

    Examples:
        >>> payload = '{"document_id":"d1","task":"pseudo","records":[]}'
        >>> parsed, raw_ok, repaired, schema_ok, error = validate_and_parse(payload, TaskId.PSEUDO)
        >>> bool(parsed), raw_ok, repaired, schema_ok, error is None
        (True, True, False, True, True)
    """

    candidate = raw_output
    raw_json_valid = True
    repair_applied = False

    try:
        json.loads(candidate)
    except json.JSONDecodeError:
        raw_json_valid = False
        candidate = extract_json_block(raw_output)
        repair_applied = candidate != raw_output
        try:
            json.loads(candidate)
        except json.JSONDecodeError as exc:
            return None, raw_json_valid, repair_applied, False, f"Invalid JSON: {exc}"

    try:
        parsed = CanonicalDocument.model_validate_json(candidate)
    except ValidationError as exc:
        message = "; ".join(
            f"{'.'.join(str(part) for part in err.get('loc', []))}: {err.get('msg', 'invalid')}"
            for err in exc.errors()
        )
        return None, raw_json_valid, repair_applied, False, f"Invalid schema: {message}"

    is_ok, error = _validate_expected_task(parsed, expected_task)
    if not is_ok:
        return None, raw_json_valid, repair_applied, False, error
    return parsed, raw_json_valid, repair_applied, True, None
