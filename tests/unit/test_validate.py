from parhaf_clinbench.core.enums import TaskId
from parhaf_clinbench.parsing.validate import validate_and_parse


def test_validate_schema_for_pseudo_requires_offsets() -> None:
    raw = (
        '{"document_id":"d1","task":"pseudo","speciality":null,"records":[{"label":"FIRST_NAME",'
        '"text":"abc","start":1,"end":3,"attributes":{}}]}'
    )
    parsed, raw_json_valid, repair_applied, schema_valid, error = validate_and_parse(raw, TaskId.PSEUDO)
    assert raw_json_valid is True
    assert repair_applied is False
    assert schema_valid is True
    assert error is None
    assert parsed is not None
    assert parsed.document_id == "d1"


def test_validate_schema_for_pseudo_rejects_unknown_label() -> None:
    raw = (
        '{"document_id":"d1","task":"pseudo","speciality":null,"records":[{"label":"UNKNOWN_LABEL",'
        '"text":"abc","start":1,"end":3,"attributes":{}}]}'
    )
    _parsed, raw_json_valid, repair_applied, schema_valid, error = validate_and_parse(raw, TaskId.PSEUDO)
    assert raw_json_valid is True
    assert repair_applied is False
    assert schema_valid is False
    assert error is not None
    assert "Label pseudo invalide" in error
