from __future__ import annotations

import json
from pathlib import Path

from parhaf_clinbench.core.models import (
    INFECTIO_LABELS,
    INFECTIO_NEGATIONS,
    PSEUDO_LABELS,
    RESPONSE_LABELS,
    SCENARIO_FIELDS,
    SCENARIO_SPECIALITIES,
)


def test_contract_snapshot_matches_runtime_constraints() -> None:
    snapshot_path = Path("configs/contracts/hf_contract_snapshot_v1.json")
    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))

    assert set(payload["pseudo"]["labels"]) == set(PSEUDO_LABELS)
    assert set(payload["infectio"]["labels"]) == set(INFECTIO_LABELS)
    assert set(payload["infectio"]["attribute_NEGATION_values"]) == set(INFECTIO_NEGATIONS)

    assert set(payload["response"]["labels_union"]) == set(RESPONSE_LABELS)

    assert set(payload["scenario"]["suggested_scenario_fields"]) == set(SCENARIO_FIELDS)
    assert set(payload["scenario"]["specialities"]) == set(SCENARIO_SPECIALITIES)
