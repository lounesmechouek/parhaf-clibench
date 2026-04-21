from __future__ import annotations

import re
from pathlib import Path


def test_pseudo_fewshot_roles_match_dataset_contract() -> None:
    content = Path("assets/fewshot/pseudo_examples.txt").read_text(encoding="utf-8")
    roles = set(re.findall(r'"role"\s*:\s*"([^"]+)"', content))
    assert roles <= {"Patient", "Carer", "Other"}
