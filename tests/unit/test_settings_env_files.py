from __future__ import annotations

from parhaf_clinbench.core.settings import _resolve_env_files


def test_resolve_env_files_contains_root_and_infra_env() -> None:
    candidates = _resolve_env_files()
    assert any(path.endswith("/.env") for path in candidates)
    assert any(path.endswith("/infra/.env") for path in candidates)
