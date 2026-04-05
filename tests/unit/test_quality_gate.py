from __future__ import annotations

from dataclasses import dataclass

import pytest

from parhaf_clinbench.ops.quality_gate import run_local_quality_gate


@dataclass
class _Completed:
    returncode: int


def test_quality_gate_runs_expected_commands(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[list[str]] = []

    def fake_run(cmd: list[str], check: bool = False) -> _Completed:
        del check
        seen.append(cmd)
        return _Completed(returncode=0)

    monkeypatch.setattr("parhaf_clinbench.ops.quality_gate.subprocess.run", fake_run)
    run_local_quality_gate()

    assert seen == [
        ["uv", "run", "ruff", "check", "src", "tests"],
        ["uv", "run", "mypy"],
        ["uv", "run", "pytest"],
    ]


def test_quality_gate_raises_when_one_command_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    def fake_run(cmd: list[str], check: bool = False) -> _Completed:
        del check
        del cmd
        nonlocal calls
        calls += 1
        if calls == 2:
            return _Completed(returncode=1)
        return _Completed(returncode=0)

    monkeypatch.setattr("parhaf_clinbench.ops.quality_gate.subprocess.run", fake_run)

    with pytest.raises(RuntimeError, match="Gate locale échouée"):
        run_local_quality_gate()
