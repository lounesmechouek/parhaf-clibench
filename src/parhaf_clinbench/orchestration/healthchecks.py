"""Runtime healthcheck utilities."""

from __future__ import annotations

from parhaf_clinbench.core.enums import RuntimeName
from parhaf_clinbench.orchestration.experiment_plan import RuntimeConfig


def run_healthcheck(runtime_name: RuntimeName, runtime_cfg: RuntimeConfig) -> None:
    """Run a best-effort runtime healthcheck.

    Args:
        runtime_name: Runtime identifier.
        runtime_cfg: Runtime configuration payload.
    """

    if runtime_name != RuntimeName.VLLM:
        return
    try:
        import requests
    except Exception:
        return
    health_url = runtime_cfg.payload.get("healthcheck_url")
    if not isinstance(health_url, str) or not health_url:
        return
    response = requests.get(health_url, timeout=5)
    response.raise_for_status()
