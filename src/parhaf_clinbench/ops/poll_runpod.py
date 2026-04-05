"""Poll RunPod status until a target state is reached."""

from __future__ import annotations

import argparse
import json

from parhaf_clinbench.core.settings import get_settings
from parhaf_clinbench.ops.runpod_client import RunpodClient


def poll() -> None:
    """Run pod status polling from CLI arguments.

    Examples:
        parhaf-poll-runpod --pod-id pod-123 --target-status RUNNING
    """

    settings = get_settings()
    parser = argparse.ArgumentParser(prog="parhaf-poll-runpod")
    parser.add_argument("--pod-id", default=settings.runpod_pod_id)
    parser.add_argument("--target-status", default="RUNNING")
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--poll-interval-seconds", type=int, default=10)
    args = parser.parse_args()

    if not args.pod_id:
        raise ValueError("Pod ID requis (--pod-id ou RUNPOD_POD_ID).")
    if settings.runpod_api_key is None:
        raise ValueError("RUNPOD_API_KEY est requis.")

    client = RunpodClient(api_base=settings.runpod_api_base, api_key=settings.runpod_api_key)
    payload = client.wait_pod(
        pod_id=args.pod_id,
        target_status=args.target_status,
        timeout_seconds=args.timeout_seconds,
        poll_interval_seconds=args.poll_interval_seconds,
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    """CLI entrypoint for pod polling."""

    poll()


if __name__ == "__main__":
    main()
