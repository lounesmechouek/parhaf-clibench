"""Stop or terminate an existing RunPod pod."""

from __future__ import annotations

import argparse
import json

from parhaf_clinbench.core.settings import get_settings
from parhaf_clinbench.ops.runpod_client import RunpodClient


def stop_or_terminate() -> None:
    """Stop or terminate a pod according to CLI flags.

    Examples:
        parhaf-stop-runpod --pod-id pod-123 --terminate
    """

    settings = get_settings()
    parser = argparse.ArgumentParser(prog="parhaf-stop-runpod")
    parser.add_argument("--pod-id", default=settings.runpod_pod_id)
    parser.add_argument("--terminate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.pod_id:
        raise ValueError("Pod ID required (--pod-id or RUNPOD_POD_ID).")
    if args.dry_run:
        action = "terminate" if args.terminate else "stop"
        print(json.dumps({"action": action, "pod_id": args.pod_id}, ensure_ascii=False, indent=2))
        return
    if settings.runpod_api_key is None:
        raise ValueError("RUNPOD_API_KEY is required.")

    client = RunpodClient(api_base=settings.runpod_api_base, api_key=settings.runpod_api_key)
    payload = client.terminate_pod(args.pod_id) if args.terminate else client.stop_pod(args.pod_id)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main() -> None:
    """CLI entrypoint for pod stop/terminate operations."""

    stop_or_terminate()


if __name__ == "__main__":
    main()
