"""Start an existing RunPod pod through the API."""

from __future__ import annotations

import argparse
import json

from parhaf_clinbench.core.settings import get_settings
from parhaf_clinbench.ops.quality_gate import run_local_quality_gate
from parhaf_clinbench.ops.runpod_client import RunpodClient

_ACTIVE_POD_STATUSES = {"RUNNING", "STARTING", "PENDING", "RESUMING"}


def launch() -> None:
    """Start an already-created RunPod pod.

    Examples:
        parhaf-launch-runpod --pod-id pod-123
    """

    settings = get_settings()
    parser = argparse.ArgumentParser(prog="parhaf-launch-runpod")
    parser.add_argument("--pod-id", default=settings.runpod_pod_id)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-gate", action="store_true")
    args = parser.parse_args()

    if not args.pod_id:
        raise ValueError("Pod ID requis (--pod-id ou RUNPOD_POD_ID).")

    if not args.skip_gate:
        run_local_quality_gate()

    if args.dry_run:
        payload = {
            "action": "start_existing_pod",
            "api_base": settings.runpod_api_base,
            "endpoint": f"{settings.runpod_api_base.rstrip('/')}/pods/{args.pod_id}/start",
            "pod_id": args.pod_id,
            "env_hints": {
                "PARHAF_SUITE": settings.parhaf_suite,
                "PARHAF_OUTPUT_DIR": settings.parhaf_output_dir,
                "MODEL_CACHE_ROOT": str(settings.model_cache_root),
                "DATASET_CACHE_ROOT": str(settings.dataset_cache_root),
                "FINAL_ARCHIVE_PATH": str(settings.final_archive_path),
                "EXPORT_DIR": str(settings.export_dir),
            },
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if settings.runpod_api_key is None:
        raise ValueError("RUNPOD_API_KEY est requis pour lancer un pod.")

    client = RunpodClient(api_base=settings.runpod_api_base, api_key=settings.runpod_api_key)
    current = client.get_pod(args.pod_id)
    current_status = str(current.get("status", "")).upper()
    if current_status in _ACTIVE_POD_STATUSES:
        print(
            json.dumps(
                {
                    "id": args.pod_id,
                    "status": current_status,
                    "action": "already_active_no_start_called",
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    try:
        response = client.start_pod(args.pod_id)
    except RuntimeError as exc:
        after_status = "UNKNOWN"
        try:
            after = client.get_pod(args.pod_id)
            after_status = str(after.get("status", "")).upper()
        except Exception:
            pass
        raise RuntimeError(
            "RunPod pod start failed. "
            f"pod_id={args.pod_id} status_before={current_status} status_after={after_status}. "
            f"API detail: {exc}"
        ) from exc
    print(json.dumps(response, ensure_ascii=False, indent=2))


def main() -> None:
    """CLI entrypoint for pod start."""

    launch()


if __name__ == "__main__":
    main()
