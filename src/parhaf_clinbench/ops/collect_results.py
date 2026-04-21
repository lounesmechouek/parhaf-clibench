"""Collect benchmark artifacts from a remote RunPod instance."""

from __future__ import annotations

import argparse
import base64
import shlex
import subprocess
import sys
import time
from pathlib import Path

from parhaf_clinbench.core.settings import get_settings


def _build_rsync_cmd(
    *,
    ssh_target: str,
    remote_path: str,
    destination: Path,
    ssh_port: int | None,
    identity_file: str | None,
) -> list[str]:
    """Build an `rsync` command with optional SSH transport parameters."""

    ssh_transport_parts = ["ssh"]
    if ssh_port is not None:
        ssh_transport_parts.extend(["-p", str(ssh_port)])
    if identity_file:
        ssh_transport_parts.extend(["-i", identity_file])
    ssh_transport = " ".join(shlex.quote(part) for part in ssh_transport_parts)
    return [
        "rsync",
        "-avz",
        "-e",
        ssh_transport,
        f"{ssh_target}:{remote_path}",
        str(destination),
    ]


def _build_ssh_cmd(
    *,
    ssh_target: str,
    ssh_port: int | None,
    identity_file: str | None,
    remote_cmd: str,
    force_pty: bool = False,
) -> list[str]:
    """Build an `ssh` command with connectivity hardening flags."""

    cmd = ["ssh"]
    cmd.extend(
        [
            "-o",
            "BatchMode=yes",
            "-o",
            "StrictHostKeyChecking=accept-new",
            "-o",
            "ConnectTimeout=10",
            "-o",
            "ServerAliveInterval=5",
            "-o",
            "ServerAliveCountMax=2",
        ]
    )
    if force_pty:
        cmd.append("-tt")
    if ssh_port is not None:
        cmd.extend(["-p", str(ssh_port)])
    if identity_file:
        cmd.extend(["-i", identity_file])
    cmd.extend([ssh_target, remote_cmd])
    return cmd


def _validate_ssh_target(ssh_target: str) -> None:
    """Validate SSH target format."""

    if ssh_target.count("@") > 1:
        raise ValueError(
            "Format --ssh-target invalide. Utilise `user@host` (un seul `@`). "
            "Exemple RunPod TCP direct: `root@<PUBLIC_IP>`."
        )


def _validate_transport_for_target(ssh_target: str, transport: str) -> None:
    """Validate transfer transport compatibility for a given SSH target."""

    if ssh_target.endswith("@ssh.runpod.io") and transport == "ssh-cat":
        raise ValueError(
            "Le transport `ssh-cat` n'est pas compatible avec le proxy RunPod `*.ssh.runpod.io` "
            "(PTY requis). Utilise `--transport auto` ou `--transport ssh-pty-base64`."
        )


def _copy_via_ssh_cat(
    *,
    ssh_target: str,
    ssh_port: int | None,
    identity_file: str | None,
    remote_path: str,
    destination: Path,
    timeout_seconds: int,
) -> int:
    """Copy a remote file using direct `ssh ... cat` streaming."""

    remote_cmd = f"exec cat {shlex.quote(remote_path)}"
    cmd = _build_ssh_cmd(
        ssh_target=ssh_target,
        ssh_port=ssh_port,
        identity_file=identity_file,
        remote_cmd=remote_cmd,
    )
    try:
        with destination.open("wb") as handle:
            result = subprocess.run(cmd, check=False, stdout=handle, timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        destination.unlink(missing_ok=True)
        return 124
    if result.returncode != 0 and destination.exists():
        destination.unlink(missing_ok=True)
    return result.returncode


def _copy_via_ssh_pty_base64(
    *,
    ssh_target: str,
    ssh_port: int | None,
    identity_file: str | None,
    remote_path: str,
    destination: Path,
    timeout_seconds: int,
) -> int:
    """Copy a remote file via PTY-safe base64 transfer."""

    marker_begin = "__PARHAF_BEGIN__"
    marker_end = "__PARHAF_END__"
    remote_cmd = (
        "sh -lc '"
        f"printf {shlex.quote(marker_begin)}; "
        f"base64 -w0 {shlex.quote(remote_path)}; "
        f"printf {shlex.quote(marker_end)}"
        "'"
    )
    cmd = _build_ssh_cmd(
        ssh_target=ssh_target,
        ssh_port=ssh_port,
        identity_file=identity_file,
        remote_cmd=remote_cmd,
        force_pty=True,
    )
    try:
        result = subprocess.run(cmd, check=False, capture_output=True, timeout=timeout_seconds)
    except subprocess.TimeoutExpired:
        destination.unlink(missing_ok=True)
        return 124
    if result.returncode != 0:
        destination.unlink(missing_ok=True)
        return result.returncode
    output = result.stdout.decode("utf-8", errors="ignore")
    begin_idx = output.find(marker_begin)
    end_idx = output.rfind(marker_end)
    if begin_idx < 0 or end_idx <= begin_idx:
        destination.unlink(missing_ok=True)
        return 98
    encoded = output[begin_idx + len(marker_begin) : end_idx]
    encoded = "".join(encoded.split())
    try:
        decoded = base64.b64decode(encoded, validate=True)
    except Exception:
        destination.unlink(missing_ok=True)
        return 97
    destination.write_bytes(decoded)
    return 0


def _is_valid_downloaded_file(path: Path, remote_path: str) -> bool:
    """Validate downloaded file presence and optional `.zst` magic header."""

    if not path.exists() or path.stat().st_size <= 0:
        return False
    if remote_path.endswith(".zst"):
        header = path.read_bytes()[:4]
        return header == b"\x28\xb5\x2f\xfd"
    return True


def collect() -> None:
    """Download `results.tar.zst` from a remote host with retries/fallbacks.

    Examples:
        parhaf-collect-results --ssh-target root@1.2.3.4 --ssh-port 22
    """

    settings = get_settings()
    parser = argparse.ArgumentParser(prog="parhaf-collect-results")
    parser.add_argument("--ssh-target", required=True, help="Ex: root@1.2.3.4")
    parser.add_argument(
        "--ssh-port",
        type=int,
        default=None,
        help="Exposed SSH port (required for RunPod in TCP direct mode).",
    )
    parser.add_argument(
        "--identity-file",
        default="",
        help="Path to SSH private key (e.g. ~/.ssh/id_ed25519).",
    )
    parser.add_argument("--remote-path", default=str(settings.final_archive_path))
    parser.add_argument("--local-dir", default="results/downloads")
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=50000,  # NOTE: ~13.9h to cover long-running benchmark sessions.
    )
    parser.add_argument(
        "--transport",
        choices=["auto", "rsync", "ssh-cat", "ssh-pty-base64"],
        default="auto",
        help="Transfer method. `auto` picks strategy based on the SSH target.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    _validate_ssh_target(args.ssh_target)
    _validate_transport_for_target(args.ssh_target, args.transport)
    local_dir = Path(args.local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    destination = local_dir / "results.tar.zst"
    rsync_cmd = _build_rsync_cmd(
        ssh_target=args.ssh_target,
        remote_path=args.remote_path,
        destination=destination,
        ssh_port=args.ssh_port,
        identity_file=args.identity_file or None,
    )
    ssh_cat_cmd = _build_ssh_cmd(
        ssh_target=args.ssh_target,
        ssh_port=args.ssh_port,
        identity_file=args.identity_file or None,
        remote_cmd=f"exec cat {shlex.quote(args.remote_path)}",
    )

    if args.transport == "auto":
        if args.ssh_target.endswith("@ssh.runpod.io"):
            plan = ["ssh-pty-base64"]
        else:
            plan = ["rsync", "ssh-cat", "ssh-pty-base64"]
    else:
        plan = [args.transport]

    if args.dry_run:
        if "rsync" in plan:
            print("RSYNC:", " ".join(rsync_cmd))
        if "ssh-cat" in plan:
            print("SSH-CAT:", " ".join(ssh_cat_cmd), ">", str(destination))
        if "ssh-pty-base64" in plan:
            print(
                "SSH-PTY-BASE64:",
                "ssh -tt ... 'base64 -w0 <remote_path>' >",
                str(destination),
            )
        return

    last_error: RuntimeError | None = None
    for attempt in range(1, args.retries + 1):
        for transport in plan:
            print(
                f"[collect] attempt {attempt}/{args.retries} transport={transport}",
                file=sys.stderr,
                flush=True,
            )
            if transport == "rsync":
                try:
                    rsync_result = subprocess.run(
                        rsync_cmd,
                        check=False,
                        timeout=args.timeout_seconds,
                    )
                    rsync_code = rsync_result.returncode
                except subprocess.TimeoutExpired:
                    rsync_code = 124
                if rsync_code == 0 and _is_valid_downloaded_file(destination, args.remote_path):
                    print(str(destination))
                    return
                if rsync_code == 0:
                    destination.unlink(missing_ok=True)
                    last_error = RuntimeError(
                        f"rsync attempt {attempt}/{args.retries} failed: invalid/corrupted file."
                    )
                else:
                    last_error = RuntimeError(
                        f"rsync attempt {attempt}/{args.retries} failed (code={rsync_code})"
                    )
                print(f"[collect] {last_error}", file=sys.stderr, flush=True)
                continue

            if transport == "ssh-cat":
                ssh_cat_code = _copy_via_ssh_cat(
                    ssh_target=args.ssh_target,
                    ssh_port=args.ssh_port,
                    identity_file=args.identity_file or None,
                    remote_path=args.remote_path,
                    destination=destination,
                    timeout_seconds=args.timeout_seconds,
                )
                if ssh_cat_code == 0 and _is_valid_downloaded_file(destination, args.remote_path):
                    print(str(destination))
                    return
                if ssh_cat_code == 0:
                    destination.unlink(missing_ok=True)
                    last_error = RuntimeError(
                        f"ssh-cat attempt {attempt}/{args.retries} failed: invalid/corrupted file."
                    )
                else:
                    last_error = RuntimeError(
                        f"ssh-cat attempt {attempt}/{args.retries} failed (code={ssh_cat_code})"
                    )
                print(f"[collect] {last_error}", file=sys.stderr, flush=True)
                continue

            if transport == "ssh-pty-base64":
                ssh_b64_code = _copy_via_ssh_pty_base64(
                    ssh_target=args.ssh_target,
                    ssh_port=args.ssh_port,
                    identity_file=args.identity_file or None,
                    remote_path=args.remote_path,
                    destination=destination,
                    timeout_seconds=args.timeout_seconds,
                )
                if ssh_b64_code == 0 and _is_valid_downloaded_file(destination, args.remote_path):
                    print(str(destination))
                    return
                if ssh_b64_code == 0:
                    destination.unlink(missing_ok=True)
                    last_error = RuntimeError(
                        f"ssh-pty-base64 attempt {attempt}/{args.retries} failed: invalid/corrupted file."
                    )
                else:
                    last_error = RuntimeError(
                        f"ssh-pty-base64 attempt {attempt}/{args.retries} failed (code={ssh_b64_code})"
                    )
                print(f"[collect] {last_error}", file=sys.stderr, flush=True)
                continue

        if last_error is None:
            last_error = RuntimeError(f"Attempt {attempt}/{args.retries} failed.")
        time.sleep(attempt)

    raise RuntimeError("Failed to retrieve artifacts.") from last_error


def main() -> None:
    """CLI entrypoint for artifact collection."""

    collect()


if __name__ == "__main__":
    main()
