#!/usr/bin/env bash
set -euo pipefail

RESULTS_DIR="${RESULTS_DIR:-/workspace/results}"
EXPORT_DIR="${EXPORT_DIR:-/workspace/exports}"
mkdir -p "${EXPORT_DIR}"

archive="${EXPORT_DIR}/results.tar.zst"
tar --zstd -cf "${archive}" -C "${RESULTS_DIR}" .
echo "Archive exportée: ${archive}"
