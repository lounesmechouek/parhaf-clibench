#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="${1:-${PARHAF_OUTPUT_DIR:-/workspace/results/default}}"
mkdir -p "${RUN_DIR}"
uv run parhaf-clinbench run \
  --suite "${PARHAF_SUITE:-configs/suites/v1_full.yaml}" \
  --output-dir "${RUN_DIR}"
