#!/usr/bin/env bash
set -euo pipefail

SUITE_PATH="${PARHAF_SUITE:-configs/suites/v1_full.yaml}"
OUTPUT_DIR="${PARHAF_OUTPUT_DIR:-${OUTPUT_DIR:-/workspace/results/default}}"
FINAL_ARCHIVE_PATH="${FINAL_ARCHIVE_PATH:-/workspace/results.tar.zst}"
EXPORT_DIR="${EXPORT_DIR:-/workspace/exports}"

HF_HOME="${HF_HOME:-/workspace/.cache/huggingface}"
HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/workspace/.cache/huggingface/hub}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/workspace/.cache/huggingface/transformers}"
MODEL_CACHE_ROOT="${MODEL_CACHE_ROOT:-/workspace/models}"
DATASET_CACHE_ROOT="${DATASET_CACHE_ROOT:-/workspace/datasets}"

export HF_HOME
export HUGGINGFACE_HUB_CACHE
export TRANSFORMERS_CACHE
export MODEL_CACHE_ROOT
export DATASET_CACHE_ROOT

mkdir -p \
  "${OUTPUT_DIR}" \
  "${EXPORT_DIR}" \
  "$(dirname "${FINAL_ARCHIVE_PATH}")" \
  "${HF_HOME}" \
  "${HUGGINGFACE_HUB_CACHE}" \
  "${TRANSFORMERS_CACHE}" \
  "${MODEL_CACHE_ROOT}" \
  "${DATASET_CACHE_ROOT}"

# Workflow unifié local/RunPod:
# - préfetch modèles + datasets selon les révisions de la suite
# - démarrage/arrêt automatique de vLLM par modèle vLLM
uv run parhaf-clinbench run \
  --suite "${SUITE_PATH}" \
  --output-dir "${OUTPUT_DIR}"

tar --zstd -cf "${FINAL_ARCHIVE_PATH}" -C "${OUTPUT_DIR}" .
cp "${FINAL_ARCHIVE_PATH}" "${EXPORT_DIR}/results.tar.zst"
echo "Archive finale: ${FINAL_ARCHIVE_PATH}"
echo "Archive export: ${EXPORT_DIR}/results.tar.zst"
