#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 CONFIG GPU" >&2
  exit 2
fi

CONFIG="$1"
GPU="$2"
REPO_ROOT="${REPO_ROOT:-/data/chenjiayu/hengyi_zhang/Hybrid_SD_for_wan2.1_v2}"
PACK_DIR="${PACK_DIR:-${REPO_ROOT}/vbench1_minimal_v2_pack}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/vbench1_minimal_v2_1seed}"
PYTHON="${PYTHON:-python}"
LOG_DIR="${OUTPUT_ROOT}/logs"
LOCK_DIR="${OUTPUT_ROOT}/.locks/${CONFIG}.lock"
DONE_FILE="${OUTPUT_ROOT}/.done/${CONFIG}"

mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}/.locks" "${OUTPUT_ROOT}/.done"

if [[ -e "${DONE_FILE}" ]]; then
  echo "${CONFIG} is already marked done"
  exit 0
fi

if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
  echo "${CONFIG} is already locked"
  exit 0
fi

cleanup() {
  rm -rf "${LOCK_DIR}"
}
trap cleanup EXIT

cd "${REPO_ROOT}"
"${PYTHON}" "${PACK_DIR}/run_vbench1_minimal_generate.py" \
  --repo_root "${REPO_ROOT}" \
  --output_root "${OUTPUT_ROOT}" \
  --gpu "${GPU}" \
  --configs "${CONFIG}" \
  --samples_per_prompt 1 \
  --temporal_samples 1 \
  --time_log "${OUTPUT_ROOT}/${CONFIG}/generation_times.csv" \
  --overwrite

touch "${DONE_FILE}"
