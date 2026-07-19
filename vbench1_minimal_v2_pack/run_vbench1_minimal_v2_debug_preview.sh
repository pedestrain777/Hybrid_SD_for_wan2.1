#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data/chenjiayu/hengyi_zhang/Hybrid_SD_for_wan2.1_v2}"
PACK_DIR="${PACK_DIR:-${REPO_ROOT}/vbench1_minimal_v2_pack}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/vbench1_minimal_v2_debug_preview}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/logs}"
PYTHON="${PYTHON:-python}"
GPU="${GPU:-5}"
CONFIGS=(${CONFIGS:-L30H10S10})
PROMPT_INDICES=(${PROMPT_INDICES:-1 6 9})

mkdir -p "${LOG_DIR}"
cd "${REPO_ROOT}"

for config in "${CONFIGS[@]}"; do
  "${PYTHON}" "${PACK_DIR}/run_vbench1_minimal_generate.py" \
    --repo_root "${REPO_ROOT}" \
    --output_root "${OUTPUT_ROOT}" \
    --gpu "${GPU}" \
    --configs "${config}" \
    --prompt_indices "${PROMPT_INDICES[@]}" \
    --samples_per_prompt 1 \
    --temporal_samples 1 \
    --debug \
    --debug_every 5 \
    --debug_topk_frames 3 \
    --no_timing \
    --overwrite \
    > "${LOG_DIR}/debug_${config}_gpu${GPU}.log" 2>&1
done
