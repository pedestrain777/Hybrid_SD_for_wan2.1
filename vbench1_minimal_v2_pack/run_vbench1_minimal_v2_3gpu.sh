#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-/data/chenjiayu/hengyi_zhang/Hybrid_SD_for_wan2.1_v2}"
PACK_DIR="${PACK_DIR:-${REPO_ROOT}/vbench1_minimal_v2_pack}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_ROOT}/results/vbench1_minimal_v2_1seed}"
LOG_DIR="${LOG_DIR:-${OUTPUT_ROOT}/logs}"
PYTHON="${PYTHON:-python}"
GPU_FREE_MEM_MIB="${GPU_FREE_MEM_MIB:-30000}"
GPU_POLL_SEC="${GPU_POLL_SEC:-300}"
GPUS=(${GPUS:-2 3 5})
CONFIGS=(${CONFIGS:-L30H10S10 L30H20S0 L35H15S0})

mkdir -p "${LOG_DIR}" "${OUTPUT_ROOT}/.locks" "${OUTPUT_ROOT}/.done"

wait_for_gpu() {
  local gpu="$1"
  while true; do
    local used
    used="$(nvidia-smi -i "${gpu}" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' ')"
    if [[ "${used}" =~ ^[0-9]+$ ]] && (( used < GPU_FREE_MEM_MIB )); then
      echo "GPU ${gpu} is available: memory.used=${used} MiB"
      break
    fi
    echo "GPU ${gpu} busy: memory.used=${used} MiB, waiting ${GPU_POLL_SEC}s"
    sleep "${GPU_POLL_SEC}"
  done
}

claim_config() {
  local config
  for config in "${CONFIGS[@]}"; do
    [[ -e "${OUTPUT_ROOT}/.done/${config}" ]] && continue
    if mkdir "${OUTPUT_ROOT}/.locks/${config}.lock" 2>/dev/null; then
      echo "${config}"
      return 0
    fi
  done
  return 1
}

run_config() {
  local gpu="$1"
  local config="$2"
  local log_file="${LOG_DIR}/generate_${config}_gpu${gpu}.log"
  cd "${REPO_ROOT}"
  echo "Starting ${config} on GPU ${gpu}" | tee -a "${log_file}"
  if "${PYTHON}" "${PACK_DIR}/run_vbench1_minimal_generate.py" \
      --repo_root "${REPO_ROOT}" \
      --output_root "${OUTPUT_ROOT}" \
      --gpu "${gpu}" \
      --configs "${config}" \
      --samples_per_prompt 1 \
      --temporal_samples 1 \
      --time_log "${OUTPUT_ROOT}/${config}/generation_times.csv" \
      --overwrite \
      >> "${log_file}" 2>&1; then
    touch "${OUTPUT_ROOT}/.done/${config}"
    rm -rf "${OUTPUT_ROOT}/.locks/${config}.lock"
    echo "Finished ${config} on GPU ${gpu}" | tee -a "${log_file}"
  else
    rm -rf "${OUTPUT_ROOT}/.locks/${config}.lock"
    echo "Failed ${config} on GPU ${gpu}; see ${log_file}" | tee -a "${log_file}"
    return 1
  fi
}

worker() {
  local gpu="$1"
  while true; do
    local config
    wait_for_gpu "${gpu}"
    if ! config="$(claim_config)"; then
      echo "GPU ${gpu}: no unclaimed configs remain"
      return 0
    fi
    run_config "${gpu}" "${config}"
  done
}

pids=()
for gpu in "${GPUS[@]}"; do
  worker "${gpu}" > "${LOG_DIR}/worker_gpu${gpu}.log" 2>&1 &
  pid=$!
  pids+=("${pid}")
  echo "${pid}" > "${LOG_DIR}/worker_gpu${gpu}.pid"
done

wait "${pids[@]}"
