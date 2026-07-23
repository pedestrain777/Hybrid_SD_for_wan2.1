#!/usr/bin/env bash
set -euo pipefail

repo_root="${WAN_THRESHOLD_REPO_ROOT:-/data/chenjiayu/hengyi_zhang/Hybrid_SD_for_wan2.1_v2}"
results_root="${WAN_THRESHOLD_RESULTS_ROOT:-${repo_root}/results/threshold_ablation_vbench1_20p}"
generation_python="${WAN_THRESHOLD_GENERATION_PYTHON:-/data/chenjiayu/miniconda3/envs/minyu_lee/bin/python}"
evaluation_python="${WAN_THRESHOLD_EVALUATION_PYTHON:-/data/chenjiayu/miniconda3/envs/vbench/bin/python}"
vbench_root="${WAN_THRESHOLD_VBENCH_ROOT:-/data/chenjiayu/hengyi_zhang/VBench}"
gpu_list=(${WAN_THRESHOLD_GPUS:-0 1 2 3 4})
thresholds=(0.10 0.15 0.20 0.25 0.30)

if [[ "${#gpu_list[@]}" -ne "${#thresholds[@]}" ]]; then
  echo "WAN_THRESHOLD_GPUS must contain exactly five GPU ids" >&2
  exit 2
fi

mkdir -p "${results_root}/logs"

wait_for_idle_gpu() {
  local gpu="$1"
  local used_mib
  while true; do
    used_mib=$(nvidia-smi -i "${gpu}" --query-compute-apps=used_memory --format=csv,noheader,nounits \
      | awk '{total += $1} END {print total + 0}')
    if [[ "${used_mib}" -le 2048 ]]; then
      return 0
    fi
    echo "$(date -Is) GPU ${gpu} busy (${used_mib} MiB); waiting"
    sleep 60
  done
}

run_one_threshold() {
  local threshold="$1"
  local gpu="$2"
  local tag="tau_${threshold/./p}"
  local log_path="${results_root}/logs/${tag}_gpu${gpu}.log"
  local completed=0

  wait_for_idle_gpu "${gpu}" >>"${log_path}" 2>&1
  if [[ -f "${results_root}/${tag}/generation_times.csv" ]]; then
    completed=$(awk -F, 'NR > 1 && $1 == "ok" {count++} END {print count + 0}' \
      "${results_root}/${tag}/generation_times.csv")
  fi
  if [[ "${completed}" -lt 20 ]]; then
    echo "$(date -Is) resuming generation threshold=${threshold} gpu=${gpu} completed=${completed}/20" >>"${log_path}"
    "${generation_python}" "${repo_root}/vbench1_minimal_v2_pack/run_threshold_ablation_generate.py" \
      --threshold "${threshold}" \
      --gpu "${gpu}" \
      --repo-root "${repo_root}" \
      --output-root "${results_root}" >>"${log_path}" 2>&1
  else
    echo "$(date -Is) generation already complete threshold=${threshold} (${completed}/20)" >>"${log_path}"
  fi

  echo "$(date -Is) starting VBench threshold=${threshold} gpu=${gpu}" >>"${log_path}"
  CUDA_VISIBLE_DEVICES="${gpu}" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 "${evaluation_python}" \
    "${repo_root}/vbench1_minimal_v2_pack/eval_threshold_ablation.py" \
    --threshold "${threshold}" \
    --device cuda \
    --repo-root "${repo_root}" \
    --results-root "${results_root}" \
    --vbench-root "${vbench_root}" >>"${log_path}" 2>&1
  echo "$(date -Is) finished threshold=${threshold}" >>"${log_path}"
}

pids=()
for index in "${!thresholds[@]}"; do
  run_one_threshold "${thresholds[$index]}" "${gpu_list[$index]}" &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done
if [[ "${failed}" -ne 0 ]]; then
  echo "At least one threshold worker failed; inspect ${results_root}/logs" >&2
  exit 1
fi

"${generation_python}" "${repo_root}/vbench1_minimal_v2_pack/summarize_threshold_ablation.py" \
  --results-root "${results_root}" | tee "${results_root}/logs/summary.log"
