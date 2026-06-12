#!/usr/bin/env bash
set -euo pipefail

GPU_ID="${1:-6}"
OUT_ROOT="${2:-/data/chenjiayu/hengyi_zhang/Hybrid_SD_for_wan2.1_v2/v2_eval_multiprompt_50step_cfg}"

STAGES="30,10,10"
NUM_FRAMES=33
HEIGHT=480
WIDTH=832
SPATIAL_CUE="cfg"

PROMPTS=(
  "a black dog running on green grass, centered subject, clear body"
  "a red sports car driving fast on a wet city street, centered subject, clear car body"
  "a white horse galloping across a sandy beach, centered subject, full body"
  "a yellow excavator moving across a construction site, centered subject, clear machine body"
)

mkdir -p "${OUT_ROOT}/videos" "${OUT_ROOT}/logs"

for idx in "${!PROMPTS[@]}"; do
  prompt="${PROMPTS[$idx]}"
  log_path="${OUT_ROOT}/logs/prompt_$((idx + 1)).log"
  echo "================================================================"
  echo "[multiprompt] prompt $((idx + 1))/${#PROMPTS[@]}: ${prompt}"
  echo "[multiprompt] log: ${log_path}"
  echo "================================================================"

  python run_hybrid_complex_landscape.py "${GPU_ID}" \
    --stages "${STAGES}" \
    --num-frames "${NUM_FRAMES}" \
    --height "${HEIGHT}" \
    --width "${WIDTH}" \
    --output-dir "${OUT_ROOT}/videos" \
    --spatial-cue "${SPATIAL_CUE}" \
    "${prompt}" 2>&1 | tee "${log_path}"
done

echo "[multiprompt] done: ${OUT_ROOT}"
