#!/bin/bash
# Hybrid-SD (Wan2.2 14B + Wan2.1 1.3B) - single prompt, seeds 879,3,293, cuda:5

set -e

cd /data/chenjiayu/minyu_lee/hybrid_wan_14B_1.3B

source /data/chenjiayu/miniconda3/etc/profile.d/conda.sh
conda activate minyu_lee

PROMPT="A dog is playing in the yard, then it suddenly starts laying on the porch."
OUTPUT_DIR="results/vbench/hybrid_dog_3seeds/videos"
GPU_ID=${1:-5}

mkdir -p ${OUTPUT_DIR}
mkdir -p logs/hybrid_dog_3seeds

echo "=========================================="
echo "Hybrid-SD (Wan2.2 14B + Wan2.1 1.3B)"
echo "Prompt: ${PROMPT}"
echo "Seeds: 879, 3, 293"
echo "GPU: ${GPU_ID}"
echo "=========================================="

# Create temp prompt file
PROMPT_FILE=$(mktemp /tmp/prompt_XXXXXX.txt)
echo "${PROMPT}" > ${PROMPT_FILE}

for seed in 879 3 293; do
    echo ""
    echo "===== seed=${seed} ====="

    CUDA_VISIBLE_DEVICES=${GPU_ID} python tools/vbench/run_simple_generation.py \
        --prompt_file ${PROMPT_FILE} \
        --output_dir ${OUTPUT_DIR} \
        --seed ${seed} \
        2>&1 | tee -a logs/hybrid_dog_3seeds/gpu${GPU_ID}.log
done

rm -f ${PROMPT_FILE}

echo ""
echo "=========================================="
echo "完成! 视频保存在: ${OUTPUT_DIR}"
echo "=========================================="
