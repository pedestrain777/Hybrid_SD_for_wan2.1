#!/bin/bash
# Hybrid-SD VBench 1.0 - Wan 2.2 14B + Wan 2.1 1.3B
# 2 GPUs (cuda:5,7), seed=0
# Steps: 28 (cloud) + 12 (edge) = 40 total

set -e

cd /data/chenjiayu/minyu_lee/hybrid_wan_14B_1.3B

source /data/chenjiayu/miniconda3/etc/profile.d/conda.sh
conda activate minyu_lee

export PYTHONPATH=/data/chenjiayu/minyu_lee/hybrid_wan_14B_1.3B:${PYTHONPATH:-}

OUTPUT_DIR="/data/chenjiayu/minyu_lee/A vbench 1.0/outputs_hybrid_wan22"
mkdir -p "${OUTPUT_DIR}" logs/vbench1_hybrid_wan22_2gpu

echo "=========================================="
echo "Hybrid-SD VBench 1.0 - Wan 2.2 14B + Wan 2.1 1.3B"
echo "  Steps: 28 (cloud) + 12 (edge) = 40"
echo "  guidance_scale: 5.0"
echo "  473 prompts, 2 GPUs (cuda:5,7), seed=0"
echo "=========================================="

# GPU 5: human_action(50) + temporal_style(50) + overall_consistency(47) + appearance_style(43) + color(43) = 233
CUDA_VISIBLE_DEVICES=5 python run_vbench1_hybrid_wan22_by_dim.py \
    --output_dir "${OUTPUT_DIR}" \
    --dims human_action temporal_style overall_consistency appearance_style color \
    --seed 0 \
    > logs/vbench1_hybrid_wan22_2gpu/gpu5.log 2>&1 &
echo "GPU 5 started, PID: $!"

# GPU 7: scene(43) + spatial_relationship(42) + multiple_objects(41) + object_class(40) + temporal_flickering(38) + subject_consistency(36) = 240
CUDA_VISIBLE_DEVICES=7 python run_vbench1_hybrid_wan22_by_dim.py \
    --output_dir "${OUTPUT_DIR}" \
    --dims scene spatial_relationship multiple_objects object_class temporal_flickering subject_consistency \
    --seed 0 \
    > logs/vbench1_hybrid_wan22_2gpu/gpu7.log 2>&1 &
echo "GPU 7 started, PID: $!"

wait
echo "=========================================="
echo "All done! Videos saved to: ${OUTPUT_DIR}/video/"
echo "=========================================="
