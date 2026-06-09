#!/bin/bash
# Hybrid SD Wan2.1 (14B+1.3B) - VBench 1.0 with augmented prompts
# 2 GPUs (cuda:2,3), seed=0
# Steps: 14B:38 + 1.3B:12 = 50 total

set -e
cd /data/chenjiayu/minyu_lee/Hybrid-sd_wan

source /data/chenjiayu/miniconda3/etc/profile.d/conda.sh
conda activate minyu_lee

export PYTHONPATH=/data/chenjiayu/minyu_lee/Hybrid-sd_wan:$PYTHONPATH

OUTPUT_DIR="/data/chenjiayu/minyu_lee/A vbench 1.0/outputs_hybrid"

mkdir -p "${OUTPUT_DIR}" logs/vbench1_hybrid_2gpu

echo "=========================================="
echo "Hybrid SD Wan2.1 (14B+1.3B) - VBench 1.0 (augmented prompts)"
echo "473 prompts, 2 GPUs (cuda:2,3), seed=0"
echo "Steps: 14B:38 + 1.3B:12 = 50"
echo "=========================================="

# GPU 2: human_action(50) + temporal_style(50) + overall_consistency(47) + appearance_style(43) + color(43) = 233
CUDA_VISIBLE_DEVICES=2 python run_vbench1_hybrid_by_dim.py \
    --output_dir "${OUTPUT_DIR}" \
    --dims human_action temporal_style overall_consistency appearance_style color \
    --seed 0 \
    > logs/vbench1_hybrid_2gpu/gpu2.log 2>&1 &
echo "GPU 2 started, PID: $!"

# GPU 3: scene(43) + spatial_relationship(42) + multiple_objects(41) + object_class(40) + temporal_flickering(38) + subject_consistency(36) = 240
CUDA_VISIBLE_DEVICES=3 python run_vbench1_hybrid_by_dim.py \
    --output_dir "${OUTPUT_DIR}" \
    --dims scene spatial_relationship multiple_objects object_class temporal_flickering subject_consistency \
    --seed 0 \
    > logs/vbench1_hybrid_2gpu/gpu3.log 2>&1 &
echo "GPU 3 started, PID: $!"

wait
echo "=========================================="
echo "All done! Videos saved to: ${OUTPUT_DIR}/video/"
echo "=========================================="
