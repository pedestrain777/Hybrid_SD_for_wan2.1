#!/bin/bash
# Hybrid-SD VBench 1.0 - Wan 2.2 14B + Wan 2.1 1.3B
# GPU 0 only, finish remaining 188 videos
# Script has skip logic - will auto-skip already generated videos

set -e

cd /data/chenjiayu/minyu_lee/hybrid_wan_14B_1.3B

source /data/chenjiayu/miniconda3/etc/profile.d/conda.sh
conda activate minyu_lee

export PYTHONPATH=/data/chenjiayu/minyu_lee/hybrid_wan_14B_1.3B:${PYTHONPATH:-}

OUTPUT_DIR="/data/chenjiayu/minyu_lee/A vbench 1.0/outputs_hybrid_wan22"
mkdir -p "${OUTPUT_DIR}" logs

echo "=========================================="
echo "Hybrid-SD VBench 1.0 - Remaining videos"
echo "  GPU 0, all 11 dims (skip existing)"
echo "=========================================="

CUDA_VISIBLE_DEVICES=0 python run_vbench1_hybrid_wan22_by_dim.py \
    --output_dir "${OUTPUT_DIR}" \
    --dims overall_consistency appearance_style color object_class temporal_flickering subject_consistency \
    --seed 0 \
    2>&1 | tee logs/vbench1_hybrid_wan22_remaining_gpu0.log

echo "=========================================="
echo "Done! Check: ${OUTPUT_DIR}/video/"
echo "=========================================="
