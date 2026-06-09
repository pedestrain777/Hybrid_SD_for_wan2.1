#!/bin/bash
# Hybrid SD Wan2.1 (14B+1.3B) complex_landscape 6 prompts 生成
# GPU 0-5 并行，每个 GPU 一个 prompt，seed=0

set -e
cd /data/chenjiayu/minyu_lee/Hybrid-sd_wan

source /data/chenjiayu/miniconda3/etc/profile.d/conda.sh
conda activate minyu_lee

OUTPUT_BASE="results/vbench/hybrid_wan2.1_14B_1.3B_complex_landscape"
mkdir -p ${OUTPUT_BASE}/videos
mkdir -p ${OUTPUT_BASE}/logs

echo "============================================================"
echo "Hybrid SD Wan2.1 (14B+1.3B) complex_landscape 生成"
echo "============================================================"
echo "6 prompts, GPU 0-5, seed=0"
echo "Steps: 38+12=50"
echo "============================================================"

for gpu in 0 1 2 3 4 5; do
    echo "[GPU${gpu}] 启动 prompt ${gpu}..."
    python run_hybrid_complex_landscape.py ${gpu} ${gpu} \
        2>&1 | tee ${OUTPUT_BASE}/logs/gpu${gpu}.log &
done

wait
echo "完成！视频在: ${OUTPUT_BASE}/videos/"
