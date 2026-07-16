#!/bin/bash
# 单独启动指定 GPU 的生成任务
# 用法: ./run_single_gpu.sh <gpu_id>
# 例如: ./run_single_gpu.sh 2

set -e

if [ -z "$1" ]; then
    echo "用法: $0 <gpu_id>"
    echo "例如: $0 2"
    exit 1
fi

GPU_ID=$1

cd /data/chenjiayu/minyu_lee/hybrid_wan_14B_1.3B

# 激活 conda 环境
source /data/chenjiayu/miniconda3/etc/profile.d/conda.sh
conda activate minyu_lee

echo "启动 GPU ${GPU_ID} 任务..."
mkdir -p logs/vbench_generation

CUDA_VISIBLE_DEVICES=${GPU_ID} python tools/vbench/run_generation.py \
    --config configs/vbench2_wan22_14B_1.3B_gpu${GPU_ID}.yaml \
    2>&1 | tee logs/vbench_generation/gpu${GPU_ID}.log
