#!/bin/bash
# Hybrid-SD (Wan2.2 14B + Wan2.1 1.3B) - 16 prompts, 8 GPU, seed=0

set -e

cd /data/chenjiayu/minyu_lee/hybrid_wan_14B_1.3B

source /data/chenjiayu/miniconda3/etc/profile.d/conda.sh
conda activate minyu_lee

PROMPT_FILE="/data/chenjiayu/minyu_lee/VDiT/results/vbench/wan22_14B_eden_16prompts/selected_prompts.txt"
OUTPUT_DIR="results/vbench/hybrid_wan22_14B_1.3B_16prompts/videos"

mkdir -p ${OUTPUT_DIR}
mkdir -p logs/vbench_16prompts

echo "=========================================="
echo "Hybrid-SD (Wan2.2 14B + Wan2.1 1.3B)"
echo "16 prompts, 8 GPU, seed=0"
echo "=========================================="

# 分配 prompts 到 8 个 GPU
python3 << 'EOF'
prompt_file = "/data/chenjiayu/minyu_lee/VDiT/results/vbench/wan22_14B_eden_16prompts/selected_prompts.txt"
output_dir = "configs/16prompts"

import os
os.makedirs(output_dir, exist_ok=True)

with open(prompt_file, "r") as f:
    prompts = [line.strip() for line in f if line.strip()]

print(f"总 prompts: {len(prompts)}")

for gpu in range(8):
    gpu_prompts = prompts[gpu*2:(gpu+1)*2]
    with open(f"{output_dir}/prompts_gpu{gpu}.txt", "w") as f:
        for p in gpu_prompts:
            f.write(p + "\n")
    print(f"GPU {gpu}: {len(gpu_prompts)} prompts")
EOF

echo "=========================================="

for gpu in 0 1 2 3 4 5 6 7; do
    echo "启动 GPU ${gpu}..."
    CUDA_VISIBLE_DEVICES=${gpu} python tools/vbench/run_simple_generation.py \
        --prompt_file configs/16prompts/prompts_gpu${gpu}.txt \
        --output_dir ${OUTPUT_DIR} \
        --seed 0 \
        > logs/vbench_16prompts/gpu${gpu}.log 2>&1 &
done

echo "所有 GPU 任务已启动，等待完成..."
wait

echo "=========================================="
echo "完成! 视频保存在: ${OUTPUT_DIR}"
echo "=========================================="
