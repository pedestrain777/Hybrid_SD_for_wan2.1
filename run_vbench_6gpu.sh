#!/bin/bash
# VBench 2.0 视频生成 - 6 GPU 并行运行脚本
# GPU: 2, 3, 4, 5, 6, 7

set -e

cd /data/chenjiayu/minyu_lee/hybrid_wan_14B_1.3B

# 激活 conda 环境
source /data/chenjiayu/miniconda3/etc/profile.d/conda.sh
conda activate minyu_lee

echo "=========================================="
echo "VBench 2.0 视频生成 - Wan2.2 14B + Wan2.1 1.3B"
echo "=========================================="
echo "总 prompts: 314"
echo "GPU 分配:"
echo "  GPU 2: 53 prompts"
echo "  GPU 3: 53 prompts"
echo "  GPU 4: 52 prompts"
echo "  GPU 5: 52 prompts"
echo "  GPU 6: 52 prompts"
echo "  GPU 7: 52 prompts"
echo "=========================================="

# 创建日志目录
mkdir -p logs/vbench_generation

# 启动6个GPU的生成任务
for gpu_id in 2 3 4 5 6 7; do
    echo "启动 GPU ${gpu_id} 任务..."
    CUDA_VISIBLE_DEVICES=${gpu_id} python tools/vbench/run_generation.py \
        --config configs/vbench2_wan22_14B_1.3B_gpu${gpu_id}.yaml \
        > logs/vbench_generation/gpu${gpu_id}.log 2>&1 &
    echo "  PID: $!"
done

echo ""
echo "所有任务已启动！"
echo "查看日志: tail -f logs/vbench_generation/gpu*.log"
echo "查看进度: watch -n 10 'ls -la results/vbench/ec_diff_wan2.2_14B_1.3B_sampled314/videos/ | wc -l'"

# 等待所有后台任务完成
wait

echo ""
echo "=========================================="
echo "所有任务完成！"
echo "=========================================="
