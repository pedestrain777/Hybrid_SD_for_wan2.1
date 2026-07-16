#!/bin/bash
# Hybrid-SD 完整版本 - 生成所有 VBench 2.0 视频
# 使用 Hybrid-SD 混合梯度diffusion方法：cloud_steps=50, edge_steps=0

set -e

echo "=========================================="
echo "Hybrid-SD 完整版本 - 生成所有 VBench 2.0 视频"
echo "使用 Hybrid-SD 混合梯度diffusion方法"
echo "=========================================="

# 激活环境
source /data/chenjiayu/miniconda3/etc/profile.d/conda.sh
conda activate minyu_lee

# 设置 CUDA 库路径
export LD_LIBRARY_PATH="/home/intern/miniforge3/envs/xyt2/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH"

# 设置 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com

# 指定使用 GPU 1
export CUDA_VISIBLE_DEVICES=1

# 进入项目目录
cd /data/chenjiayu/minyu_lee/Hybird-SD-mian_for_v2i

# 验证环境
# ========== 修复后的环境验证代码 ==========
# 1. 先加载CUDA环境变量（必须放在最前面）
export CUDA_HOME=/usr/local/cuda-13.0
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH
export CUDA_VISIBLE_DEVICES=0

# 2. 合并所有检测逻辑到一个Python进程（关键！）
echo ""
echo "环境信息:"
echo "Python: $(which python)"
echo "Python 版本: $(python --version | awk '{print $2}')"
python -c "
import torch
# 一次性输出所有环境信息，确保在同一个进程里检测
print(f'PyTorch 版本: {torch.__version__}')
print(f'CUDA 可用: {torch.cuda.is_available()}')
print(f'CUDA 设备数: {torch.cuda.device_count()}')
if torch.cuda.is_available():
    print(f'当前GPU: {torch.cuda.get_device_name(0)}')
"
# =========================================

# 生成 metadata.csv
echo ""
echo "=========================================="
echo "生成 metadata.csv..."
echo "=========================================="
python tools/vbench/build_metadata.py \
    --config configs/configs/hybrid_sd_vbench2_full.yaml

# 运行完整生成（所有1013个视频）
echo ""
echo "=========================================="
echo "开始生成所有 1013 个视频..."
echo "预计耗时: ~30-40 小时"
echo "=========================================="
python tools/vbench/run_generation.py \
    --config configs/configs/hybrid_sd_vbench2_full.yaml \
    --enable_profiling

echo ""
echo "=========================================="
echo "✅ 所有视频生成完成！"
echo "输出目录: results/vbench/hybrid_sd_full/Hybrid-SD_CogvideoX-5B+2B_VBench/"
echo "=========================================="
