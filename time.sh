#!/bin/bash
# Hybrid-SD Wan14B + Wan1.3B 测试脚本
# 使用 Wan2.1-T2V-14B (云侧) + Wan2.1-T2V-1.3B (边缘侧) 进行视频生成
# 记录每个视频的云侧和端侧用时

set -e

echo "=========================================="
echo "Hybrid-SD Wan14B + 1.3B 测试脚本"
echo "使用 Wan2.1-T2V-14B (云侧) + Wan2.1-T2V-1.3B (边缘侧)"
echo "=========================================="

# 激活环境
source /data/chenjiayu/miniconda3/etc/profile.d/conda.sh
conda activate minyu_lee

# 设置 CUDA 库路径
export LD_LIBRARY_PATH="/home/intern/miniforge3/envs/xyt2/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:$LD_LIBRARY_PATH"

# 设置 HuggingFace 镜像
export HF_ENDPOINT=https://hf-mirror.com

# 指定使用 GPU 0
export CUDA_VISIBLE_DEVICES=0

# 进入项目目录
cd /data/chenjiayu/minyu_lee/Hybrid-sd_wan

# 验证环境
export CUDA_HOME=/usr/local/cuda-13.0
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export PATH=$CUDA_HOME/bin:$PATH

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

# 记录开始时间
START_TIME=$(date +%s)
START_DATETIME=$(date '+%Y-%m-%d %H:%M:%S')

echo ""
echo "=========================================="
echo "开始生成 VBench 2.0 视频..."
echo "开始时间: $START_DATETIME"
echo "配置文件: configs/vbench2_half.yaml"
echo "=========================================="

# 运行生成（所有 VBench 2.0 视频）
# 注意: Hybrid-SD框架在生成视频时会自动记录云侧(14B)和端侧(1.3B)的推理时间
# 这些时间数据由pipeline的last_model_timing属性实时记录，保存到model_timing.json中
python tools/vbench/run_generation.py \
    --config configs/vbench2_half.yaml

# 记录结束时间
END_TIME=$(date +%s)
END_DATETIME=$(date '+%Y-%m-%d %H:%M:%S')
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_HOURS=$((ELAPSED / 3600))
ELAPSED_MINUTES=$(((ELAPSED % 3600) / 60))
ELAPSED_SECONDS=$((ELAPSED % 60))

echo ""
echo "=========================================="
echo "✅ 所有视频生成完成！"
echo "结束时间: $END_DATETIME"
echo "总耗时: ${ELAPSED_HOURS}小时${ELAPSED_MINUTES}分钟${ELAPSED_SECONDS}秒"
echo "输出目录: results/vbench/vbench2_half/"
echo "=========================================="

# 处理时间记录
echo ""
echo "=========================================="
echo "处理时间记录..."
echo "(注意: 时间数据由Hybrid-SD框架在生成时记录，解析不影响生成速度)"
echo "=========================================="

# 查找生成的时间记录文件
METRICS_DIR="results/vbench/vbench2_half/metrics"
TIMING_FILE="${METRICS_DIR}/model_timing.json"

if [ -f "$TIMING_FILE" ]; then
    echo "找到时间记录文件: $TIMING_FILE"
    
    # 使用Python脚本解析并显示时间统计
    python -c "
import json
import sys
from pathlib import Path

timing_file = Path('$TIMING_FILE')
if timing_file.exists():
    with open(timing_file, 'r') as f:
        data = json.load(f)
    
    print('\n' + '='*60)
    print('Hybrid-SD Wan14B + 1.3B 时间统计')
    print('='*60)
    
    # 显示每个模型的总时间
    if 'totals' in data:
        totals = data['totals']
        print('\n模型总耗时:')
        for model, time_sec in totals.items():
            print(f'  {model}: {time_sec:.2f}秒')
    
    # 尝试从latency.json获取每个视频的详细时间
    latency_file = Path('${METRICS_DIR}/latency.json')
    if latency_file.exists():
        with open(latency_file, 'r') as f:
            latency_data = json.load(f)
        
        if 'items' in latency_data and len(latency_data['items']) > 0:
            print('\n每个视频的详细时间:')
            print('-'*60)
            print(f'{'序号':<6} {'延迟(秒)':<15}')
            print('-'*60)
            
            total_latency = 0.0
            
            for idx, item in enumerate(latency_data['items'], 1):
                latency = item.get('latency', 0.0)
                total_latency += latency
                print(f'{idx:<6} {latency:<15.2f}')
            
            print('-'*60)
            print(f'{'总计':<6} {total_latency:<15.2f}')
            print('='*60)
            
            # 计算平均时间
            num_videos = len(latency_data['items'])
            if num_videos > 0:
                print(f'\n平均每视频:')
                print(f'  总延迟: {total_latency/num_videos:.2f}秒')
                
                # 计算各模型占比
                if 'totals' in data and total_latency > 0:
                    print(f'\n时间占比:')
                    for model, time_sec in data['totals'].items():
                        pct = time_sec / total_latency * 100
                        print(f'  {model}: {pct:.1f}%')
else:
    print(f'警告: 未找到时间记录文件 {TIMING_FILE}')
    print('请检查生成是否成功完成')
    sys.exit(1)
"
    
    # 复制时间记录到结果目录
    TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
    cp "$TIMING_FILE" "${METRICS_DIR}/model_timing_${TIMESTAMP}.json"
    echo ""
    echo "时间记录已保存并备份"
    
else
    echo "警告: 未找到时间记录文件 $TIMING_FILE"
    echo "请检查生成是否成功完成"
fi

echo ""
echo "=========================================="
echo "✅ 测试脚本执行完成！"
echo "=========================================="
