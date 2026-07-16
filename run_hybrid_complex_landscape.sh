#!/bin/bash
# Hybrid Wan2.1 (14B+1.3B) complex_landscape 6 prompts 生成
# GPU 0-5 并行，每个 GPU 一个 prompt，seed=0

set -e
cd /data/chenjiayu/minyu_lee/hybrid_wan_14B_1.3B

source /data/chenjiayu/miniconda3/etc/profile.d/conda.sh
conda activate minyu_lee

export PYTHONPATH="/data/chenjiayu/minyu_lee/hybrid_wan_14B_1.3B:$PYTHONPATH"

OUTPUT_BASE="results/vbench/hybrid_wan2.1_14B_1.3B_complex_landscape"
PROMPT_FILE="/data/chenjiayu/minyu_lee/EC-Diff-main_for_v2i/prompts_complex_landscape.txt"

mkdir -p ${OUTPUT_BASE}/videos
mkdir -p ${OUTPUT_BASE}/logs

echo "============================================================"
echo "Hybrid Wan2.1 (14B+1.3B) complex_landscape 生成"
echo "============================================================"
echo "6 prompts, GPU 0-5, seed=0"
echo "Steps: 32,8"
echo "============================================================"

# 为每个 GPU 分配一个 prompt
for gpu in 0 1 2 3 4 5; do
    prompt=$(sed -n "$((gpu+1))p" ${PROMPT_FILE})
    if [ -z "$prompt" ]; then
        echo "[GPU${gpu}] 无 prompt，跳过"
        continue
    fi

    safe_prompt=$(echo "$prompt" | cut -c1-150 | tr '/' '_' | tr '\\' '_')
    OUTPUT_VIDEO="${OUTPUT_BASE}/videos/${safe_prompt}-0.mp4"

    # 跳过已存在的视频
    if [ -f "$OUTPUT_VIDEO" ]; then
        echo "[GPU${gpu}] 跳过已存在: ${safe_prompt:0:50}..."
        continue
    fi

    echo "[GPU${gpu}] 启动: ${prompt:0:50}..."

    CUDA_VISIBLE_DEVICES=$gpu python tools/vbench/run_generation_single.py \
        --prompt "$prompt" \
        --output_path "$OUTPUT_VIDEO" \
        --models /data/chenjiayu/models/Wan2.1-T2V-14B-Diffusers /data/chenjiayu/models/Wan2.1-T2V-1.3B-Diffusers \
        --steps 32,8 \
        --num_frames 81 \
        --height 720 \
        --width 1280 \
        --guidance_scale 5.0 \
        --fps 16 \
        --seed 0 \
        2>&1 | tee ${OUTPUT_BASE}/logs/gpu${gpu}.log &
done

wait
echo "完成！视频在: ${OUTPUT_BASE}/videos/"
