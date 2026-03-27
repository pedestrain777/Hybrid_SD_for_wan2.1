#!/bin/bash
export PYTHONPATH='.'
MODEL_ROOT=/data/models/hybridsd_checkpoint
MODEL_LARGE=THUDM--CogVideoX-5B
MODEL_SMALL=THUDM--CogVideoX-2B

PATH_MODEL_LARGE=$MODEL_ROOT/$MODEL_LARGE
PATH_MODEL_SMALL=$MODEL_ROOT/$MODEL_SMALL

GPU_NUM=0

# 运行 (25,0) 配置：大模型运行25步，小模型运行0步
STEP="25,0"

# 创建专门的输出文件夹
OUTPUT_DIR=results/run_25_0

echo "开始运行配置 (25,0)"
echo "输出目录: $OUTPUT_DIR"

CUDA_VISIBLE_DEVICES=$GPU_NUM python3 examples/hybrid_sd/hybrid_video.py \
        --model_id $PATH_MODEL_LARGE $PATH_MODEL_SMALL \
        --steps $STEP \
        --prompts_file examples/hybrid_sd/prompts.txt \
        --seed 1674753452 \
        --num_frames 49 \
        --height 480 \
        --width 720 \
        --output_dir $OUTPUT_DIR \
        --num_videos_per_prompt 1 \
        --num_videos 1 \
        --enable_xformers_memory_efficient_attention \
        --use_dpm_solver \
        --guidance_scale 6.0 \
        --use_dynamic_cfg \
        --fps 8

echo "运行完成！视频已保存到: $OUTPUT_DIR"

