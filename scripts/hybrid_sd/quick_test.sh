#!/bin/bash
# 快速测试脚本 - 只运行一个步数配置
export PYTHONPATH='.'

MODEL_ROOT=/data/models/hybridsd_checkpoint
PATH_MODEL_LARGE=$MODEL_ROOT/CompVis--stable-diffusion-v1-4
PATH_MODEL_SMALL=$MODEL_ROOT/bk-sdm-tiny

GPU_NUM=0

# 快速测试：大模型10步 + 小模型15步
OUTPUT_DIR=results/quick_test
CUDA_VISIBLE_DEVICES=$GPU_NUM python3 examples/hybrid_sd/hybrid.py \
        --model_id $PATH_MODEL_LARGE $PATH_MODEL_SMALL\
        --steps 10,15 \
        --prompts_file examples/hybrid_sd/prompts.txt \
        --seed 1674753452 \
        --img_sz 512 \
        --output_dir $OUTPUT_DIR \
        --num_images_per_prompt 1 \
        --num_images 1 \
        --enable_xformers_memory_efficient_attention \
        --save_middle \
        --use_dpm_solver \
        --guidance_scale 7

echo "测试完成！结果保存在: $OUTPUT_DIR"

