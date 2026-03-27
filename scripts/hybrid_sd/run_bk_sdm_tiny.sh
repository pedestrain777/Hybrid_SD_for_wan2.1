#!/bin/bash
export PYTHONPATH='.'
MODEL_ROOT=/data/models/hybridsd_checkpoint
MODEL_LARGE=CompVis--stable-diffusion-v1-4
MODEL_SMALL=bk-sdm-tiny

PATH_MODEL_LARGE=$MODEL_ROOT/$MODEL_LARGE
PATH_MODEL_SMALL=$MODEL_ROOT/$MODEL_SMALL

GPU_NUM=0

# 协同推理步数配置：格式为 "大模型步数,小模型步数"
# 例如 "10,15" 表示大模型运行10步，小模型运行15步
step_list=("0,25" "10,15" "15,10" "25,0")


for STEP in ${step_list[@]}
do
    OUTPUT_DIR=results/HybridSD_bk_sdm_tiny/$MODEL_LARGE-$MODEL_SMALL-$STEP
    CUDA_VISIBLE_DEVICES=$GPU_NUM python3 examples/hybrid_sd/hybrid.py \
            --model_id $PATH_MODEL_LARGE $PATH_MODEL_SMALL\
            --steps $STEP  \
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
done

