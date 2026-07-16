#!/bin/bash
source /data/chenjiayu/miniconda3/bin/activate minyu_lee
cd /data/chenjiayu/minyu_lee/Hybird-SD-mian_for_v2i
export PYTHONPATH='.'

MODEL_LARGE=/data/chenjiayu/minyu_lee/EC-Diff-main_for_v2i/pretrained_models/CogVideoX-5b
MODEL_SMALL=/data/chenjiayu/minyu_lee/EC-Diff-main_for_v2i/pretrained_models/CogVideoX-2b
OUTPUT_DIR=/data/chenjiayu/minyu_lee/hybridsd_complex_landscape/prompt_$1
STEPS="50,0"

PROMPT="$2"

CUDA_VISIBLE_DEVICES=$1 python3 examples/hybrid_sd/hybrid_video.py \
    --model_id $MODEL_LARGE $MODEL_SMALL \
    --steps $STEPS \
    --val_prompts "$PROMPT" \
    --neg_val_prompts "" \
    --seed 1234 \
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
    --fps 8 \
    --device cuda:0