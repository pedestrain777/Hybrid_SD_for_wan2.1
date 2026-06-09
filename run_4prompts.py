#!/usr/bin/env python3
"""
Hybrid-SD Wan 生成 4 个视频
"""

import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

sys.path.insert(0, "/data/chenjiayu/minyu_lee/Hybrid-sd_wan")

from pathlib import Path
from compression.hybrid_sd.inference_pipeline import HybridVideoInferencePipeline

PROMPTS = [
    "A piece of styrofoam is placed on the surface of a bowl filled with water",
    "A dog is running through the house, then it suddenly jumps onto the couch",
    "A giraffe breakdancing, spinning on its back and flipping gracefully despite its towering height",
    "A lion with the ears of a bat, the body of a whale, the claws of an eagle, and the wings of a dragon, an unstoppable predator both in the sea and in the sky",
]

OUTPUT_DIR = Path("/data/chenjiayu/minyu_lee/Hybrid-sd_wan/results/4prompts")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 模型配置 - Wan2.1-14B + Wan2.1-1.3B (Diffusers 格式)
MODEL_PATHS = [
    "/data/chenjiayu/models/Wan2.1-T2V-14B-Diffusers",
    "/data/chenjiayu/models/Wan2.1-T2V-1.3B-Diffusers",
]
STEPS = [38, 12]  # 云侧 38 步，边缘侧 12 步

class Args:
    def __init__(self):
        self.enable_xformers_memory_efficient_attention = False
        self.steps = STEPS

def main():
    print("=" * 60)
    print("Hybrid-SD Wan 生成 4 个视频")
    print("=" * 60)
    print(f"云侧模型: {MODEL_PATHS[0]}")
    print(f"边缘模型: {MODEL_PATHS[1]}")
    print(f"步数配置: {STEPS}")
    print(f"输出目录: {OUTPUT_DIR}")
    print()

    # 构建 pipeline
    print("加载模型...")
    args = Args()
    pipeline = HybridVideoInferencePipeline(
        weight_folders=MODEL_PATHS,
        seed=42,
        device="cuda:0",
        args=args,
    )
    pipeline.set_pipe_and_generator()
    print("模型加载完成")
    print()

    # 生成视频
    for i, prompt in enumerate(PROMPTS):
        print(f"[{i+1}/4] {prompt[:60]}...")
        output_path = OUTPUT_DIR / f"video_{i}.mp4"

        if output_path.exists():
            print(f"  已存在，跳过")
            continue

        t0 = time.time()

        video_frames = pipeline.generate(
            prompt=prompt,
            negative_prompt="",
            num_frames=81,
            height=720,
            width=1280,
            guidance_scale=5.0,
            num_videos_per_prompt=1,
            output_type="pil",
        )

        # 保存视频
        from diffusers.utils import export_to_video
        if isinstance(video_frames, list) and video_frames and isinstance(video_frames[0], list):
            video = video_frames[0]
        else:
            video = video_frames
        export_to_video(video, str(output_path), fps=16)

        dt = time.time() - t0
        print(f"  完成 ({dt:.1f}s) -> {output_path}")

    print()
    print("全部完成!")

if __name__ == "__main__":
    main()
