#!/usr/bin/env python3
"""
Hybrid Wan2.2 14B + Wan2.1 1.3B - 简单生成脚本
基于 run_diversity_generation.py 的工作实现
"""
import argparse
import os
import sys
import time
from pathlib import Path

import torch
import imageio
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from compression.hybrid_sd.inference_pipeline import HybridVideoInferencePipeline

# 模型配置
MODEL_CONFIG = {
    "name": "Hybrid-SD (Wan2.2 A14B + Wan2.1 1.3B)",
    "models": [
        "/data/chenjiayu/models/Wan2.2-T2V-A14B-Diffusers",
        "/data/chenjiayu/models/Wan2.1-T2V-1.3B-Diffusers",
    ],
    "steps": [28, 12],  # 28步 14B + 12步 1.3B = 40步
}

GEN_PARAMS = {
    "num_frames": 81,
    "height": 720,
    "width": 1280,
    "guidance_scale": 5.0,
    "fps": 16,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt_file", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # 读取 prompts
    with open(args.prompt_file, "r") as f:
        prompts = [line.strip() for line in f if line.strip()]

    print(f"共 {len(prompts)} 个 prompt")
    print(f"模式: {MODEL_CONFIG['name']}")
    print(f"seed: {args.seed}")

    class Args:
        def __init__(self):
            self.use_dpm_solver = True
            self.logger = None
            self.enable_xformers_memory_efficient_attention = True
            self.steps = MODEL_CONFIG["steps"]
            self.vae_device = None
            self.use_ecdiff = False
            self.p_steps = 0
            self.k_steps = 0
            self.alpha_smooth = 0.5
            self.s = None

    pipe_args = Args()
    pipe = HybridVideoInferencePipeline(
        weight_folders=MODEL_CONFIG["models"],
        seed=args.seed,
        device="cuda:0",
        args=pipe_args,
    )
    pipe.set_pipe_and_generator()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, prompt in enumerate(prompts):
        video_name = f"{prompt}-{args.seed}.mp4"
        video_path = output_dir / video_name

        if video_path.exists():
            print(f"[{i+1}/{len(prompts)}] 跳过: {prompt[:50]}...")
            continue

        print(f"[{i+1}/{len(prompts)}] 生成: {prompt[:50]}...")

        # 更新 seed
        pipe.seed = args.seed
        pipe.generator = torch.Generator(device=pipe.device).manual_seed(args.seed)

        start_time = time.time()

        output = pipe.generate(
            prompt=prompt,
            negative_prompt="",
            num_frames=GEN_PARAMS["num_frames"],
            height=GEN_PARAMS["height"],
            width=GEN_PARAMS["width"],
            guidance_scale=GEN_PARAMS["guidance_scale"],
            num_videos_per_prompt=1,
            use_dynamic_cfg=True,
            output_type="pil",
        )

        elapsed = time.time() - start_time

        # 保存视频 - 来自工作实现
        if hasattr(output, 'frames'):
            frames = output.frames[0] if isinstance(output.frames, list) else output.frames
        elif isinstance(output, list):
            if len(output) > 0 and isinstance(output[0], list):
                frames = output[0]
            else:
                frames = output
        else:
            frames = output

        processed_frames = []
        for frame in frames:
            if isinstance(frame, torch.Tensor):
                f = frame.cpu().numpy()
                if f.max() <= 1.0:
                    f = (f * 255).astype(np.uint8)
                else:
                    f = f.astype(np.uint8)
                if f.ndim == 3 and f.shape[0] in [1, 3, 4]:
                    f = np.transpose(f, (1, 2, 0))
                processed_frames.append(f)
            else:
                f = np.array(frame)
                if f.ndim == 3 and f.shape[-1] == 4:
                    f = f[:, :, :3]
                processed_frames.append(f)

        imageio.mimsave(str(video_path), processed_frames, fps=GEN_PARAMS["fps"], codec='libx264')
        print(f"    完成 ({elapsed:.1f}s), 保存: {video_path.name[:50]}...")

    print("完成!")

if __name__ == "__main__":
    main()
