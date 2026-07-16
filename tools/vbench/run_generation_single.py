#!/usr/bin/env python3
"""
Hybrid Wan2.1 14B + 1.3B - 单 prompt 生成脚本
"""
import argparse
import sys
import time
from pathlib import Path

import torch
import imageio
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from compression.hybrid_sd.inference_pipeline import HybridVideoInferencePipeline


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--models", type=str, nargs="+", required=True)
    parser.add_argument("--steps", type=str, default="32,8")
    parser.add_argument("--num_frames", type=int, default=81)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--guidance_scale", type=float, default=5.0)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print("=" * 60)
    print("Hybrid Wan 视频生成")
    print("=" * 60)
    print(f"Prompt: {args.prompt[:50]}...")
    print(f"Output: {args.output_path}")
    print(f"Models: {[m.split('/')[-1] for m in args.models]}")
    print(f"Steps: {args.steps}")
    print(f"Seed: {args.seed}")
    print("=" * 60)

    class Args:
        def __init__(self):
            self.use_dpm_solver = True
            self.logger = None
            self.enable_xformers_memory_efficient_attention = True
            self.steps = [int(x) for x in args.steps.split(",")]
            self.vae_device = None
            self.use_ecdiff = False
            self.p_steps = 0
            self.k_steps = 0
            self.alpha_smooth = 0.5
            self.s = None

    pipe_args = Args()

    print("加载模型...")
    t0 = time.time()
    pipe = HybridVideoInferencePipeline(
        weight_folders=args.models,
        seed=args.seed,
        device="cuda:0",
        args=pipe_args,
    )
    pipe.set_pipe_and_generator()
    print(f"模型加载完成 ({time.time() - t0:.1f}s)")

    print("开始生成...")
    t0 = time.time()
    output = pipe.generate(
        prompt=args.prompt,
        negative_prompt="",
        num_frames=args.num_frames,
        height=args.height,
        width=args.width,
        guidance_scale=args.guidance_scale,
        num_videos_per_prompt=1,
        use_dynamic_cfg=True,
        output_type="pil",
    )
    gen_time = time.time() - t0
    print(f"生成完成 ({gen_time:.1f}s)")

    # 处理输出帧
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

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(output_path), processed_frames, fps=args.fps, codec='libx264')
    print(f"视频已保存: {output_path}")


if __name__ == "__main__":
    main()
