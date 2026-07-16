#!/usr/bin/env python3
"""
CogVideoX-5B + 2B Hybrid-SD Diversity 视频生成脚本
- 3 个 prompts (索引 0, 3, 7)
- 每个 prompt 生成 20 个视频（不同 seed）
- 总共 60 个视频

配置:
  帧数: 49
  帧率: 8 fps
  时长: 6 秒
  分辨率: 480x720
  guidance_scale: 6.0
  总步数: 50 (云侧38步 + 边缘12步)

Usage:
    python tools/run_diversity_hybrid_sd.py --gpu 1
"""

import argparse
import os
import sys

# 在导入 torch 之前解析 GPU 参数并设置环境变量
def _get_gpu_arg():
    for i, arg in enumerate(sys.argv):
        if arg == "--gpu" and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return "1"

os.environ["CUDA_VISIBLE_DEVICES"] = _get_gpu_arg()

import time
import json
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from diffusers.utils import export_to_video

# 项目根目录
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from compression.hybrid_sd.inference_pipeline import HybridVideoInferencePipeline

# Diversity prompts (与 EC-Diff 相同的 3 个)
DIVERSITY_PROMPTS = [
    "A wooden toy is placed gently on the surface of a small bowl of water.",  # 0
    "A man is playing basketball.",  # 3
    "A person is eating ice cream.",  # 7
]

# 模型配置
MODEL_CONFIG = {
    "models": [
        "pretrained_models/CogVideoX-5b",
        "pretrained_models/CogVideoX-2b",
    ],
    "steps": [38, 12],  # 云侧38步 + 边缘12步 = 50步
}

# 生成参数
GEN_PARAMS = {
    "num_frames": 49,
    "height": 480,
    "width": 720,
    "guidance_scale": 6.0,
    "fps": 8,
}


def parse_args():
    parser = argparse.ArgumentParser(description="CogVideoX Hybrid-SD Diversity 视频生成")
    parser.add_argument("--gpu", type=str, default="1", help="GPU ID")
    parser.add_argument("--num_videos", type=int, default=20,
                        help="每个 prompt 生成的视频数量")
    parser.add_argument("--base_seed", type=int, default=42,
                        help="基础随机种子")
    parser.add_argument("--output_dir", type=str,
                        default="/data/chenjiayu/minyu_lee/Diversity/hybrid_sd_cogvideo_5B__2B",
                        help="输出目录")
    return parser.parse_args()


class Args:
    """Pipeline 参数"""
    def __init__(self):
        self.enable_xformers_memory_efficient_attention = False
        self.vae_device = "cuda:0"  # 明确设置 VAE 设备
        self.steps = MODEL_CONFIG["steps"]  # [38, 12]


def build_pipeline(device, seed=42):
    """构建 Hybrid-SD pipeline"""
    print(f"\n{'='*60}")
    print("加载 CogVideoX-5B + 2B Hybrid-SD 模型")
    print(f"{'='*60}")

    model_paths = [str(PROJECT_ROOT / m) for m in MODEL_CONFIG["models"]]
    print(f"  云侧模型: {model_paths[0]}")
    print(f"  边缘模型: {model_paths[1]}")
    print(f"  步数配置: {MODEL_CONFIG['steps']}")
    print(f"  设备: {device}")

    args = Args()
    pipeline = HybridVideoInferencePipeline(
        weight_folders=model_paths,
        seed=seed,
        device=device,
        args=args,
    )

    pipeline.set_pipe_and_generator()

    print("✅ Pipeline 加载完成")
    return pipeline


def generate_video(pipeline, prompt, seed, output_path, device):
    """生成单个视频"""
    generator = torch.Generator(device=device).manual_seed(seed)

    video_frames = pipeline.pipe(
        prompt=prompt,
        num_frames=GEN_PARAMS["num_frames"],
        height=GEN_PARAMS["height"],
        width=GEN_PARAMS["width"],
        guidance_scale=GEN_PARAMS["guidance_scale"],
        generator=generator,
    ).frames[0]

    export_to_video(video_frames, output_path, fps=GEN_PARAMS["fps"])
    return True


def main():
    args = parse_args()
    device = "cuda:0"

    # 创建输出目录
    output_dir = Path(args.output_dir)
    videos_dir = output_dir / "Diversity"
    logs_dir = output_dir / "logs"
    videos_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print("CogVideoX-5B + 2B Hybrid-SD Diversity 生成")
    print(f"{'='*60}")
    print(f"GPU: {args.gpu}")
    print(f"输出目录: {output_dir}")
    print(f"每个 prompt 视频数: {args.num_videos}")
    print(f"总视频数: {len(DIVERSITY_PROMPTS) * args.num_videos}")

    # 构建 pipeline
    pipeline = build_pipeline(device, args.base_seed)

    # 生成结果记录
    results = []
    total_videos = len(DIVERSITY_PROMPTS) * args.num_videos
    completed = 0

    for prompt_idx, prompt in enumerate(DIVERSITY_PROMPTS):
        print(f"\n[Prompt {prompt_idx}] {prompt[:50]}...")

        for video_idx in range(args.num_videos):
            seed = args.base_seed + video_idx
            video_name = f"prompt{prompt_idx}_seed{seed}.mp4"
            video_path = videos_dir / video_name

            # 跳过已存在的视频
            if video_path.exists():
                print(f"  [{video_idx+1}/{args.num_videos}] 已存在，跳过")
                completed += 1
                continue

            print(f"  [{video_idx+1}/{args.num_videos}] seed={seed} ...", end=" ", flush=True)

            try:
                start_time = time.time()
                generate_video(pipeline, prompt, seed, str(video_path), device)
                elapsed = time.time() - start_time
                print(f"完成 ({elapsed:.1f}s)")

                results.append({
                    "prompt_idx": prompt_idx,
                    "seed": seed,
                    "video_path": str(video_path),
                    "time": elapsed,
                    "success": True,
                })
                completed += 1

            except Exception as e:
                print(f"失败: {e}")
                results.append({
                    "prompt_idx": prompt_idx,
                    "seed": seed,
                    "error": str(e),
                    "success": False,
                })

            print(f"  进度: {completed}/{total_videos}")

    # 保存结果
    results_file = logs_dir / "generation_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)

    success_count = sum(1 for r in results if r.get("success", False))
    print(f"\n{'='*60}")
    print("生成完成!")
    print(f"{'='*60}")
    print(f"成功: {success_count}/{total_videos}")
    print(f"结果文件: {results_file}")


if __name__ == "__main__":
    main()
