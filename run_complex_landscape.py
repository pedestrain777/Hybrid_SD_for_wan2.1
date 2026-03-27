#!/usr/bin/env python3
"""
在指定 GPU 上生成 Complex_Landscape 视频
用法: python run_complex_landscape.py <gpu_id> <start_idx> <end_idx>
"""

import os
import sys
import time
import json

gpu_id = sys.argv[1] if len(sys.argv) > 1 else "6"
start_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0
end_idx = int(sys.argv[3]) if len(sys.argv) > 3 else 5

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

sys.path.insert(0, "/data/chenjiayu/minyu_lee/Hybrid-sd_wan")

from pathlib import Path
from compression.hybrid_sd.inference_pipeline import HybridVideoInferencePipeline

# 加载 Complex_Landscape prompts
with open('/data/chenjiayu/minyu_lee/Hybrid-sd_wan/results/vbench/default_exp/complex_landscape_10_prompts.json', 'r') as f:
    all_prompts = json.load(f)

PROMPTS = [p['prompt'] for p in all_prompts[start_idx:end_idx]]

OUTPUT_DIR = Path("/data/chenjiayu/minyu_lee/Hybrid-sd_wan/results/vbench/default_exp/videos_complex_landscape")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 模型配置
MODEL_PATHS = [
    "/data/chenjiayu/models/Wan2.1-T2V-14B-Diffusers",
    "/data/chenjiayu/models/Wan2.1-T2V-1.3B-Diffusers",
]
STEPS = [38, 12]  # 总共 50 步 (38+12)

# 生成参数
NUM_FRAMES = 81
HEIGHT = 720
WIDTH = 1280
GUIDANCE_SCALE = 5.0
FPS = 16

class Args:
    def __init__(self):
        self.enable_xformers_memory_efficient_attention = False
        self.use_dpm_solver = True
        self.logger = None
        self.steps = STEPS

def main():
    print("=" * 60)
    print(f"Complex_Landscape 视频生成 - GPU {gpu_id}")
    print(f"Prompts: {start_idx} to {end_idx-1} (共 {len(PROMPTS)} 个)")
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
        seed=1234,
        device="cuda:0",
        args=args,
    )
    pipeline.set_pipe_and_generator()
    print("模型加载完成")
    print()

    # 生成视频
    for i, prompt in enumerate(PROMPTS):
        global_idx = start_idx + i
        print(f"[{i+1}/{len(PROMPTS)}] Prompt {global_idx}: {prompt[:60]}...")
        output_path = OUTPUT_DIR / f"prompt_{global_idx:05d}_0.mp4"

        if output_path.exists():
            print(f"  已存在，跳过")
            continue

        t0 = time.time()

        video_frames = pipeline.generate(
            prompt=prompt,
            negative_prompt="",
            num_frames=NUM_FRAMES,
            height=HEIGHT,
            width=WIDTH,
            guidance_scale=GUIDANCE_SCALE,
            num_videos_per_prompt=1,
            output_type="pil",
        )

        # 保存视频
        from diffusers.utils import export_to_video
        if isinstance(video_frames, list) and video_frames and isinstance(video_frames[0], list):
            video = video_frames[0]
        else:
            video = video_frames
        export_to_video(video, str(output_path), fps=FPS)

        dt = time.time() - t0
        print(f"  完成 ({dt:.1f}s) -> {output_path}")

    print()
    print(f"GPU {gpu_id} 生成完成!")

if __name__ == "__main__":
    main()
