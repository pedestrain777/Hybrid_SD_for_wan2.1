#!/usr/bin/env python3
"""
Hybrid SD Wan2.1 14B+1.3B - Complex Landscape 生成
用法: python run_hybrid_complex_landscape.py <gpu_id> <prompt_idx>
"""

import os
import sys
import time

gpu_id = sys.argv[1] if len(sys.argv) > 1 else "0"
prompt_idx = int(sys.argv[2]) if len(sys.argv) > 2 else 0

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

sys.path.insert(0, "/data/chenjiayu/minyu_lee/Hybrid-sd_wan")

from pathlib import Path
from compression.hybrid_sd.inference_pipeline import HybridVideoInferencePipeline

# 加载 prompts
PROMPT_FILE = "/data/chenjiayu/minyu_lee/EC-Diff-main_for_v2i/prompts_complex_landscape.txt"
with open(PROMPT_FILE, 'r') as f:
    all_prompts = [line.strip() for line in f if line.strip()]

prompt = all_prompts[prompt_idx]

OUTPUT_DIR = Path("/data/chenjiayu/minyu_lee/Hybrid-sd_wan/results/vbench/hybrid_wan2.1_14B_1.3B_complex_landscape/videos")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 模型配置
MODEL_PATHS = [
    "/data/chenjiayu/models/Wan2.1-T2V-14B-Diffusers",
    "/data/chenjiayu/models/Wan2.1-T2V-1.3B-Diffusers",
]
STEPS = [38, 12]  # 总共 50 步

# 生成参数
NUM_FRAMES = 81
HEIGHT = 720
WIDTH = 1280
GUIDANCE_SCALE = 5.0
FPS = 16
SEED = 0

class Args:
    def __init__(self):
        self.enable_xformers_memory_efficient_attention = False
        self.use_dpm_solver = True
        self.logger = None
        self.steps = STEPS

def main():
    safe_prompt = prompt[:150].replace('/', '_').replace('\\', '_')
    output_path = OUTPUT_DIR / f"{safe_prompt}-{SEED}.mp4"

    if output_path.exists():
        print(f"[GPU{gpu_id}] 跳过已存在: {safe_prompt[:50]}...")
        return

    print("=" * 60)
    print(f"Hybrid SD Wan2.1 14B+1.3B - GPU {gpu_id}")
    print(f"Prompt {prompt_idx}: {prompt[:60]}...")
    print("=" * 60)
    print(f"云侧模型: {MODEL_PATHS[0]}")
    print(f"边缘模型: {MODEL_PATHS[1]}")
    print(f"步数配置: {STEPS}")
    print(f"Seed: {SEED}")
    print(f"输出: {output_path}")
    print()

    print("加载模型...")
    args = Args()
    pipeline = HybridVideoInferencePipeline(
        weight_folders=MODEL_PATHS,
        seed=SEED,
        device="cuda:0",
        args=args,
    )
    pipeline.set_pipe_and_generator()
    print("模型加载完成")
    print()

    print(f"开始生成...")
    t0 = time.time()

    output = pipeline.generate(
        prompt=prompt,
        negative_prompt="",
        num_frames=NUM_FRAMES,
        height=HEIGHT,
        width=WIDTH,
        guidance_scale=GUIDANCE_SCALE,
        num_videos_per_prompt=1,
        use_dynamic_cfg=True,
        output_type="pil",
    )

    gen_time = time.time() - t0
    print(f"生成完成 ({gen_time:.1f}s)")

    # 保存视频
    from diffusers.utils import export_to_video
    if hasattr(output, 'frames'):
        frames = output.frames[0] if isinstance(output.frames, list) else output.frames
    elif isinstance(output, list):
        frames = output[0] if len(output) > 0 and isinstance(output[0], list) else output
    else:
        frames = output

    export_to_video(frames, str(output_path), fps=FPS)
    print(f"视频已保存: {output_path}")

if __name__ == "__main__":
    main()
