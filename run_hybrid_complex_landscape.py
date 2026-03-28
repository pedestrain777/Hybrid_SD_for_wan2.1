#!/usr/bin/env python3
"""
Hybrid SD Wan2.1 14B+1.3B - Complex Landscape 生成

用法:
  python run_hybrid_complex_landscape.py <gpu_id>                    # 默认用 prompts 文件第 0 行
  python run_hybrid_complex_landscape.py <gpu_id> <prompt_idx>       # 用文件里第 prompt_idx 行（从 0 计数）
  python run_hybrid_complex_landscape.py <gpu_id> "黑色小狗在跑步"    # 自定义整句 prompt（含中文）
"""

import hashlib
import os
import sys
import time
from pathlib import Path

gpu_id = sys.argv[1] if len(sys.argv) > 1 else "0"
PROMPT_FILE = "/data/chenjiayu/minyu_lee/EC-Diff-main_for_v2i/prompts_complex_landscape.txt"


def _load_prompts():
    with open(PROMPT_FILE, "r", encoding="utf-8", errors="replace") as f:
        return [line.strip() for line in f if line.strip()]


prompt_idx = 0
prompt_tag = "idx_000"

if len(sys.argv) > 2:
    try:
        idx_candidate = int(sys.argv[2])
        if len(sys.argv) == 3 and idx_candidate >= 0:
            prompt_idx = idx_candidate
            all_prompts = _load_prompts()
            if prompt_idx >= len(all_prompts):
                raise SystemExit(f"prompt_idx={prompt_idx} 超出文件行数 {len(all_prompts)}")
            prompt = all_prompts[prompt_idx]
            prompt_tag = f"idx_{prompt_idx:03d}"
        else:
            prompt = " ".join(sys.argv[2:])
            prompt_idx = -1
            prompt_tag = "custom_" + hashlib.md5(prompt.encode("utf-8")).hexdigest()[:10]
    except ValueError:
        prompt = " ".join(sys.argv[2:])
        prompt_idx = -1
        prompt_tag = "custom_" + hashlib.md5(prompt.encode("utf-8")).hexdigest()[:10]
else:
    all_prompts = _load_prompts()
    prompt = all_prompts[0]
    prompt_idx = 0
    prompt_tag = "idx_000"

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

# 本仓库根目录，确保加载本地的 compression/hybrid_sd（不要使用其它路径下的旧副本）
_REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT))

from compression.hybrid_sd.inference_pipeline import HybridVideoInferencePipeline

OUTPUT_DIR = Path("/data/chenjiayu/minyu_lee/Hybrid-sd_wan/results/vbench/hybrid_wan2.1_14B_1.3B_complex_landscape/videos")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 模型配置
MODEL_PATHS = [
    "/data/chenjiayu/models/Wan2.1-T2V-14B-Diffusers",
    "/data/chenjiayu/models/Wan2.1-T2V-1.3B-Diffusers",
]
STAGE_STEPS = [30, 10, 10]

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

        self.stage_steps = STAGE_STEPS
        self.steps = STAGE_STEPS

        # ROI router（思路二）
        self.hybrid_ema_alpha = 0.85
        self.hybrid_relative_diff = True

        self.hybrid_temporal_top_ratio = 0.15
        self.hybrid_temporal_dilate = 1
        self.hybrid_max_segments = 2

        self.hybrid_spatial_top_ratio = 0.08
        self.hybrid_spatial_dilate = 1

        self.hybrid_spatial_blur_kernel = 9
        self.hybrid_spatial_peak_ratio = 0.60
        self.hybrid_spatial_cc_min_area = 12
        self.hybrid_spatial_min_bbox_ratio = 0.01
        self.hybrid_spatial_max_bbox_ratio = 0.55

        self.hybrid_projection_keep_ratio_h = 0.65
        self.hybrid_projection_keep_ratio_w = 0.65
        self.hybrid_projection_blur_kernel = 9

        self.hybrid_margin_t = 1
        self.hybrid_margin_h = 4
        self.hybrid_margin_w = 4

        self.hybrid_min_crop_t = 1
        self.hybrid_min_crop_h = 8
        self.hybrid_min_crop_w = 8

        self.hybrid_align_h = 2
        self.hybrid_align_w = 2

        self.hybrid_smooth_iou_thresh = 0.25
        self.hybrid_smooth_momentum = 0.6

        self.hybrid_debug_every = 1
        self.hybrid_debug_topk_frames = 5
        self.hybrid_debug_save_dir = str(
            OUTPUT_DIR.parent / "debug_roi" / f"{prompt_tag}_seed_{SEED}"
        )


def main():
    safe_prompt = prompt[:150].replace('/', '_').replace('\\', '_')
    output_path = OUTPUT_DIR / f"{safe_prompt}-{SEED}.mp4"

    if output_path.exists():
        print(f"[GPU{gpu_id}] 跳过已存在: {safe_prompt[:50]}...")
        return

    print("=" * 60)
    print(f"Hybrid SD Wan2.1 14B+1.3B - GPU {gpu_id}")
    print(f"Prompt (tag={prompt_tag}, idx={prompt_idx}): {prompt[:80]}...")
    print("=" * 60)
    print(f"云侧模型: {MODEL_PATHS[0]}")
    print(f"边缘模型: {MODEL_PATHS[1]}")
    print(f"三阶段步数配置: {STAGE_STEPS}")
    print(f"Seed: {SEED}")
    print(f"输出: {output_path}")
    args = Args()
    print(f"ROI debug 目录: {args.hybrid_debug_save_dir}")
    print()

    print("加载模型...")
    pipeline = HybridVideoInferencePipeline(
        weight_folders=MODEL_PATHS,
        seed=SEED,
        device="cuda:0",
        args=args,
    )
    pipeline.set_pipe_and_generator()
    print("模型加载完成")
    print()

    print("开始生成...")
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
