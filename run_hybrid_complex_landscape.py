#!/usr/bin/env python3
"""
Hybrid SD Wan2.1 14B+1.3B - Complex Landscape 生成

用法:
  python run_hybrid_complex_landscape.py <gpu_id>                    # 默认 prompts 第 0 行、默认步数
  python run_hybrid_complex_landscape.py <gpu_id> <prompt_idx>       # 文件第 prompt_idx 行
  python run_hybrid_complex_landscape.py <gpu_id> "黑色小狗在跑步"    # 自定义 prompt
  python run_hybrid_complex_landscape.py 0 --stages 40,10 "黑色小狗"  # 40 步 large + 0 hybrid + 10 small
  python run_hybrid_complex_landscape.py 0 --stages 30,0,20 "提示"   # 显式三段

输出 mp4 命名: {prompt}__{CONFIG_SLUG}.mp4
  CONFIG_SLUG 含 L?H?S?、seed、帧数、高x宽、guidance、fps；同路径重跑会覆盖，不再因「已存在」跳过。
"""

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

PROMPT_FILE = "/data/chenjiayu/minyu_lee/EC-Diff-main_for_v2i/prompts_complex_landscape.txt"

DEFAULT_STAGE_STEPS = [30, 10, 10]


def _load_prompts():
    with open(PROMPT_FILE, "r", encoding="utf-8", errors="replace") as f:
        return [line.strip() for line in f if line.strip()]


def _parse_stages_str(s: str) -> list:
    parts = [int(x.strip()) for x in s.split(",") if x.strip()]
    if len(parts) == 2:
        return [parts[0], 0, parts[1]]
    if len(parts) == 3:
        return parts
    raise SystemExit(
        f"--stages / 环境变量 格式错误: {s!r}，需要两个数 large,small（无 hybrid）"
        f"或三个数 large,hybrid,small，例如 40,10 或 30,10,10"
    )


def _resolve_stage_steps(cli_stages: str | None) -> list:
    if cli_stages:
        return _parse_stages_str(cli_stages)
    env = os.environ.get("WAN_HYBRID_STAGE_STEPS", "").strip()
    if env:
        return _parse_stages_str(env)
    return list(DEFAULT_STAGE_STEPS)


def _parse_cli():
    parser = argparse.ArgumentParser(
        description="Hybrid SD Wan2.1 14B+1.3B - Complex Landscape",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s 0
  %(prog)s 0 3
  %(prog)s 0 --stages 40,10 "黑色小狗在跑步"
  %(prog)s 0 "黑色小狗在跑步" --stages 40,10
  %(prog)s --stages 40,10 0 "黑色小狗在跑步"
  %(prog)s 0 --stages 30,20 "黑色小狗在跑步"
  %(prog)s 0 --stages 30,10,10 "黑色小狗在跑步"
  WAN_HYBRID_STAGE_STEPS=40,0,10 %(prog)s 0 "prompt"

说明: 使用 --stages 时，若把 prompt 写在 --stages 后面，须用 parse_intermixed_args（本脚本已启用），
      或写成「prompt 在前」如: %(prog)s 0 "提示" --stages 40,10
        """,
    )
    parser.add_argument(
        "gpu",
        nargs="?",
        default="0",
        help="物理 GPU id（写入 CUDA_VISIBLE_DEVICES）",
    )
    parser.add_argument(
        "--stages",
        default=None,
        metavar="L,S",
        help="两数: large,small（hybrid 固定为 0），如 40,10；三数: large,hybrid,small，如 30,10,10。也可用环境变量 WAN_HYBRID_STAGE_STEPS",
    )
    parser.add_argument(
        "prompt_args",
        nargs="*",
        help="省略=文件第 0 行；仅一个非负整数=该行 prompt；否则整句合并为自定义 prompt",
    )
    # 允许「gpu / prompt」与「--stages」任意穿插，例如: 0 --stages 40,10 "中文提示"
    if hasattr(parser, "parse_intermixed_args"):
        return parser.parse_intermixed_args()
    return parser.parse_args()


_ns = _parse_cli()
gpu_id = _ns.gpu
STAGE_STEPS = _resolve_stage_steps(_ns.stages)
_stage_slug = f"L{STAGE_STEPS[0]}H{STAGE_STEPS[1]}S{STAGE_STEPS[2]}"

prompt_idx = 0
prompt_tag = "idx_000"
pa = _ns.prompt_args

if len(pa) == 0:
    all_prompts = _load_prompts()
    prompt = all_prompts[0]
    prompt_idx = 0
    prompt_tag = "idx_000"
elif len(pa) == 1 and pa[0].isdigit():
    prompt_idx = int(pa[0])
    if prompt_idx < 0:
        raise SystemExit("prompt 行号必须 >= 0")
    all_prompts = _load_prompts()
    if prompt_idx >= len(all_prompts):
        raise SystemExit(f"prompt_idx={prompt_idx} 超出文件行数 {len(all_prompts)}")
    prompt = all_prompts[prompt_idx]
    prompt_tag = f"idx_{prompt_idx:03d}"
else:
    prompt = " ".join(pa)
    prompt_idx = -1
    prompt_tag = "custom_" + hashlib.md5(prompt.encode("utf-8")).hexdigest()[:10]

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

# 生成参数
NUM_FRAMES = 81
HEIGHT = 720
WIDTH = 1280
GUIDANCE_SCALE = 5.0
FPS = 16
SEED = 0


def _guidance_slug(gs: float) -> str:
    if float(gs).is_integer():
        return str(int(gs))
    return str(gs).replace(".", "p")


# 视频文件名与 debug 子目录共用：阶段 + 采样/画幅/引导/fps，避免同 prompt 不同配置互相覆盖
CONFIG_SLUG = (
    f"{_stage_slug}_seed{SEED}_f{NUM_FRAMES}_{HEIGHT}x{WIDTH}_"
    f"g{_guidance_slug(GUIDANCE_SCALE)}_fps{FPS}"
)


class Args:
    def __init__(self):
        self.enable_xformers_memory_efficient_attention = False
        self.use_dpm_solver = True
        self.logger = None

        self.stage_steps = STAGE_STEPS
        self.steps = STAGE_STEPS

        # ROI router（思路二）
        self.hybrid_ema_alpha = 0.85
        self.hybrid_aux_ema_alpha = 0.70
        self.hybrid_relative_diff = True

        self.hybrid_step_diff_weight = 1.0
        self.hybrid_cfg_gap_weight = 0.8
        self.hybrid_ls_gap_weight = 0.0
        self.hybrid_motion_weight = 0.5

        self.hybrid_ls_gap_every = 0
        self.hybrid_force_ls_gap_first = False
        self.hybrid_motion_blur_kernel = 3

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

        self.hybrid_max_rois_per_segment = 2
        self.hybrid_max_total_rois = 4
        self.hybrid_roi_nms_iou_thresh = 0.12

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

        self.hybrid_use_tube_spatial = True
        self.hybrid_tube_link_iou_thresh = 0.15
        self.hybrid_tube_debug_max_frames = 8

        self.hybrid_debug_every = 1
        self.hybrid_debug_topk_frames = 5
        self.hybrid_debug_save_dir = str(
            OUTPUT_DIR.parent / "debug_roi" / f"{prompt_tag}__{CONFIG_SLUG}"
        )


def main():
    safe_prompt = prompt[:150].replace("/", "_").replace("\\", "_")
    output_path = OUTPUT_DIR / f"{safe_prompt}__{CONFIG_SLUG}.mp4"

    print("=" * 60)
    print(f"Hybrid SD Wan2.1 14B+1.3B - GPU {gpu_id}")
    print(f"Prompt (tag={prompt_tag}, idx={prompt_idx}): {prompt[:80]}...")
    print("=" * 60)
    print(f"云侧模型: {MODEL_PATHS[0]}")
    print(f"边缘模型: {MODEL_PATHS[1]}")
    print(f"三阶段步数 [large, hybrid, small]: {STAGE_STEPS}  (slug={_stage_slug})")
    print(f"配置标签 CONFIG_SLUG: {CONFIG_SLUG}")
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
    if hasattr(output, "frames"):
        frames = output.frames[0] if isinstance(output.frames, list) else output.frames
    elif isinstance(output, list):
        frames = output[0] if len(output) > 0 and isinstance(output[0], list) else output
    else:
        frames = output

    export_to_video(frames, str(output_path), fps=FPS)
    print(f"视频已保存: {output_path}")


if __name__ == "__main__":
    main()
