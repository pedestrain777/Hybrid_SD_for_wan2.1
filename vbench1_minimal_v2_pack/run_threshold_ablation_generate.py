#!/usr/bin/env python3
"""Generate one-seed VBench-1.0 calibration videos for one switch threshold."""

import argparse
import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path

import torch


CSV_FIELDS = [
    "status", "config", "threshold", "gpu", "prompt_idx", "category", "prompt",
    "seed", "height", "width", "num_frames", "fps", "guidance_scale",
    "generation_sec", "export_sec", "switch_step", "output_path", "error",
]


def threshold_tag(value: float) -> str:
    return f"tau_{value:.2f}".replace(".", "p")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument(
        "--repo-root",
        default="/data/chenjiayu/hengyi_zhang/Hybrid_SD_for_wan2.1_v2",
    )
    parser.add_argument(
        "--output-root",
        default="/data/chenjiayu/hengyi_zhang/Hybrid_SD_for_wan2.1_v2/results/threshold_ablation_vbench1_20p",
    )
    parser.add_argument("--model-large", default="/data/chenjiayu/models/Wan2.1-T2V-14B-Diffusers")
    parser.add_argument("--model-small", default="/data/chenjiayu/models/Wan2.1-T2V-1.3B-Diffusers")
    parser.add_argument("--base-seed", type=int, default=0)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--num-frames", type=int, default=81)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--guidance-scale", type=float, default=5.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def append_csv(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


def successful_prompt_indices(path: Path):
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {
            int(row["prompt_idx"])
            for row in csv.DictReader(handle)
            if row.get("status") == "ok"
        }


def extract_frames(output):
    if hasattr(output, "frames"):
        return output.frames[0] if isinstance(output.frames, list) else output.frames
    if isinstance(output, list):
        return output[0] if output and isinstance(output[0], list) else output
    return output


class PipelineArgs:
    def __init__(self, threshold: float):
        self.enable_xformers_memory_efficient_attention = False
        self.use_dpm_solver = True
        self.logger = None
        self.stage_steps = [30, 20]
        self.steps = self.stage_steps

        self.hybrid_routing_mode = "two_stage_cube"
        self.hybrid_temporal_cue = "frame_diff"
        self.hybrid_temporal_top_ratio = 0.15
        self.hybrid_max_temporal_segments = 2
        self.hybrid_spatial_cue = "cfg"
        self.hybrid_spatial_top_ratio = 0.08
        self.hybrid_max_cubes = 2
        self.hybrid_warp_max_shift = 2

        self.hybrid_margin_t = 1
        self.hybrid_margin_h = 4
        self.hybrid_margin_w = 4
        self.hybrid_min_crop_t = 1
        self.hybrid_min_crop_h = 8
        self.hybrid_min_crop_w = 8
        self.hybrid_align_h = 2
        self.hybrid_align_w = 2
        self.hybrid_position_aware_rope = True
        self.hybrid_fusion_mode = "feather"
        self.hybrid_feather_t = 1
        self.hybrid_feather_h = 2
        self.hybrid_feather_w = 2

        self.hybrid_dynamic_switch = True
        self.hybrid_dynamic_switch_min_step = 28
        self.hybrid_dynamic_switch_max_step = 38
        self.hybrid_dynamic_switch_threshold = threshold
        self.hybrid_dynamic_switch_patience = 2

        self.hybrid_debug_every = 1
        self.hybrid_debug_topk_frames = 5
        self.hybrid_debug_log = False
        self.hybrid_debug_save_all_cues = False
        self.hybrid_debug_save_dir = None


def main():
    args = parse_args()
    if args.threshold <= 0:
        raise SystemExit("--threshold must be positive")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    repo_root = Path(args.repo_root).resolve()
    sys.path.insert(0, str(repo_root))

    manifest_path = repo_root / "vbench1_minimal_v2_pack" / "threshold_ablation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = threshold_tag(args.threshold)
    config_root = Path(args.output_root) / config
    videos_dir = config_root / "videos"
    videos_dir.mkdir(parents=True, exist_ok=True)
    timing_csv = config_root / "generation_times.csv"
    completed = successful_prompt_indices(timing_csv) if not args.overwrite else set()

    from compression.hybrid_sd.inference_pipeline import HybridVideoInferencePipeline
    from diffusers.utils import export_to_video

    print(f"[{config}] GPU={args.gpu}; prompts={len(manifest)}; output={config_root}")
    pipeline = HybridVideoInferencePipeline(
        weight_folders=[args.model_large, args.model_small],
        seed=args.base_seed,
        device="cuda:0",
        args=PipelineArgs(args.threshold),
    )
    pipeline.set_pipe_and_generator()

    for prompt_idx, item in enumerate(manifest):
        prompt = item["prompt"]
        output_path = videos_dir / f"{prompt}-0.mp4"
        if prompt_idx in completed and output_path.exists() and not args.overwrite:
            print(f"[{config}] skip completed {prompt_idx + 1:02d}/{len(manifest)}")
            continue

        seed = args.base_seed + prompt_idx * 1000
        pipeline.generator = torch.Generator(device=pipeline.device).manual_seed(seed)
        row = {
            "config": config,
            "threshold": f"{args.threshold:.2f}",
            "gpu": args.gpu,
            "prompt_idx": prompt_idx,
            "category": item["category"],
            "prompt": prompt,
            "seed": seed,
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "fps": args.fps,
            "guidance_scale": args.guidance_scale,
            "output_path": str(output_path),
        }
        print(f"[{config}] {prompt_idx + 1:02d}/{len(manifest)} seed={seed}: {prompt}")
        try:
            torch.cuda.synchronize()
            started = time.perf_counter()
            output = pipeline.generate(
                prompt=prompt,
                negative_prompt="",
                num_frames=args.num_frames,
                height=args.height,
                width=args.width,
                guidance_scale=args.guidance_scale,
                num_videos_per_prompt=1,
                use_dynamic_cfg=False,
                output_type="pil",
            )
            torch.cuda.synchronize()
            generation_sec = time.perf_counter() - started

            export_started = time.perf_counter()
            export_to_video(extract_frames(output), str(output_path), fps=args.fps)
            export_sec = time.perf_counter() - export_started
            row.update({
                "status": "ok",
                "generation_sec": f"{generation_sec:.3f}",
                "export_sec": f"{export_sec:.3f}",
                "switch_step": pipeline.pipe.dynamic_switch_step,
            })
            print(
                f"[{config}] saved; generation={generation_sec:.1f}s; "
                f"switch={pipeline.pipe.dynamic_switch_step}"
            )
        except Exception as exc:
            row.update({"status": "error", "error": repr(exc)})
            print(traceback.format_exc())
            append_csv(timing_csv, row)
            raise
        append_csv(timing_csv, row)

    pipeline.clear()
    print(f"[{config}] done")


if __name__ == "__main__":
    main()
