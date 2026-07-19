#!/usr/bin/env python3
"""
Generate the V2 one-seed minimal VBench-1.0 subset for Hybrid Wan2.1.

This is an internal ablation run, not an official VBench submission layout:
each prompt has one generated video, named with the standard sample-0 suffix.
"""

import argparse
import csv
import os
import re
import sys
import time
import traceback
from pathlib import Path

import torch

PROMPTS = [
    "a person swimming in ocean",
    "Close up of grapes on a rotating table.",
    "alley",
    "In a still frame, a stop sign",
    "a person",
    "a bird and a cat",
    "A person is riding a bike",
    "a red bicycle",
    "a bicycle on the left of a car, front view",
    "A beautiful coastal beach in spring, waves lapping on sand, in super slow motion",
    "A beautiful coastal beach in spring, waves lapping on sand, Van Gogh style",
]

TEMPORAL_FLICKERING_PROMPT = "In a still frame, a stop sign"

STAGE_CONFIGS = {
    "L50H0": [50, 0],
    "DYN28_38": [30, 20],
    "L50H0S0": [50, 0, 0],
    "L30H10S10": [30, 10, 10],
    "L30H20S0": [30, 20, 0],
    "L35H15S0": [35, 15, 0],
}

DIM_COVERAGE = {
    "a person swimming in ocean": ["subject_consistency", "dynamic_degree", "motion_smoothness"],
    "Close up of grapes on a rotating table.": ["overall_consistency", "aesthetic_quality", "imaging_quality"],
    "alley": ["scene", "background_consistency"],
    "In a still frame, a stop sign": ["temporal_flickering"],
    "a person": ["object_class"],
    "a bird and a cat": ["multiple_objects"],
    "A person is riding a bike": ["human_action"],
    "a red bicycle": ["color"],
    "a bicycle on the left of a car, front view": ["spatial_relationship"],
    "A beautiful coastal beach in spring, waves lapping on sand, in super slow motion": ["temporal_style"],
    "A beautiful coastal beach in spring, waves lapping on sand, Van Gogh style": ["appearance_style"],
}

CSV_FIELDS = [
    "status",
    "config",
    "stage_steps",
    "gpu",
    "prompt_idx",
    "prompt",
    "sample_idx",
    "seed",
    "height",
    "width",
    "num_frames",
    "fps",
    "guidance_scale",
    "dynamic_cfg",
    "switch_step",
    "debug",
    "timed",
    "elapsed_sec",
    "output_path",
    "error",
]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--repo_root", default="/data/chenjiayu/hengyi_zhang/Hybrid_SD_for_wan2.1_v2")
    p.add_argument("--model_large", default="/data/chenjiayu/models/Wan2.1-T2V-14B-Diffusers")
    p.add_argument("--model_small", default="/data/chenjiayu/models/Wan2.1-T2V-1.3B-Diffusers")
    p.add_argument("--output_root", default="/data/chenjiayu/hengyi_zhang/Hybrid_SD_for_wan2.1_v2/results/vbench1_minimal_v2_1seed")
    p.add_argument("--gpu", default="0")
    p.add_argument("--base_seed", type=int, default=0)
    p.add_argument("--samples_per_prompt", type=int, default=1)
    p.add_argument("--temporal_samples", type=int, default=1)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--num_frames", type=int, default=81)
    p.add_argument("--fps", type=int, default=16)
    p.add_argument("--guidance_scale", type=float, default=5.0)
    p.add_argument("--configs", nargs="*", default=list(STAGE_CONFIGS.keys()), choices=list(STAGE_CONFIGS.keys()))
    p.add_argument("--prompt_indices", nargs="*", type=int, default=None,
                   help="Optional 0-based prompt indices to run. Default: all prompts.")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--debug", action="store_true", help="save sparse ROI debug images")
    p.add_argument("--debug_every", type=int, default=5)
    p.add_argument("--debug_topk_frames", type=int, default=3)
    p.add_argument("--debug_save_all_cues", action="store_true")
    p.add_argument("--no_timing", action="store_true", help="do not write per-video timing rows")
    p.add_argument("--time_log", default=None)
    return p.parse_args()


def safe_debug_name(prompt: str) -> str:
    name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", prompt.strip())[:80]
    return name or "prompt"


def selected_prompts(indices):
    if indices is None:
        return list(enumerate(PROMPTS))
    invalid = [idx for idx in indices if idx < 0 or idx >= len(PROMPTS)]
    if invalid:
        raise ValueError(f"Invalid prompt indices: {invalid}")
    return [(idx, PROMPTS[idx]) for idx in indices]


def append_csv(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


class Args:
    def __init__(self, config_name, stage_steps, debug_dir=None, debug_every=5, debug_topk_frames=3, debug_save_all_cues=False):
        self.enable_xformers_memory_efficient_attention = False
        self.use_dpm_solver = True
        self.logger = None
        self.stage_steps = stage_steps
        self.steps = stage_steps

        # V2 simplified two-stage cube router.
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
        self.hybrid_dynamic_switch = config_name == "DYN28_38"
        self.hybrid_dynamic_switch_min_step = 28
        self.hybrid_dynamic_switch_max_step = 38
        self.hybrid_dynamic_switch_threshold = 0.20
        self.hybrid_dynamic_switch_patience = 2

        self.hybrid_debug_every = debug_every
        self.hybrid_debug_topk_frames = debug_topk_frames
        self.hybrid_debug_save_all_cues = debug_save_all_cues
        self.hybrid_debug_save_dir = debug_dir


def extract_frames(out):
    if hasattr(out, "frames"):
        return out.frames[0] if isinstance(out.frames, list) else out.frames
    if isinstance(out, list):
        return out[0] if len(out) > 0 and isinstance(out[0], list) else out
    return out


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    repo_root = Path(args.repo_root).resolve()
    sys.path.insert(0, str(repo_root))

    from compression.hybrid_sd.inference_pipeline import HybridVideoInferencePipeline
    from diffusers.utils import export_to_video

    output_root = Path(args.output_root)
    model_paths = [args.model_large, args.model_small]
    prompt_items = selected_prompts(args.prompt_indices)

    print("=" * 80)
    print("V2 VBench1.0 minimal one-seed generation")
    print(f"prompts={len(prompt_items)}/{len(PROMPTS)} samples_per_prompt={args.samples_per_prompt} temporal_samples={args.temporal_samples}")
    print(f"GPU: {args.gpu} | base_seed: {args.base_seed} | debug={args.debug} | timed={not args.no_timing}")
    print(f"Output root: {output_root}")
    print(f"Configs: {args.configs}")
    print("=" * 80)

    for config_name in args.configs:
        stage_steps = STAGE_CONFIGS[config_name]
        videos_dir = output_root / config_name / "videos"
        debug_root = output_root / config_name / "debug_roi"
        videos_dir.mkdir(parents=True, exist_ok=True)
        if args.debug:
            debug_root.mkdir(parents=True, exist_ok=True)

        time_log = Path(args.time_log) if args.time_log else output_root / config_name / "generation_times.csv"

        print(f"\n[{config_name}] stage_steps={stage_steps}: loading pipeline...")
        pipe_args = Args(
            config_name,
            stage_steps,
            debug_dir=str(debug_root) if args.debug else None,
            debug_every=args.debug_every,
            debug_topk_frames=args.debug_topk_frames,
            debug_save_all_cues=args.debug_save_all_cues,
        )
        pipe = HybridVideoInferencePipeline(
            weight_folders=model_paths,
            seed=args.base_seed,
            device="cuda:0",
            args=pipe_args,
        )
        pipe.set_pipe_and_generator()
        print(f"[{config_name}] pipeline ready")

        for prompt_idx, prompt in prompt_items:
            n_samples = args.temporal_samples if prompt == TEMPORAL_FLICKERING_PROMPT else args.samples_per_prompt
            for sample_idx in range(n_samples):
                output_path = videos_dir / f"{prompt}-{sample_idx}.mp4"
                seed = args.base_seed + prompt_idx * 1000 + sample_idx
                row = {
                    "config": config_name,
                    "stage_steps": "-".join(str(x) for x in stage_steps),
                    "gpu": args.gpu,
                    "prompt_idx": prompt_idx,
                    "prompt": prompt,
                    "sample_idx": sample_idx,
                    "seed": seed,
                    "height": args.height,
                    "width": args.width,
                    "num_frames": args.num_frames,
                    "fps": args.fps,
                    "guidance_scale": args.guidance_scale,
                    "dynamic_cfg": False,
                    "debug": args.debug,
                    "timed": not args.no_timing,
                    "output_path": str(output_path),
                }
                if output_path.exists() and not args.overwrite:
                    print(f"[{config_name}] skip existing: {output_path.name}")
                    row.update({"status": "skipped_existing", "elapsed_sec": 0.0})
                    if not args.no_timing:
                        append_csv(time_log, row)
                    continue

                pipe.generator = torch.Generator(device=pipe.device).manual_seed(seed)
                if args.debug:
                    pipe.pipe.set_hybrid_roi_config({
                        **pipe.pipe.hybrid_roi_config,
                        "save_debug_dir": str(debug_root / safe_debug_name(prompt) / f"sample_{sample_idx}"),
                        "debug_every": args.debug_every,
                        "debug_topk_frames": args.debug_topk_frames,
                        "debug_save_all_cues": args.debug_save_all_cues,
                    })

                print(f"[{config_name}] prompt={prompt_idx+1:02d}/{len(PROMPTS)} sample={sample_idx:02d} seed={seed}")
                print(f"    {prompt}")
                print(f"    covers: {', '.join(DIM_COVERAGE[prompt])}")
                t0 = time.time()
                try:
                    out = pipe.generate(
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
                    frames = extract_frames(out)
                    export_to_video(frames, str(output_path), fps=args.fps)
                    elapsed = time.time() - t0
                    row.update({
                        "status": "ok",
                        "elapsed_sec": f"{elapsed:.3f}",
                        "switch_step": pipe.pipe.dynamic_switch_step,
                    })
                    print(f"    saved: {output_path} | time={elapsed:.1f}s")
                except Exception as exc:
                    elapsed = time.time() - t0
                    row.update({
                        "status": "error",
                        "elapsed_sec": f"{elapsed:.3f}",
                        "error": repr(exc),
                    })
                    print(traceback.format_exc())
                    if not args.no_timing:
                        append_csv(time_log, row)
                    raise

                if not args.no_timing:
                    append_csv(time_log, row)

        pipe.clear()

    print("\nDone.")


if __name__ == "__main__":
    main()
