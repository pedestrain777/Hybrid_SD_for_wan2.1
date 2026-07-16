#!/usr/bin/env python3
"""
Hybrid-SD (Wan2.2 14B + Wan2.1 1.3B) - VBench 1.0 generation with augmented prompts
Output: video/{dimension}/{original_prompt}-{seed}.mp4
Uses original short prompt for file naming, augmented prompt for generation.
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

MODEL_CONFIG = {
    "name": "Hybrid-SD (Wan2.2 A14B + Wan2.1 1.3B)",
    "models": [
        "/data/chenjiayu/models/Wan2.2-T2V-A14B-Diffusers",
        "/data/chenjiayu/models/Wan2.1-T2V-1.3B-Diffusers",
    ],
    "steps": [28, 12],  # 28 cloud + 12 edge = 40 total
}

GEN_PARAMS = {
    "num_frames": 81,
    "height": 720,
    "width": 1280,
    "guidance_scale": 5.0,
    "fps": 16,
}

PROMPT_DIR = Path("/data/chenjiayu/minyu_lee/A vbench 1.0/prompts")
AUG_PROMPT_DIR = Path("/data/chenjiayu/minyu_lee/A vbench 1.0/prompts_wan21_aug")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--dims", type=str, nargs="+", required=True)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    # Collect tasks
    all_tasks = []
    for dim in args.dims:
        orig_file = PROMPT_DIR / f"{dim}.txt"
        aug_file = AUG_PROMPT_DIR / f"{dim}.txt"
        if not orig_file.exists():
            print(f"[WARN] {orig_file} not found, skipping")
            continue
        if not aug_file.exists():
            print(f"[WARN] {aug_file} not found, skipping")
            continue

        with open(orig_file) as f:
            orig_prompts = [l.strip() for l in f if l.strip()]
        with open(aug_file) as f:
            aug_prompts = [l.strip() for l in f if l.strip()]

        assert len(orig_prompts) == len(aug_prompts), \
            f"{dim}: orig({len(orig_prompts)}) != aug({len(aug_prompts)})"

        video_dir = output_dir / "video" / dim
        video_dir.mkdir(parents=True, exist_ok=True)
        for orig_p, aug_p in zip(orig_prompts, aug_prompts):
            all_tasks.append((dim, orig_p, aug_p, video_dir))

    print(f"Total: {len(all_tasks)} videos across {len(args.dims)} dimensions")
    print(f"Seed: {args.seed}")
    print(f"Steps: {MODEL_CONFIG['steps']} (cloud:{MODEL_CONFIG['steps'][0]} + edge:{MODEL_CONFIG['steps'][1]})")
    print(f"Guidance: {GEN_PARAMS['guidance_scale']}")

    # Load pipeline
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

    print(f"Loading models: {MODEL_CONFIG['models']}")
    pipe = HybridVideoInferencePipeline(
        weight_folders=MODEL_CONFIG["models"],
        seed=args.seed,
        device="cuda:0",
        args=Args(),
    )
    pipe.set_pipe_and_generator()
    print("Models loaded")

    done = 0
    skipped = 0
    for i, (dim, orig_prompt, aug_prompt, video_dir) in enumerate(all_tasks):
        video_name = f"{orig_prompt}-{args.seed}.mp4"
        video_path = video_dir / video_name

        if video_path.exists():
            skipped += 1
            print(f"[{i+1}/{len(all_tasks)}] skip: {dim}/{orig_prompt[:50]}...")
            continue

        print(f"[{i+1}/{len(all_tasks)}] {dim}: {orig_prompt[:50]}...", flush=True)

        t0 = time.time()
        try:
            pipe.seed = args.seed
            pipe.generator = torch.Generator(device=pipe.device).manual_seed(args.seed)

            output = pipe.generate(
                prompt=aug_prompt,
                negative_prompt="",
                num_frames=GEN_PARAMS["num_frames"],
                height=GEN_PARAMS["height"],
                width=GEN_PARAMS["width"],
                guidance_scale=GEN_PARAMS["guidance_scale"],
                num_videos_per_prompt=1,
                use_dynamic_cfg=True,
                output_type="pil",
            )

            # Extract frames
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
            elapsed = time.time() - t0
            done += 1
            print(f"  done in {elapsed:.1f}s", flush=True)
        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            import traceback
            traceback.print_exc()
            continue

    print(f"\nFinished: {done} generated, {skipped} skipped")


if __name__ == "__main__":
    main()
