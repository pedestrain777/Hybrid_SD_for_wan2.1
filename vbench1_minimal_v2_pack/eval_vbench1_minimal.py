#!/usr/bin/env python3
"""
Evaluate the VBench-1.0 minimal subset.

This is an internal ablation helper. It uses a reduced full_info JSON, so the output is
not a complete official VBench score.
"""

import argparse
import sys
from pathlib import Path

DIMENSIONS = [
    "subject_consistency", "background_consistency", "temporal_flickering",
    "motion_smoothness", "dynamic_degree", "aesthetic_quality", "imaging_quality",
    "object_class", "multiple_objects", "human_action", "color",
    "spatial_relationship", "scene", "temporal_style", "appearance_style",
    "overall_consistency",
]

CONFIGS = ["L50H0", "DYN28_38"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--vbench_root", required=True, help="Path to VBench-master")
    p.add_argument("--videos_root", default="/data/chenjiayu/hengyi_zhang/Hybrid_SD_for_wan2.1_v2/results/vbench1_minimal_v2_1seed")
    p.add_argument("--mini_full_info", default="/data/chenjiayu/hengyi_zhang/Hybrid_SD_for_wan2.1_v2/vbench1_minimal_v2_pack/mini_vbench1_full_info.json")
    p.add_argument("--output_root", default="/data/chenjiayu/hengyi_zhang/Hybrid_SD_for_wan2.1_v2/results/vbench1_minimal_v2_1seed_eval")
    p.add_argument("--device", default="cuda")
    p.add_argument("--configs", nargs="*", default=CONFIGS, choices=CONFIGS)
    p.add_argument("--dimensions", nargs="*", default=DIMENSIONS)
    p.add_argument("--local", action="store_true")
    p.add_argument("--read_frame", action="store_true")
    p.add_argument("--mode", default="vbench_standard", choices=["vbench_standard", "custom_input"])
    return p.parse_args()


def main():
    args = parse_args()
    vbench_root = Path(args.vbench_root).resolve()
    sys.path.insert(0, str(vbench_root))

    import torch
    from vbench import VBench

    mini_info = Path(args.mini_full_info).resolve()
    videos_root = Path(args.videos_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    print("=" * 80)
    print("VBench1.0 minimal subset evaluation")
    print(f"mini_full_info: {mini_info}")
    print(f"videos_root: {videos_root}")
    print(f"output_root: {output_root}")
    print("dimensions:", " ".join(args.dimensions))
    print("=" * 80)

    for config in args.configs:
        videos_path = videos_root / config / "videos"
        out_dir = output_root / config
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[{config}] evaluating videos_path={videos_path}")
        bench = VBench(device, str(mini_info), str(out_dir))
        bench.evaluate(
            videos_path=str(videos_path),
            name=f"{config}_minimal",
            dimension_list=args.dimensions,
            local=args.local,
            read_frame=args.read_frame,
            mode=args.mode,
            imaging_quality_preprocessing_mode="longer",
        )


if __name__ == "__main__":
    main()
