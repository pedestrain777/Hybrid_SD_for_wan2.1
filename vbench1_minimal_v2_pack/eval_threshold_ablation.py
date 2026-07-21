#!/usr/bin/env python3
"""Evaluate threshold-ablation videos on their VBench-1.0 dimensions."""

import argparse
import json
import sys
from pathlib import Path


DIMENSIONS = [
    "subject_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "dynamic_degree",
    "multiple_objects",
    "spatial_relationship",
    "human_action",
]


def threshold_tag(value: float) -> str:
    return f"tau_{value:.2f}".replace(".", "p")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--vbench-root", default="/data/chenjiayu/hengyi_zhang/VBench")
    parser.add_argument(
        "--repo-root",
        default="/data/chenjiayu/hengyi_zhang/Hybrid_SD_for_wan2.1_v2",
    )
    parser.add_argument(
        "--results-root",
        default="/data/chenjiayu/hengyi_zhang/Hybrid_SD_for_wan2.1_v2/results/threshold_ablation_vbench1_20p",
    )
    return parser.parse_args()


def build_subset(full_info_path: Path, manifest_path: Path, output_path: Path):
    full_info = json.loads(full_info_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    by_prompt = {item["prompt_en"]: item for item in full_info}
    missing = [item["prompt"] for item in manifest if item["prompt"] not in by_prompt]
    if missing:
        raise ValueError(f"Prompts missing from VBench_full_info.json: {missing}")
    subset = [by_prompt[item["prompt"]] for item in manifest]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(subset, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main():
    args = parse_args()
    repo_root = Path(args.repo_root)
    vbench_root = Path(args.vbench_root)
    results_root = Path(args.results_root)
    config = threshold_tag(args.threshold)
    config_root = results_root / config
    output_root = config_root / "vbench_eval"
    subset_path = results_root / "threshold_ablation_full_info.json"

    build_subset(
        vbench_root / "vbench" / "VBench_full_info.json",
        repo_root / "vbench1_minimal_v2_pack" / "threshold_ablation_manifest.json",
        subset_path,
    )
    sys.path.insert(0, str(vbench_root))
    import torch
    from vbench import VBench

    output_root.mkdir(parents=True, exist_ok=True)
    benchmark = VBench(torch.device(args.device), str(subset_path), str(output_root))
    benchmark.evaluate(
        videos_path=str(config_root / "videos"),
        name=config,
        dimension_list=DIMENSIONS,
        local=False,
        read_frame=False,
        mode="vbench_standard",
        imaging_quality_preprocessing_mode="longer",
    )
    print(f"[{config}] evaluation done: {output_root}")


if __name__ == "__main__":
    main()
