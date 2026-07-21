#!/usr/bin/env python3
"""Select the fastest threshold within 1% of the best calibration VBench mean."""

import argparse
import csv
import json
from pathlib import Path


THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30]
DIMENSIONS = [
    "subject_consistency", "temporal_flickering", "motion_smoothness", "dynamic_degree",
    "multiple_objects", "spatial_relationship", "human_action",
]


def threshold_tag(value: float) -> str:
    return f"tau_{value:.2f}".replace(".", "p")


def scalar(value):
    return value[0] if isinstance(value, list) and value else value


def read_scores(root: Path, config: str):
    path = root / config / "vbench_eval" / f"{config}_eval_results.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return {dimension: float(scalar(data[dimension])) for dimension in DIMENSIONS if dimension in data}


def read_timings(root: Path, config: str):
    path = root / config / "generation_times.csv"
    rows = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("status") == "ok":
                rows[int(row["prompt_idx"])] = row
    return list(rows.values())


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        default="/data/chenjiayu/hengyi_zhang/Hybrid_SD_for_wan2.1_v2/results/threshold_ablation_vbench1_20p",
    )
    parser.add_argument("--quality-tolerance", type=float, default=0.01)
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(args.results_root)
    records = []
    for threshold in THRESHOLDS:
        config = threshold_tag(threshold)
        scores = read_scores(root, config)
        timings = read_timings(root, config)
        quality = sum(scores.values()) / len(scores)
        generation_times = [float(row["generation_sec"]) for row in timings]
        switch_steps = [int(row["switch_step"]) for row in timings if row.get("switch_step")]
        records.append({
            "threshold": threshold,
            "config": config,
            "scores": scores,
            "calibration_vbench_mean": quality,
            "videos": len(generation_times),
            "mean_generation_sec": sum(generation_times) / len(generation_times),
            "mean_switch_step": sum(switch_steps) / len(switch_steps),
        })

    best_quality = max(row["calibration_vbench_mean"] for row in records)
    quality_floor = best_quality * (1.0 - args.quality_tolerance)
    eligible = [row for row in records if row["calibration_vbench_mean"] >= quality_floor]
    selected = min(eligible, key=lambda row: row["mean_generation_sec"])
    baseline_time = next(
        row["mean_generation_sec"] for row in records if row["threshold"] == THRESHOLDS[0]
    )
    for row in records:
        row["relative_quality_drop"] = (best_quality - row["calibration_vbench_mean"]) / best_quality
        row["speedup_vs_tau_0p10"] = baseline_time / row["mean_generation_sec"]
        row["eligible_within_1pct"] = row in eligible

    result = {
        "selection_rule": "fastest threshold within 1% relative of the best seven-dimension calibration VBench mean",
        "quality_tolerance": args.quality_tolerance,
        "best_quality": best_quality,
        "quality_floor": quality_floor,
        "selected_threshold": selected["threshold"],
        "records": records,
    }
    json_path = root / "threshold_selection.json"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Dynamic Switch Threshold Ablation",
        "",
        "Selection rule: choose the fastest threshold whose seven-dimension calibration VBench mean is within 1% relative of the best value.",
        "",
        "| tau | VBench mean | quality drop | generation sec | speedup vs 0.10 | mean switch | eligible |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | :---: |",
    ]
    for row in records:
        lines.append(
            f"| {row['threshold']:.2f} | {row['calibration_vbench_mean']:.4f} | "
            f"{100 * row['relative_quality_drop']:.2f}% | {row['mean_generation_sec']:.1f} | "
            f"{row['speedup_vs_tau_0p10']:.3f}x | {row['mean_switch_step']:.1f} | "
            f"{'yes' if row['eligible_within_1pct'] else 'no'} |"
        )
    lines.extend(["", f"Selected threshold: **{selected['threshold']:.2f}**", ""])
    markdown_path = root / "README_threshold_ablation.md"
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    print(markdown_path)
    print(json_path)
    print(f"SELECTED_THRESHOLD={selected['threshold']:.2f}")


if __name__ == "__main__":
    main()
