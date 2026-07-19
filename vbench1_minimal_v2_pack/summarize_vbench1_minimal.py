#!/usr/bin/env python3
"""Summarize V2 one-seed minimal VBench scores and generation timing."""

import argparse
import csv
import json
from pathlib import Path

CONFIGS = ["L30H10S10", "L30H20S0", "L35H15S0"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--results_root", default="/data/chenjiayu/hengyi_zhang/Hybrid_SD_for_wan2.1_v2/results/vbench1_minimal_v2_1seed")
    p.add_argument("--eval_root", default="/data/chenjiayu/hengyi_zhang/Hybrid_SD_for_wan2.1_v2/results/vbench1_minimal_v2_1seed_eval")
    p.add_argument("--output_md", default="/data/chenjiayu/hengyi_zhang/Hybrid_SD_for_wan2.1_v2/results/vbench1_minimal_v2_1seed/README_summary.md")
    p.add_argument("--configs", nargs="*", default=CONFIGS)
    return p.parse_args()


def scalar(value):
    if isinstance(value, list) and value:
        return value[0]
    return value


def read_scores(eval_root: Path, config: str):
    path = eval_root / config / f"{config}_minimal_eval_results.json"
    if not path.exists():
        return {}, path
    data = json.loads(path.read_text(encoding="utf-8"))
    return {key: scalar(value) for key, value in data.items()}, path


def read_times(results_root: Path, config: str):
    path = results_root / config / "generation_times.csv"
    rows_by_key = {}
    if not path.exists():
        return [], path
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            if row.get("status") != "ok" or row.get("timed") != "True":
                continue
            key = (row.get("config"), row.get("prompt_idx"), row.get("sample_idx"))
            rows_by_key[key] = row
    rows = sorted(rows_by_key.values(), key=lambda r: (r.get("config", ""), int(r.get("prompt_idx", 0)), int(r.get("sample_idx", 0))))
    return rows, path


def format_float(value, digits=4):
    if value in ("", None):
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except Exception:
        return str(value)


def main():
    args = parse_args()
    results_root = Path(args.results_root)
    eval_root = Path(args.eval_root)
    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    score_data = {}
    time_data = {}
    for config in args.configs:
        score_data[config], _ = read_scores(eval_root, config)
        time_data[config], _ = read_times(results_root, config)

    dims = []
    for config in args.configs:
        for dim in score_data[config]:
            if dim not in dims:
                dims.append(dim)

    lines = []
    lines.append("# V2 Minimal VBench One-Seed Summary")
    lines.append("")
    lines.append("## Settings")
    lines.append("")
    lines.append("- Prompts: 11 prompt minimal VBench subset")
    lines.append("- Videos: 1 seed/video per prompt")
    lines.append("- Stage configs: L30H10S10=[30,10,10], L30H20S0=[30,20,0], L35H15S0=[35,15,0]")
    lines.append("- Resolution/frames: 720x1280, 81 frames, 16 fps")
    lines.append("- Guidance: static CFG, guidance_scale=5.0")
    lines.append("- V2 router: temporal cue=frame_diff, spatial cue=cfg, max_cubes=2")
    lines.append("- Timing: formal generation runs only; sparse debug preview is excluded")
    lines.append("")

    lines.append("## VBench Scores")
    lines.append("")
    header = ["dimension"] + args.configs
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for dim in dims:
        cells = [dim]
        for config in args.configs:
            cells.append(format_float(score_data[config].get(dim)))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Generation Time")
    lines.append("")
    lines.append("| config | videos | total_sec | mean_sec | min_sec | max_sec |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: |")
    for config in args.configs:
        elapsed = [float(row["elapsed_sec"]) for row in time_data[config] if row.get("elapsed_sec")]
        if elapsed:
            lines.append(
                f"| {config} | {len(elapsed)} | {sum(elapsed):.1f} | "
                f"{sum(elapsed)/len(elapsed):.1f} | {min(elapsed):.1f} | {max(elapsed):.1f} |"
            )
        else:
            lines.append(f"| {config} | 0 |  |  |  |  |")
    lines.append("")

    lines.append("## Per-Video Time")
    lines.append("")
    lines.append("| config | prompt_idx | prompt | seed | elapsed_sec |")
    lines.append("| --- | ---: | --- | ---: | ---: |")
    for config in args.configs:
        for row in time_data[config]:
            prompt = row["prompt"].replace("|", "\\|")
            lines.append(f"| {config} | {row['prompt_idx']} | {prompt} | {row['seed']} | {float(row['elapsed_sec']):.1f} |")
    lines.append("")

    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(output_md)


if __name__ == "__main__":
    main()
