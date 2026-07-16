#!/usr/bin/env python3
"""
为多GPU生成metadata.csv，每个GPU有不同的prompt_id起始编号
"""

import csv
import yaml
from pathlib import Path

def main():
    # 读取配置
    config_base = Path("configs/vbench2_half_gpu4.yaml")
    with open(config_base) as f:
        cfg = yaml.safe_load(f)

    # 参数
    seed_list = cfg.get("seed_list", [1234])
    num_frames = cfg.get("num_frames", 81)
    resolution = cfg.get("resolution", {})
    height = resolution.get("height", 720)
    width = resolution.get("width", 1280)
    fps = cfg.get("fps", 16)

    # 输出目录
    output_dir = Path("results/vbench/default_exp")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 为每个GPU生成metadata
    gpu_configs = [
        (4, "VBench-master/VBench-2.0/prompts/VBench2_half_gpu4.txt", 0),
        (5, "VBench-master/VBench-2.0/prompts/VBench2_half_gpu5.txt", 153),
        (6, "VBench-master/VBench-2.0/prompts/VBench2_half_gpu6.txt", 306),
        (7, "VBench-master/VBench-2.0/prompts/VBench2_half_gpu7.txt", 459),
    ]

    all_rows = []

    for gpu_id, prompt_file, start_id in gpu_configs:
        print(f"处理 GPU {gpu_id}: {prompt_file}, 起始ID: {start_id}")

        # 读取prompts
        with open(prompt_file) as f:
            prompts = [line.strip() for line in f if line.strip()]

        print(f"  读取 {len(prompts)} 个prompts")

        # 生成rows
        for i, prompt in enumerate(prompts):
            prompt_id = start_id + i
            for seed in seed_list:
                row = [
                    f"{prompt_id:05d}",  # prompt_id
                    prompt,               # prompt
                    "",                   # negative_prompt
                    str(seed),            # seed
                    str(num_frames),      # num_frames
                    str(height),          # height
                    str(width),           # width
                    str(fps),             # fps
                ]
                all_rows.append(row)

    # 写入metadata.csv
    metadata_path = output_dir / "metadata.csv"
    with open(metadata_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["prompt_id", "prompt", "negative_prompt", "seed", "num_frames", "height", "width", "fps"])
        writer.writerows(all_rows)

    print(f"\n✓ 生成metadata.csv: {metadata_path}")
    print(f"  总行数: {len(all_rows)}")
    print(f"  Prompt ID范围: 00000-{len(all_rows)-1:05d}")

if __name__ == "__main__":
    main()
