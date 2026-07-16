#!/usr/bin/env python3
"""
为 VBench2 sampled314 生成 metadata.csv
"""

import csv
import yaml
from pathlib import Path

def main():
    # 参数
    seed_list = [1234]
    num_frames = 81
    height = 720
    width = 1280
    fps = 16

    # 输出目录
    output_dir = Path("/data/chenjiayu/minyu_lee/hybrid_wan_14B_1.3B/results/vbench/ec_diff_wan2.2_14B_1.3B_sampled314")
    output_dir.mkdir(parents=True, exist_ok=True)

    # GPU 配置
    gpu_configs = [
        (2, "VBench-master/VBench-2.0/prompts/VBench2_sampled314_gpu2.txt", 0),
        (3, "VBench-master/VBench-2.0/prompts/VBench2_sampled314_gpu3.txt", 53),
        (4, "VBench-master/VBench-2.0/prompts/VBench2_sampled314_gpu4.txt", 106),
        (5, "VBench-master/VBench-2.0/prompts/VBench2_sampled314_gpu5.txt", 158),
        (6, "VBench-master/VBench-2.0/prompts/VBench2_sampled314_gpu6.txt", 210),
        (7, "VBench-master/VBench-2.0/prompts/VBench2_sampled314_gpu7.txt", 262),
    ]

    all_rows = []

    for gpu_id, prompt_file, start_id in gpu_configs:
        prompt_path = Path("/data/chenjiayu/minyu_lee/hybrid_wan_14B_1.3B") / prompt_file
        print(f"处理 GPU {gpu_id}: {prompt_file}, 起始ID: {start_id}")

        # 读取prompts
        with open(prompt_path) as f:
            prompts = [line.strip() for line in f if line.strip()]

        print(f"  读取 {len(prompts)} 个prompts")

        # 生成rows
        for i, prompt in enumerate(prompts):
            prompt_id = start_id + i
            for seed in seed_list:
                video_name = f"{prompt}-{seed}.mp4"
                all_rows.append({
                    "prompt_id": f"prompt_{prompt_id:05d}",
                    "prompt": prompt,
                    "seed": seed,
                    "video_path": video_name,
                    "num_frames": num_frames,
                    "height": height,
                    "width": width,
                    "fps": fps,
                })

    # 写入CSV
    csv_path = output_dir / "metadata.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["prompt_id", "prompt", "seed", "video_path", "num_frames", "height", "width", "fps"])
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"\n生成 metadata.csv: {csv_path}")
    print(f"总行数: {len(all_rows)}")

    # 创建 videos 目录
    videos_dir = output_dir / "videos"
    videos_dir.mkdir(exist_ok=True)
    print(f"创建 videos 目录: {videos_dir}")

if __name__ == "__main__":
    main()
