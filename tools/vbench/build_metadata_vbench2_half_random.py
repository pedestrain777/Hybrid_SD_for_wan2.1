#!/usr/bin/env python3
"""
为VBench 2.0生成metadata.csv，每个维度随机取一半，保持原始编号
"""

import csv
import json
import random
import yaml
from pathlib import Path

def main():
    # 设置随机种子，保证可复现
    random.seed(42)

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

    # 读取VBench2完整的prompt列表（这个文件的行号就是原始编号）
    full_text_file = Path("VBench-master/VBench-2.0/prompts/VBench2_full_text.txt")
    with open(full_text_file) as f:
        full_prompts = [line.strip() for line in f if line.strip()]

    print(f"VBench 2.0总prompts: {len(full_prompts)}")

    # 读取维度信息
    with open("VBench-master/VBench-2.0/prompts/VBench2_full_text_info.json") as f:
        info = json.load(f)

    # 为每个prompt建立索引
    prompt_to_id = {prompt: idx for idx, prompt in enumerate(full_prompts)}

    # 读取各维度的prompts
    meta_info_dir = Path("VBench-master/VBench-2.0/prompts/meta_info")
    dimensions = [
        "Camera_Motion", "Complex_Landscape", "Complex_Plot", "Composition",
        "Diversity", "Dynamic_Attribute", "Dynamic_Spatial_Relationship",
        "Human_Anatomy", "Human_Clothes", "Human_Identity", "Human_Interaction",
        "Instance_Preservation", "Material", "Mechanics", "Motion_Order_Understanding",
        "Motion_Rationality", "Multi-View_Consistency", "Thermotics"
    ]

    selected_prompts = []

    for dim in dimensions:
        json_file = meta_info_dir / f"{dim}.json"
        with open(json_file) as f:
            dim_prompts = json.load(f)

        # 随机取一半
        total = len(dim_prompts)
        half_count = total // 2
        selected_indices = random.sample(range(total), half_count)
        selected_indices.sort()  # 排序，保持一定顺序

        print(f"{dim}: {total} prompts, 随机选择 {half_count} 个")

        for idx in selected_indices:
            item = dim_prompts[idx]
            prompt_text = item["prompt_en"]

            # 找到这个prompt在VBench2_full_text.txt中的原始编号
            if prompt_text in prompt_to_id:
                original_id = prompt_to_id[prompt_text]
                selected_prompts.append({
                    "prompt_id": f"{original_id:05d}",
                    "prompt": prompt_text,
                    "dimension": dim,
                })
            else:
                print(f"  警告: 未找到prompt: {prompt_text[:50]}")

    print(f"\n总共选择: {len(selected_prompts)} 个prompts")

    # 按prompt_id排序（可选，也可以保持维度顺序）
    selected_prompts.sort(key=lambda x: int(x["prompt_id"]))

    # 生成metadata.csv
    output_dir = Path("results/vbench/default_exp")
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for item in selected_prompts:
        for seed in seed_list:
            row = [
                item["prompt_id"],
                item["prompt"],
                "",  # negative_prompt
                str(seed),
                str(num_frames),
                str(height),
                str(width),
                str(fps),
            ]
            rows.append(row)

    metadata_path = output_dir / "metadata.csv"
    with open(metadata_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["prompt_id", "prompt", "negative_prompt", "seed", "num_frames", "height", "width", "fps"])
        writer.writerows(rows)

    print(f"\n✓ 生成metadata.csv: {metadata_path}")
    print(f"  总行数: {len(rows)}")
    print(f"  Prompt ID范围: {selected_prompts[0]['prompt_id']} - {selected_prompts[-1]['prompt_id']} (不连续)")

    # 保存选择的prompt列表（用于后续参考）
    selected_list_path = output_dir / "selected_prompts.json"
    with open(selected_list_path, 'w', encoding='utf-8') as f:
        json.dump(selected_prompts, f, indent=2, ensure_ascii=False)

    print(f"✓ 保存选择列表: {selected_list_path}")

if __name__ == "__main__":
    main()
