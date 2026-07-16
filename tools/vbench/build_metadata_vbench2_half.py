#!/usr/bin/env python3
"""
为VBench 2.0生成metadata.csv，保持维度信息和原始顺序
每个维度取一半prompts，4张GPU并行处理
"""

import csv
import json
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

    # 读取所有维度的prompts
    meta_info_dir = Path("VBench-master/VBench-2.0/prompts/meta_info")
    dimensions = [
        "Camera_Motion", "Complex_Landscape", "Complex_Plot", "Composition",
        "Diversity", "Dynamic_Attribute", "Dynamic_Spatial_Relationship",
        "Human_Anatomy", "Human_Clothes", "Human_Identity", "Human_Interaction",
        "Instance_Preservation", "Material", "Mechanics", "Motion_Order_Understanding",
        "Motion_Rationality", "Multi-View_Consistency", "Thermotics"
    ]

    all_prompts = []
    prompt_id = 0

    for dim in dimensions:
        json_file = meta_info_dir / f"{dim}.json"
        with open(json_file) as f:
            dim_prompts = json.load(f)

        # 每个维度取一半
        half_count = len(dim_prompts) // 2
        selected = dim_prompts[:half_count]

        print(f"{dim}: {len(dim_prompts)} -> {half_count} prompts")

        for item in selected:
            prompt_text = item["prompt_en"]
            all_prompts.append({
                "prompt_id": f"{prompt_id:05d}",
                "prompt": prompt_text,
                "dimension": dim,
                "auxiliary_info": item.get("auxiliary_info", "")
            })
            prompt_id += 1

    print(f"\n总prompts: {len(all_prompts)}")

    # 生成metadata.csv
    rows = []
    for item in all_prompts:
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

    # 保存prompt到dimension的映射（用于后续评测）
    mapping_path = output_dir / "prompt_dimension_mapping.json"
    mapping = {item["prompt_id"]: {"dimension": item["dimension"], "prompt": item["prompt"]} for item in all_prompts}
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    print(f"✓ 生成维度映射: {mapping_path}")

if __name__ == "__main__":
    main()
