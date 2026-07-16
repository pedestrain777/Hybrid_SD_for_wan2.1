#!/usr/bin/env python3
"""按维度采样 VBench2 prompts 并分配到多张 GPU 卡上

采样策略：复用之前 EC-Diff_wan 的采样结果，保持一致性
"""

import json
import random
from pathlib import Path
from collections import defaultdict

# 目标采样数量 (来自之前的采样)
TARGET_SAMPLES = {
    "Human_Anatomy": 65,
    "Human_Clothes": 43,
    "Motion_Order_Understanding": 42,
    "Human_Interaction": 35,
    "Camera_Motion": 30,
    "Human_Identity": 30,
    "Dynamic_Attribute": 26,
    "Multi-View_Consistency": 25,
    "Dynamic_Spatial_Relationship": 20,
    "Instance_Preservation": 18,
    "Complex_Plot": 16,
    "Motion_Rationality": 16,
    "Composition": 15,
    "Material": 15,
    "Mechanics": 15,
    "Thermotics": 15,
    "Complex_Landscape": 10,
    "Diversity": 10,
}

def main():
    # 直接复用之前 EC-Diff_wan 的采样结果
    prev_sample_dir = Path("/data/chenjiayu/minyu_lee/EC-Diff_wan/results/vbench/ec_diff_wan2.1_sampled314/evaluation_by_dim")

    # 读取 VBench2 prompt 信息（用于维度映射）
    info_path = Path("/data/chenjiayu/minyu_lee/hybrid_wan_14B_1.3B/VBench-master/VBench-2.0/prompts/VBench2_full_text_info.json")
    with open(info_path) as f:
        prompt_info = json.load(f)

    # 从之前的采样结果中提取 prompts
    sampled_prompts = set()
    dim_sampled = defaultdict(list)

    for dim_dir in prev_sample_dir.iterdir():
        if not dim_dir.is_dir():
            continue
        dim_name = dim_dir.name
        # 统一维度名称
        if dim_name == "Multi_View_Consistency":
            dim_name = "Multi-View_Consistency"

        for video_file in dim_dir.glob("*.mp4"):
            # 从文件名提取 prompt (去掉 -0.mp4 后缀)
            prompt = video_file.stem
            if prompt.endswith("-0"):
                prompt = prompt[:-2]

            sampled_prompts.add(prompt)
            if dim_name in TARGET_SAMPLES:
                dim_sampled[dim_name].append(prompt)

    print(f"=== 从之前采样结果中提取 ===")
    print(f"总 prompt 数: {len(sampled_prompts)}")

    for dim in TARGET_SAMPLES:
        print(f"{dim}: {len(dim_sampled[dim])}")

    # 转换为列表并排序
    all_prompts = sorted(list(sampled_prompts))

    # 分配到6张GPU卡 (2,3,4,5,6,7)
    gpus = [2, 3, 4, 5, 6, 7]
    num_gpus = len(gpus)
    prompts_per_gpu = len(all_prompts) // num_gpus
    remainder = len(all_prompts) % num_gpus

    output_dir = Path("/data/chenjiayu/minyu_lee/hybrid_wan_14B_1.3B/VBench-master/VBench-2.0/prompts")

    # 写入完整采样文件
    full_output = output_dir / "VBench2_sampled314.txt"
    with open(full_output, "w") as f:
        for prompt in all_prompts:
            f.write(prompt + "\n")
    print(f"\n完整采样文件: {full_output}")

    # 分配到各GPU
    start = 0
    gpu_assignments = {}
    for i, gpu_id in enumerate(gpus):
        count = prompts_per_gpu + (1 if i < remainder else 0)
        gpu_prompts = all_prompts[start:start + count]
        start += count

        gpu_file = output_dir / f"VBench2_sampled314_gpu{gpu_id}.txt"
        with open(gpu_file, "w") as f:
            for prompt in gpu_prompts:
                f.write(prompt + "\n")

        gpu_assignments[gpu_id] = len(gpu_prompts)
        print(f"GPU {gpu_id}: {len(gpu_prompts)} prompts -> {gpu_file}")

    # 生成维度到prompt的映射文件（用于评测）
    dim_mapping = {}
    for prompt in all_prompts:
        dims = prompt_info[prompt]["dimension"]
        dim_mapping[prompt] = dims

    mapping_file = output_dir / "VBench2_sampled314_dim_mapping.json"
    with open(mapping_file, "w") as f:
        json.dump(dim_mapping, f, indent=2, ensure_ascii=False)
    print(f"\n维度映射文件: {mapping_file}")

if __name__ == "__main__":
    main()
