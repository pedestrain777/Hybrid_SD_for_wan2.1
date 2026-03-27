#!/usr/bin/env python3
"""
EC-Diff Wan2.1 (14B+1.3B) 生成 314 个 VBench 视频
GPU 0 和 GPU 1 并行
"""

import os
import sys
import time
import multiprocessing as mp
from pathlib import Path

def run_gpu(gpu_id, prompts, output_dir):
    """在指定 GPU 上运行生成"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    sys.path.insert(0, "/data/chenjiayu/minyu_lee/Hybrid-sd_wan")
    from compression.hybrid_sd.inference_pipeline import HybridVideoInferencePipeline
    from diffusers.utils import export_to_video
    
    # 模型配置
    MODEL_PATHS = [
        "/data/chenjiayu/models/Wan2.1-T2V-14B-Diffusers",
        "/data/chenjiayu/models/Wan2.1-T2V-1.3B-Diffusers",
    ]
    
    class Args:
        def __init__(self):
            self.enable_xformers_memory_efficient_attention = False
            self.steps = [38, 12]  # 云侧 38 步，边缘侧 12 步
            # EC-Diff 参数
            self.use_ecdiff = True
            self.p_steps = 10  # 云侧预推理步数
            self.k_steps = 6   # 噪声近似步数
            self.alpha_smooth = 0.5  # 梯度平滑因子
            self.s = 38  # 云-边切换点
    
    args = Args()
    
    print(f"[GPU {gpu_id}] 加载模型...")
    pipe = HybridVideoInferencePipeline(
        weight_folders=MODEL_PATHS,
        seed=1234,
        device="cuda:0",
        args=args,
    )
    pipe.set_pipe_and_generator()
    print(f"[GPU {gpu_id}] 模型加载完成")
    
    total = len(prompts)
    for idx, prompt in enumerate(prompts):
        sample_id = f"gpu{gpu_id}_{idx}"
        output_path = output_dir / f"{sample_id}.mp4"
        
        if output_path.exists():
            print(f"[GPU {gpu_id}] [{idx+1}/{total}] 跳过已存在: {sample_id}")
            continue
        
        print(f"[GPU {gpu_id}] [{idx+1}/{total}] 生成: {sample_id}")
        print(f"  Prompt: {prompt[:60]}...")
        
        t0 = time.time()
        try:
            video_frames = pipe.generate(
                prompt=prompt,
                negative_prompt=None,
                num_frames=81,
                height=720,
                width=1280,
                guidance_scale=5.0,
                num_videos_per_prompt=1,
                output_type="pil",
            )
            
            if isinstance(video_frames, list) and video_frames and isinstance(video_frames[0], list):
                video = video_frames[0]
            else:
                video = video_frames
            
            export_to_video(video, str(output_path), fps=16)
            print(f"[GPU {gpu_id}] ✅ 完成 ({time.time()-t0:.1f}s): {output_path.name}")
        except Exception as e:
            print(f"[GPU {gpu_id}] ❌ 失败: {e}")
    
    print(f"[GPU {gpu_id}] 所有任务完成")


def main():
    # 读取 314 prompts
    prompts_files = [
        "/data/chenjiayu/minyu_lee/VDiT/results/vbench/wan22_14B_sampled314/prompts_gpu2.txt",
        "/data/chenjiayu/minyu_lee/VDiT/results/vbench/wan22_14B_sampled314/prompts_gpu3.txt",
        "/data/chenjiayu/minyu_lee/VDiT/results/vbench/wan22_14B_sampled314/prompts_gpu4.txt",
        "/data/chenjiayu/minyu_lee/VDiT/results/vbench/wan22_14B_sampled314/prompts_gpu5.txt",
        "/data/chenjiayu/minyu_lee/VDiT/results/vbench/wan22_14B_sampled314/prompts_gpu6.txt",
        "/data/chenjiayu/minyu_lee/VDiT/results/vbench/wan22_14B_sampled314/prompts_gpu7.txt",
    ]
    
    all_prompts = []
    for f in prompts_files:
        with open(f, 'r') as fp:
            all_prompts.extend([line.strip() for line in fp if line.strip()])
    
    print(f"总共 {len(all_prompts)} 个 prompts")
    
    # 分成两份
    mid = len(all_prompts) // 2
    prompts_gpu0 = all_prompts[:mid]
    prompts_gpu1 = all_prompts[mid:]
    
    print(f"GPU 0: {len(prompts_gpu0)} prompts")
    print(f"GPU 1: {len(prompts_gpu1)} prompts")
    
    output_dir = Path("/data/chenjiayu/minyu_lee/Hybrid-sd_wan/results/vbench/ec_diff_wan2.1_314/videos")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("EC-Diff Wan2.1 (14B+1.3B) VBench 314 生成")
    print("EC-Diff 参数: p=10, k=6, alpha=0.5, s=38")
    print("=" * 60)
    
    # 并行启动
    p0 = mp.Process(target=run_gpu, args=(0, prompts_gpu0, output_dir))
    p1 = mp.Process(target=run_gpu, args=(1, prompts_gpu1, output_dir))
    
    p0.start()
    p1.start()
    
    p0.join()
    p1.join()
    
    print("所有生成任务完成")


if __name__ == "__main__":
    mp.set_start_method('spawn')
    main()
