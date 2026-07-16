# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Text-to-Video generation using Hybrid CogVideoX with collaborative inference.
Supports switching between multiple transformers (e.g., CogVideoX-5B and CogVideoX-2B) during denoising.
"""

import os
import argparse
import time
from datetime import datetime
import numpy as np
import torch
from diffusers.utils import check_min_version, is_wandb_available, export_to_video

from compression.hybrid_sd.inference_pipeline import HybridVideoInferencePipeline
from compression.utils.seed import set_random_seed
from compression.utils.logger import LoggerWithDepth
from copy import deepcopy
import json

import logging
logger = logging.getLogger(__name__)

if is_wandb_available():
    import wandb


def parse_args():
    parser = argparse.ArgumentParser(description="Hybrid CogVideoX Text-to-Video Generation")
    parser.add_argument(
        "--model_id",
        type=str,
        default=[
            "/data/models/hybridsd_checkpoint/CogVideoX-5b",
            "/data/models/hybridsd_checkpoint/CogVideoX-2b"
        ],
        nargs="+",
        help="Path to pretrained model directories.",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default="10,15",
        help="Step configuration for collaborative inference, format: 'large_steps,small_steps'",
    )
    parser.add_argument(
        "--val_prompts",
        type=str,
        default=None,
        nargs="+",
        help="A set of prompts for video generation.",
    )
    parser.add_argument(
        "--neg_val_prompts",
        type=str,
        default=None,
        nargs="+",
        help="A set of negative prompts for video generation.",
    )
    parser.add_argument(
        "--prompts_file",
        type=str,
        default="./examples/hybrid_sd/prompts.txt",
        help="File containing prompts (alternating positive and negative prompts).",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--num_frames", type=int, default=49, help="Number of frames in the video")
    parser.add_argument("--height", type=int, default=480, help="Video height")
    parser.add_argument("--width", type=int, default=720, help="Video width")
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="Guidance scale for classifier-free guidance")
    parser.add_argument("--num_videos_per_prompt", type=int, default=1, help="Number of videos to generate per prompt")
    parser.add_argument("--use_dynamic_cfg", action="store_true", help="Whether to use dynamic CFG")
    parser.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help='The integration to report the results and logs to. Supported platforms are `"tensorboard"`, `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.',
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results",
        help="The output directory where the generated videos will be saved.",
    )
    parser.add_argument(
        "--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers."
    )
    parser.add_argument("--num_videos", type=int, default=1, help="Number of videos to generate per prompt")
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use, cuda:gpu_number or cpu')
    parser.add_argument("--use_dpm_solver", action='store_true', help='use CogVideoXDPMScheduler')
    parser.add_argument("--fps", type=int, default=8, help="FPS for output video")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    check_min_version("0.19.0.dev0")
    set_random_seed(args.seed)
    
    # Parse steps configuration
    args.steps = list(map(int, args.steps.split(',')))
    
    # Extract model names
    model_names = []
    for index, model in enumerate(args.model_id):
        model_name = model.split("/")[-1]
        model_names.append(model_name)
    
    print(f"Model: {model_names}, Steps: {args.steps}")
    save_path = args.output_dir
    print(f"output_dir={save_path}")
    os.makedirs(save_path, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.DEBUG,  # Use DEBUG level to see latents statistics
        handlers=[
            logging.FileHandler(os.path.join(save_path, 'log.txt')),
            logging.StreamHandler()
        ]
    )
    args.logger = logger
    
    # Initialize pipeline
    logger.info("Initializing HybridVideoInferencePipeline...")
    pipeline = HybridVideoInferencePipeline(
        weight_folders=args.model_id,
        seed=args.seed,
        device=args.device,
        args=args
    )
    pipeline.set_pipe_and_generator()
    
    # Get prompts
    if args.val_prompts is not None and args.neg_val_prompts is not None:
        val_prompts = args.val_prompts
        neg_val_prompts = args.neg_val_prompts
        assert len(val_prompts) == len(neg_val_prompts), "Number of prompts and negative prompts must match"
    elif args.prompts_file is not None:
        with open(args.prompts_file, "r") as file:
            lines = [line.strip() for line in file.readlines()]
            # Filter out empty lines
            lines = [line for line in lines if line]
            # Parse prompts: each pair of lines is (prompt, negative_prompt)
            val_prompts = []
            neg_val_prompts = []
            for i in range(0, len(lines), 2):
                if i + 1 < len(lines):
                    val_prompts.append(lines[i])
                    neg_val_prompts.append(lines[i + 1])
                else:
                    # If odd number of non-empty lines, use empty string as negative prompt
                    val_prompts.append(lines[i])
                    neg_val_prompts.append("")
            logger.info(f"Loaded {len(val_prompts)} prompts from {args.prompts_file}")
            assert len(val_prompts) == len(neg_val_prompts), "Number of prompts and negative prompts must match"
    else:
        raise NotImplementedError("Please provide prompts via --val_prompts/--neg_val_prompts or --prompts_file")
    
    # Generate videos
    time_list = []
    for index in range(len(val_prompts)):
        prompt = val_prompts[index]
        neg_prompt = neg_val_prompts[index] if neg_val_prompts[index] != "" else None
        
        prompt_path = os.path.join(save_path, f"prompt_{index}")
        os.makedirs(prompt_path, exist_ok=True)
        
        for i in range(args.num_videos):
            logger.info(f"Generating video for prompt {index} / video {i}")
            video_path = os.path.join(prompt_path, f"video_{i}")
            os.makedirs(video_path, exist_ok=True)
            
            t0 = time.perf_counter()
            
            # Generate video
            try:
                video_frames = pipeline.generate(
                    prompt=prompt,
                    negative_prompt=neg_prompt,
                    num_frames=args.num_frames,
                    height=args.height,
                    width=args.width,
                    guidance_scale=args.guidance_scale,
                    num_videos_per_prompt=args.num_videos_per_prompt,
                    use_dynamic_cfg=args.use_dynamic_cfg,
                    output_type="pil",
                )
                
                logger.info(f"Generated video_frames type: {type(video_frames)}, length: {len(video_frames) if video_frames is not None else 'None'}")
                if video_frames is not None:
                    logger.info(f"First element type: {type(video_frames[0]) if len(video_frames) > 0 else 'Empty'}")
                    if len(video_frames) > 0 and isinstance(video_frames[0], list):
                        logger.info(f"First video has {len(video_frames[0])} frames")
                
                # Save video
                # video_frames from pipeline.generate() returns output.frames
                # which is a list of videos (one per num_videos_per_prompt)
                if video_frames is not None and len(video_frames) > 0:
                    # Take the first video (if multiple videos per prompt)
                    video = video_frames[0] if isinstance(video_frames, list) and isinstance(video_frames[0], list) else video_frames
                    
                    logger.info(f"Video to save type: {type(video)}, length: {len(video) if isinstance(video, list) else 'Not a list'}")
                    
                    output_video_path = os.path.join(prompt_path, f"{i}.mp4")
                    try:
                        export_to_video(video, output_video_path, fps=args.fps)
                        logger.info(f"Video saved to {output_video_path}")
                    except Exception as e:
                        logger.error(f"Error saving video to {output_video_path}: {e}", exc_info=True)
                        raise
                else:
                    logger.warning(f"Empty video frames for prompt {index}, video {i}")
            except Exception as e:
                logger.error(f"Error generating video for prompt {index}, video {i}: {e}", exc_info=True)
                raise
            
            latency = time.perf_counter() - t0
            logger.info(f"{latency:.2f} sec elapsed")
            time_list.append(latency / args.num_videos_per_prompt)
    
    avg_latency = sum(time_list) / len(time_list) if time_list else 0
    logger.info(f"Average latency: {avg_latency:.2f} sec per video")
    
    # Clean up
    pipeline.clear()
    
    # Save model info
    model_info = {
        'model_names': model_names,
        'steps': args.steps,
        'num_frames': args.num_frames,
        'height': args.height,
        'width': args.width,
        'guidance_scale': args.guidance_scale,
        'avg_latency': avg_latency,
    }
    
    model_info_path = os.path.join(save_path, "model_info.json")
    with open(model_info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    logger.info("Video generation completed!")

