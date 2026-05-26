"""Utilities for ST-Hybrid motivation/observation scripts on Wan2.1.

These scripts are diagnostic only: they do not run ST-Hybrid and do not decode
final videos. They inspect Wan's latent denoising trajectory and model
predictions to generate motivation figures.
"""

from __future__ import annotations

import gc
import math
import os
import random
import sys
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.cuda.amp as amp

# Allow running scripts from analysis_scripts/ while importing the local wan package.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import wan  # noqa: E402
from wan.configs import SIZE_CONFIGS, WAN_CONFIGS  # noqa: E402
from wan.utils.fm_solvers import (  # noqa: E402
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler  # noqa: E402


DEFAULT_PROMPTS = [
    "A dog running across a green grass field, cinematic lighting.",
    "A dancer spinning quickly under colorful stage lights.",
    "A person pouring water from a bottle into a glass on a wooden table.",
    "A camera slowly pans across a busy city street at night.",
    "Two children playing basketball on an outdoor court.",
    "A boat floating on a calm lake with mountains in the background.",
    "A horse galloping through shallow water on a beach.",
    "A cat jumps from a sofa onto a table in a living room.",
]


@dataclass
class PromptItem:
    prompt: str
    seed: int


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_size(size: str) -> Tuple[int, int]:
    """Parse Wan-style size string, e.g., '832*480', into (width, height)."""
    if "*" in size:
        w, h = size.lower().split("*")
        return int(w), int(h)
    if "x" in size.lower():
        w, h = size.lower().split("x")
        return int(w), int(h)
    if size in SIZE_CONFIGS:
        return SIZE_CONFIGS[size]
    raise ValueError(f"Unsupported size format: {size}. Use e.g. 832*480.")


def parse_steps(steps: str) -> List[int]:
    out = []
    for x in steps.split(","):
        x = x.strip()
        if x:
            out.append(int(x))
    if not out:
        raise ValueError("selected steps cannot be empty")
    return out


def load_prompt_items(prompt_file: Optional[str], base_seed: int = 0) -> List[PromptItem]:
    """Load prompts. Each line can be either 'prompt' or 'seed<TAB>prompt'."""
    prompts: List[PromptItem] = []
    if prompt_file:
        with open(prompt_file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "\t" in line:
                    left, prompt = line.split("\t", 1)
                    try:
                        seed = int(left.strip())
                    except ValueError:
                        seed = base_seed + i
                        prompt = line
                else:
                    seed = base_seed + i
                    prompt = line
                prompts.append(PromptItem(prompt=prompt.strip(), seed=seed))
    else:
        prompts = [PromptItem(prompt=p, seed=base_seed + i) for i, p in enumerate(DEFAULT_PROMPTS)]
    if not prompts:
        raise ValueError("No prompts found.")
    return prompts


@contextmanager
def maybe_no_sync(model):
    no_sync = getattr(model, "no_sync", None)
    if no_sync is None:
        yield
    else:
        with no_sync():
            yield


def create_t2v_pipeline(
    task: str,
    ckpt_dir: str,
    device_id: int = 0,
    t5_cpu: bool = False,
):
    if task not in WAN_CONFIGS:
        raise ValueError(f"Unknown Wan task {task}. Available: {list(WAN_CONFIGS)}")
    cfg = WAN_CONFIGS[task]
    pipe = wan.WanT2V(
        config=cfg,
        checkpoint_dir=ckpt_dir,
        device_id=device_id,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=t5_cpu,
    )
    pipe.model.eval().requires_grad_(False)
    return pipe


def cleanup_pipeline(pipe) -> None:
    try:
        pipe.model.cpu()
    except Exception:
        pass
    try:
        pipe.text_encoder.model.cpu()
    except Exception:
        pass
    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def compute_target_shape(pipe, size: Tuple[int, int], frame_num: int) -> Tuple[int, int, int, int]:
    width, height = size
    return (
        pipe.vae.model.z_dim,
        (frame_num - 1) // pipe.vae_stride[0] + 1,
        height // pipe.vae_stride[1],
        width // pipe.vae_stride[2],
    )


def compute_seq_len(pipe, target_shape: Tuple[int, int, int, int]) -> int:
    return math.ceil(
        (target_shape[2] * target_shape[3])
        / (pipe.patch_size[1] * pipe.patch_size[2])
        * target_shape[1]
        / pipe.sp_size
    ) * pipe.sp_size


def encode_prompt(pipe, prompt: str, negative_prompt: str = ""):
    if negative_prompt == "":
        negative_prompt = pipe.sample_neg_prompt
    if not pipe.t5_cpu:
        pipe.text_encoder.model.to(pipe.device)
        context = pipe.text_encoder([prompt], pipe.device)
        context_null = pipe.text_encoder([negative_prompt], pipe.device)
    else:
        context = pipe.text_encoder([prompt], torch.device("cpu"))
        context_null = pipe.text_encoder([negative_prompt], torch.device("cpu"))
        context = [t.to(pipe.device) for t in context]
        context_null = [t.to(pipe.device) for t in context_null]
    return context, context_null


def initial_latent(
    target_shape: Tuple[int, int, int, int],
    device: torch.device,
    seed: int,
) -> Tuple[List[torch.Tensor], torch.Generator]:
    seed_g = torch.Generator(device=device)
    seed_g.manual_seed(seed)
    noise = torch.randn(
        target_shape[0], target_shape[1], target_shape[2], target_shape[3],
        dtype=torch.float32, device=device, generator=seed_g)
    return [noise], seed_g


def make_scheduler(
    solver: str,
    num_train_timesteps: int,
    sampling_steps: int,
    shift: float,
    device: torch.device,
):
    if solver == "unipc":
        scheduler = FlowUniPCMultistepScheduler(
            num_train_timesteps=num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False,
        )
        scheduler.set_timesteps(sampling_steps, device=device, shift=shift)
        timesteps = scheduler.timesteps
    elif solver == "dpm++":
        scheduler = FlowDPMSolverMultistepScheduler(
            num_train_timesteps=num_train_timesteps,
            shift=1,
            use_dynamic_shifting=False,
        )
        sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
        timesteps, _ = retrieve_timesteps(scheduler, device=device, sigmas=sampling_sigmas)
    else:
        raise NotImplementedError(f"Unsupported solver: {solver}")
    return scheduler, timesteps


def cfg_noise_prediction(
    pipe,
    latent: torch.Tensor,
    timestep: torch.Tensor,
    context,
    context_null,
    seq_len: int,
    guide_scale: float,
) -> torch.Tensor:
    """Return classifier-free guided noise/velocity prediction [C,F,H,W]."""
    pipe.model.to(pipe.device)
    timestep_tensor = torch.stack([timestep]) if timestep.dim() == 0 else timestep
    latents = [latent.to(pipe.device)]
    arg_c = {"context": context, "seq_len": seq_len}
    arg_null = {"context": context_null, "seq_len": seq_len}
    with amp.autocast(dtype=pipe.param_dtype), torch.no_grad(), maybe_no_sync(pipe.model):
        pred_cond = pipe.model(latents, t=timestep_tensor, **arg_c)[0]
        pred_uncond = pipe.model(latents, t=timestep_tensor, **arg_null)[0]
        pred = pred_uncond + guide_scale * (pred_cond - pred_uncond)
    return pred.float()


def scheduler_step(scheduler, pred: torch.Tensor, timestep: torch.Tensor, latent: torch.Tensor, seed_g):
    nxt = scheduler.step(
        pred.unsqueeze(0),
        timestep,
        latent.unsqueeze(0),
        return_dict=False,
        generator=seed_g,
    )[0]
    return nxt.squeeze(0).float()


def token_l2(x: torch.Tensor) -> torch.Tensor:
    """L2 norm over channel dimension for [C,F,H,W] -> [F,H,W]."""
    return torch.linalg.vector_norm(x.float(), dim=0)


def normalized_prediction_difference(
    pred_large: torch.Tensor,
    pred_small: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Per-token normalized difference in [F,H,W], mostly within [0,1]."""
    num = token_l2(pred_large - pred_small)
    denom = token_l2(pred_large) + token_l2(pred_small) + eps
    return num / denom


def normalized_update_map(
    x_next: torch.Tensor,
    x_cur: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Per-token step_diff map [F,H,W], normalized by global latent magnitude."""
    delta = token_l2(x_next - x_cur)
    scale = token_l2(x_cur).mean().clamp_min(eps)
    return delta / scale


def segment_vector(frame_scores: torch.Tensor, segment_len: int) -> torch.Tensor:
    """Average frame-level scores into temporal segments."""
    if segment_len <= 1:
        return frame_scores.float()
    vals = []
    for start in range(0, frame_scores.numel(), segment_len):
        vals.append(frame_scores[start:start + segment_len].mean())
    return torch.stack(vals).float()


def percentile_clip(arr: np.ndarray, pct: float = 99.0) -> np.ndarray:
    if arr.size == 0:
        return arr
    vmax = np.percentile(arr, pct)
    return np.clip(arr, None, vmax)
