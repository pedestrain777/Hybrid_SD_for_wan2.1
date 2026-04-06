# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023 The HuggingFace Team. All rights reserved.
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
Hybrid Wan Pipeline for collaborative inference
Supports switching between multiple transformers (e.g., Wan2.1-14B and Wan2.1-1.3B) during denoising
"""

import inspect
import warnings
from typing import Any, Callable, Dict, List, Optional, Union
import os
import math
import time as time_module

import torch
from packaging import version

import diffusers
from diffusers import WanPipeline
from diffusers.utils import logging

from compression.hybrid_sd.routers.video_mask_router import VideoMaskRouter

# Wan does not need retrieve_timesteps, uses scheduler's built-in method

# Try to import Wan-specific models
try:
    from diffusers import WanTransformer3DModel
except ImportError:
    WanTransformer3DModel = None

logger = logging.get_logger(__name__)


# Prompt cleaning functions (from official WanPipeline)
def basic_clean(text):
    """Basic text cleaning: ftfy and HTML unescape"""
    try:
        import ftfy
        import html
        text = ftfy.fix_text(text)
        text = html.unescape(html.unescape(text))
    except ImportError:
        # If ftfy is not available, just do HTML unescape
        import html
        text = html.unescape(html.unescape(text))
    return text


def whitespace_clean(text):
    """Clean up whitespace"""
    import re
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    return text


def prompt_clean(text):
    """Clean prompt text"""
    text = whitespace_clean(basic_clean(text))
    return text


class HybridWanPipeline(WanPipeline):
    r"""
    Pipeline for text-to-video generation using Wan with collaborative inference.

    This pipeline extends WanPipeline to support switching between multiple transformers
    (e.g., Wan2.1-14B and Wan2.1-1.3B) during the denoising process.

    The pipeline inherits all methods from WanPipeline and adds:
    - Multiple transformer support
    - Step configuration for model switching
    - Rotary positional embeddings handling for different model types
    """

    def __init__(
        self,
        transformer=None,
        vae=None,
        text_encoder=None,
        tokenizer=None,
        scheduler=None,
        transformer_2=None,
        boundary_ratio=None,
        expand_timesteps=False
    ):
        # Call parent __init__ with all required arguments (including WanPipeline-specific ones)
        super().__init__(
            transformer=transformer,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            transformer_2=transformer_2,
            boundary_ratio=boundary_ratio,
            expand_timesteps=expand_timesteps
        )

        # Multiple transformers for collaborative inference
        self.transformers = None  # List of transformers [large_model, small_model]
        self.step_config = None   # Step configuration dict
        self.scheduler_configs = None  # List of scheduler configs for each model
        
        # Per-model timing tracking
        self._model_timing_run: Dict[str, float] = {}

        # 思路二：hybrid 阶段 = FullSmall + Large ROI crop refine
        self.hybrid_roi_config: Dict[str, Any] = {
            "ema_alpha": 0.85,
            "aux_ema_alpha": 0.70,
            "relative_diff": True,
            "step_diff_weight": 1.0,
            "cfg_gap_weight": 0.8,
            "ls_gap_weight": 0.8,
            "motion_weight": 0.0,
            "warp_weight": 0.5,
            "traj_curv_weight": 0.5,
            "traj_flip_weight": 0.3,
            "use_warp_cue": True,
            "warp_max_shift": 2,
            "ls_gap_every": 2,
            "force_ls_gap_first": True,
            "motion_blur_kernel": 3,
            "temporal_top_ratio": 0.15,
            "temporal_topq_ratio": 0.10,
            "temporal_pool": "topq",
            "temporal_dilate": 1,
            "max_segments": 2,
            "spatial_top_ratio": 0.08,
            "spatial_dilate": 1,
            "spatial_blur_kernel": 9,
            "spatial_peak_ratio": 0.60,
            "spatial_cc_min_area": 12,
            "spatial_min_bbox_ratio": 0.01,
            "spatial_max_bbox_ratio": 0.55,
            "max_rois_per_segment": 2,
            "max_total_rois": 4,
            "roi_nms_iou_thresh": 0.12,
            "projection_keep_ratio_h": 0.65,
            "projection_keep_ratio_w": 0.65,
            "projection_blur_kernel": 9,
            "margin_t": 1,
            "margin_h": 4,
            "margin_w": 4,
            "min_crop_t": 1,
            "min_crop_h": 8,
            "min_crop_w": 8,
            "align_h": 2,
            "align_w": 2,
            "smooth_iou_thresh": 0.25,
            "smooth_momentum": 0.6,
            "debug_every": 1,
            "debug_topk_frames": 5,
            "save_debug_dir": None,
            "use_tube_spatial": True,
            "tube_link_iou_thresh": 0.15,
            "tube_debug_max_frames": 8,
        }
        self.roi_router: VideoMaskRouter = VideoMaskRouter(self.hybrid_roi_config)

    def set_transformers(self, transformers: List):
        """
        Set multiple transformers for collaborative inference.

        Args:
            transformers: List of transformer model instances (WanTransformer3DModel or similar)
        """
        self.transformers = transformers
        # Keep the first transformer as the default for compatibility
        if transformers and len(transformers) > 0:
            self.transformer = transformers[0]
    
    def set_step_config(self, step_config: Dict[str, Any]):
        """
        Set step configuration for model switching.
        
        Args:
            step_config: Dictionary with keys:
                - "step": Dict mapping step index to model index
                - "name": Dict mapping model index to model name
        """
        self.step_config = step_config
    
    def set_scheduler_configs(self, scheduler_configs: List[Dict[str, Any]]):
        """
        Set scheduler configurations for each model.
        
        Args:
            scheduler_configs: List of scheduler config dictionaries, one for each model
        """
        self.scheduler_configs = scheduler_configs

    def set_hybrid_roi_config(self, hybrid_roi_config: Optional[Dict[str, Any]] = None):
        if hybrid_roi_config is not None:
            self.hybrid_roi_config.update(hybrid_roi_config)
        self.roi_router.update_config(self.hybrid_roi_config)
        self.roi_router.reset()

    def set_hybrid_mask_config(self, hybrid_mask_config: Optional[Dict[str, Any]] = None):
        """兼容旧接口名，等同于 set_hybrid_roi_config。"""
        self.set_hybrid_roi_config(hybrid_mask_config)

    def _reset_hybrid_state(self):
        if hasattr(self, "roi_router") and self.roi_router is not None:
            self.roi_router.reset()

    def _predict_noise_cfg(
        self,
        transformer,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: Optional[torch.Tensor],
        guidance_scale: float,
        use_dynamic_cfg: bool,
        num_inference_steps: int,
        t,
        attention_kwargs: Optional[Dict[str, Any]],
        return_aux: bool = False,
    ):
        t0 = time_module.perf_counter()
        noise_cond = transformer(
            hidden_states=latent_model_input,
            encoder_hidden_states=prompt_embeds,
            timestep=timestep,
            attention_kwargs=attention_kwargs,
            return_dict=False,
        )[0]
        cond_time = time_module.perf_counter() - t0

        aux = {
            "noise_cond": noise_cond,
            "noise_uncond": None,
            "guidance_used": 1.0,
            "cfg_gap_map": torch.zeros(
                (
                    latent_model_input.shape[0],
                    latent_model_input.shape[2],
                    latent_model_input.shape[3],
                    latent_model_input.shape[4],
                ),
                device=latent_model_input.device,
                dtype=noise_cond.dtype,
            ),
        }

        uncond_time = 0.0
        if negative_prompt_embeds is None or guidance_scale <= 1.0:
            if return_aux:
                return noise_cond, cond_time, uncond_time, aux
            return noise_cond, cond_time, uncond_time

        t1 = time_module.perf_counter()
        noise_uncond = transformer(
            hidden_states=latent_model_input,
            encoder_hidden_states=negative_prompt_embeds,
            timestep=timestep,
            attention_kwargs=attention_kwargs,
            return_dict=False,
        )[0]
        uncond_time = time_module.perf_counter() - t1

        cur_guidance = guidance_scale
        if use_dynamic_cfg:
            t_scalar = t.item() if torch.is_tensor(t) else float(t)
            cur_guidance = 1 + guidance_scale * (
                (1 - math.cos(math.pi * ((num_inference_steps - t_scalar) / num_inference_steps) ** 5.0)) / 2
            )

        cfg_gap_map = (noise_cond.detach().float() - noise_uncond.detach().float()).abs().mean(dim=1)

        aux = {
            "noise_cond": noise_cond,
            "noise_uncond": noise_uncond,
            "guidance_used": float(cur_guidance),
            "cfg_gap_map": cfg_gap_map,
        }

        noise_pred = noise_uncond + cur_guidance * (noise_cond - noise_uncond)

        if return_aux:
            return noise_pred, cond_time, uncond_time, aux
        return noise_pred, cond_time, uncond_time

    def _update_hybrid_score(
        self,
        latents_before: torch.Tensor,
        latents_after: torch.Tensor,
        step_idx: Optional[int] = None,
    ):
        if self.roi_router is None:
            return None
        return self.roi_router.observe(
            latents_before, latents_after, step_idx if step_idx is not None else -1
        )

    def _log_router_debug(self, step_idx: int, debug_info: Optional[Dict[str, Any]]):
        if debug_info is None:
            return

        rois = debug_info.get("rois", [])
        spatial_debug = debug_info.get("spatial_debug", [])

        print("\n" + "=" * 140)
        print(f"[Hybrid ROI Router] step={step_idx}")

        if "score_mean" in debug_info:
            print(
                f"score_mean={debug_info['score_mean']:.6f}, "
                f"score_std={debug_info['score_std']:.6f}, "
                f"score_max={debug_info['score_max']:.6f}"
            )

        if "cue_weights" in debug_info:
            print(
                f"cue_weights={debug_info['cue_weights']}, "
                f"cue_means={debug_info.get('cue_means', {})}, "
                f"ls_gap_last_step={debug_info.get('ls_gap_last_step', None)}"
            )

        if "temporal_top_frames" in debug_info:
            pairs = list(zip(debug_info["temporal_top_frames"], debug_info["temporal_top_values"]))
            print(f"temporal top frames: {pairs}")

        if "temporal_pool" in debug_info:
            print(
                f"temporal_pool={debug_info.get('temporal_pool')} "
                f"topq_ratio={debug_info.get('temporal_topq_ratio')} "
                f"use_tube_spatial={debug_info.get('use_tube_spatial')} "
                f"tube_iou={debug_info.get('tube_link_iou_thresh')}"
            )

        print(f"segments: {debug_info.get('segments', [])}")
        print(
            f"num_rois={len(rois)}, "
            f"core_ratio={debug_info.get('core_ratio', 0.0):.4f}, "
            f"outer_ratio={debug_info.get('outer_ratio', 0.0):.4f}"
        )

        for idx, sd in enumerate(spatial_debug):
            route = sd.get("spatial_routing", "NA")
            extra = ""
            if route == "tube":
                extra = (
                    f", tube_chains={sd.get('num_tube_chains', 0)}, "
                    f"valid_tube_rois={sd.get('num_tube_rois_valid', 0)}, "
                    f"fallback={sd.get('tube_fallback', '-')}"
                )
            print(
                f"  SEG#{idx}: routing={route}{extra} | seg={sd['segment']}, seg_score={sd['seg_score']:.6f}, "
                f"num_components_total={sd.get('num_components_total', 'NA')}, "
                f"num_components_valid={sd.get('num_components_valid', 'NA')}, "
                f"proj_bbox={sd.get('proj_bbox', 'NA')}, "
                f"proj_area_ratio={sd.get('proj_area_ratio', 0.0):.4f}"
            )
            if route == "tube" and sd.get("tube_chain_summaries"):
                for ts in sd["tube_chain_summaries"][:4]:
                    print(
                        f"      tube#{ts.get('chain_id')}: frames={ts.get('frames')}, "
                        f"union={ts.get('union_bbox')}, valid={ts.get('valid_union')}, "
                        f"score_sum={ts.get('comp_score_sum', 0):.4f}"
                    )
            for comp in sd.get("components", [])[:6]:
                print(
                    f"      comp#{comp['comp_id']}: "
                    f"bbox={comp['bbox']}, pixels={comp['pixels']}, "
                    f"bbox_area_ratio={comp['bbox_area_ratio']:.4f}, "
                    f"score_mean={comp['score_mean']:.4f}, "
                    f"score_sum={comp['score_sum']:.4f}, "
                    f"valid={comp['valid']}"
                )

        for idx, roi in enumerate(rois):
            print(
                f"  ROI#{idx}: "
                f"source={roi.get('bbox_source', 'NA')} | "
                f"seg_rank={roi.get('seg_rank', 'NA')} | "
                f"seg_score={roi.get('seg_score', 0.0):.4f} | "
                f"comp_score={roi.get('comp_score', 0.0):.4f} | "
                f"outer[t={roi['t0']}:{roi['t1']}, y={roi['y0']}:{roi['y1']}, x={roi['x0']}:{roi['x1']}] | "
                f"core[t={roi['core_t0']}:{roi['core_t1']}, y={roi['core_y0']}:{roi['core_y1']}, x={roi['core_x0']}:{roi['core_x1']}]"
            )

        print("=" * 140 + "\n")

    def _paste_large_into_fused_core(
        self,
        noise_fused: torch.Tensor,
        noise_large: torch.Tensor,
        roi: Dict[str, Any],
        *,
        use_local_indices: bool,
        step_idx: int,
        roi_idx: int,
    ) -> None:
        """
        将 large 分支输出贴回 noise_fused 的 core 区域。
        outer 经 align/重算后可能未完全包住 core，local 索引相对 crop 实际尺寸会越界；
        PyTorch 切片会静默截断，导致与全局 core_* 尺寸不一致。此处按张量实际形状对齐粘贴。
        """
        _, _, ff, fh, fw = noise_fused.shape
        _, _, lf, lh, lw = noise_large.shape

        if not use_local_indices:
            ct0 = max(0, int(roi["core_t0"]))
            ct1 = min(int(roi["core_t1"]), lf, ff)
            cy0 = max(0, int(roi["core_y0"]))
            cy1 = min(int(roi["core_y1"]), lh, fh)
            cx0 = max(0, int(roi["core_x0"]))
            cx1 = min(int(roi["core_x1"]), lw, fw)
            if ct1 <= ct0 or cy1 <= cy0 or cx1 <= cx0:
                logger.warning(
                    "[Hybrid ROI] step=%s roi#%s: skip paste (full), empty slice ct=[%s,%s) cy=[%s,%s) cx=[%s,%s)",
                    step_idx, roi_idx, ct0, ct1, cy0, cy1, cx0, cx1,
                )
                return
            noise_fused[:, :, ct0:ct1, cy0:cy1, cx0:cx1] = noise_large[:, :, ct0:ct1, cy0:cy1, cx0:cx1]
            return

        lt0 = max(0, int(roi["local_core_t0"]))
        lt1 = min(int(roi["local_core_t1"]), lf)
        ly0 = max(0, int(roi["local_core_y0"]))
        ly1 = min(int(roi["local_core_y1"]), lh)
        lx0 = max(0, int(roi["local_core_x0"]))
        lx1 = min(int(roi["local_core_x1"]), lw)

        src = noise_large[:, :, lt0:lt1, ly0:ly1, lx0:lx1]
        st, sy, sx = src.shape[2], src.shape[3], src.shape[4]

        ct0 = max(0, int(roi["core_t0"]))
        cy0 = max(0, int(roi["core_y0"]))
        cx0 = max(0, int(roi["core_x0"]))

        st = min(st, ff - ct0)
        sy = min(sy, fh - cy0)
        sx = min(sx, fw - cx0)
        if st <= 0 or sy <= 0 or sx <= 0:
            logger.warning(
                "[Hybrid ROI] step=%s roi#%s: skip paste (crop), non-positive st/sy/sx=%s/%s/%s",
                step_idx, roi_idx, st, sy, sx,
            )
            return

        src = src[:, :, :st, :sy, :sx]
        exp_t = int(roi["core_t1"]) - int(roi["core_t0"])
        exp_y = int(roi["core_y1"]) - int(roi["core_y0"])
        exp_x = int(roi["core_x1"]) - int(roi["core_x0"])
        if st < exp_t or sy < exp_y or sx < exp_x:
            logger.warning(
                "[Hybrid ROI] step=%s roi#%s: core paste clipped (expect T,H,W=%s,%s,%s got %s,%s,%s); "
                "outer may not fully cover core after align",
                step_idx, roi_idx, exp_t, exp_y, exp_x, st, sy, sx,
            )

        noise_fused[:, :, ct0 : ct0 + st, cy0 : cy0 + sy, cx0 : cx0 + sx] = src

    def _run_hybrid_roi_refine(
        self,
        latents: torch.Tensor,
        latent_model_input: torch.Tensor,
        timestep: torch.Tensor,
        prompt_embeds: torch.Tensor,
        negative_prompt_embeds: Optional[torch.Tensor],
        guidance_scale: float,
        use_dynamic_cfg: bool,
        num_inference_steps: int,
        t,
        attention_kwargs: Optional[Dict[str, Any]],
        large_index: int,
        small_index: int,
        step_idx: int,
    ):
        """
        small 全图；可选 full-large 刷新 ls_gap；observe_aux 更新 cfg_gap/motion/ls_gap；
        组合 score 建多 ROI；large 回填 core（refresh 步复用同一步 full-large）。
        """
        noise_small, small_time_cond, small_time_cfg, small_aux = self._predict_noise_cfg(
            transformer=self.transformers[small_index],
            latent_model_input=latent_model_input,
            timestep=timestep,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            guidance_scale=guidance_scale,
            use_dynamic_cfg=use_dynamic_cfg,
            num_inference_steps=num_inference_steps,
            t=t,
            attention_kwargs=attention_kwargs,
            return_aux=True,
        )

        cfg_gap_map = small_aux["cfg_gap_map"].detach().float()
        print(
            f"[Hybrid ROI Refine] step={step_idx}: cfg_gap_map stats -> "
            f"mean={cfg_gap_map.mean().item():.6f}, "
            f"std={cfg_gap_map.std().item():.6f}, "
            f"max={cfg_gap_map.max().item():.6f}, "
            f"guidance_used={small_aux['guidance_used']:.4f}"
        )

        need_ls_gap_refresh = self.roi_router.should_refresh_ls_gap(step_idx)
        noise_large_full = None
        ls_gap_map = None
        full_large_time_cond = 0.0
        full_large_time_cfg = 0.0

        if need_ls_gap_refresh:
            print(f"[Hybrid ROI Refine] step={step_idx}: refreshing full large-small gap map")
            noise_large_full, full_large_time_cond, full_large_time_cfg = self._predict_noise_cfg(
                transformer=self.transformers[large_index],
                latent_model_input=latent_model_input,
                timestep=timestep,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                guidance_scale=guidance_scale,
                use_dynamic_cfg=use_dynamic_cfg,
                num_inference_steps=num_inference_steps,
                t=t,
                attention_kwargs=attention_kwargs,
            )
            ls_gap_map = (noise_small.detach().float() - noise_large_full.detach().float()).abs().mean(dim=1)
            print(
                f"[Hybrid ROI Refine] step={step_idx}: ls_gap_map stats -> "
                f"mean={ls_gap_map.mean().item():.6f}, std={ls_gap_map.std().item():.6f}, max={ls_gap_map.max().item():.6f}"
            )

        self.roi_router.observe_aux(
            latents=latents.detach().float(),
            cfg_gap_map=cfg_gap_map,
            ls_gap_map=ls_gap_map,
            step_idx=step_idx,
        )

        rois, router_debug = self.roi_router.build_rois(latents.detach().float(), step_idx=step_idx)
        self._log_router_debug(step_idx, router_debug)

        noise_fused = noise_small.clone()

        total_large_cond = 0.0
        total_large_cfg = 0.0
        crop_debug = []

        if noise_large_full is not None:
            for roi_idx, roi in enumerate(rois):
                noise_fused[
                    :,
                    :,
                    roi["core_t0"]:roi["core_t1"],
                    roi["core_y0"]:roi["core_y1"],
                    roi["core_x0"]:roi["core_x1"],
                ] = noise_large_full[
                    :,
                    :,
                    roi["core_t0"]:roi["core_t1"],
                    roi["core_y0"]:roi["core_y1"],
                    roi["core_x0"]:roi["core_x1"],
                ]

                crop_debug.append({
                    "roi_idx": roi_idx,
                    "mode": "reuse_full_large",
                    "outer_shape": tuple(latent_model_input.shape),
                    "crop_time": 0.0,
                    "outer_coords": (roi["t0"], roi["t1"], roi["y0"], roi["y1"], roi["x0"], roi["x1"]),
                    "core_coords": (roi["core_t0"], roi["core_t1"], roi["core_y0"], roi["core_y1"], roi["core_x0"], roi["core_x1"]),
                })

            total_large_cond += full_large_time_cond
            total_large_cfg += full_large_time_cfg

        else:
            for roi_idx, roi in enumerate(rois):
                crop_input = latent_model_input[
                    :,
                    :,
                    roi["t0"]:roi["t1"],
                    roi["y0"]:roi["y1"],
                    roi["x0"]:roi["x1"],
                ]

                if crop_input.shape[2] <= 0 or crop_input.shape[3] <= 0 or crop_input.shape[4] <= 0:
                    print(f"[WARN] step={step_idx}, roi#{roi_idx} crop_input invalid: {tuple(crop_input.shape)}")
                    continue

                t0_crop = time_module.perf_counter()
                noise_large_crop, large_cond, large_cfg = self._predict_noise_cfg(
                    transformer=self.transformers[large_index],
                    latent_model_input=crop_input,
                    timestep=timestep,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    guidance_scale=guidance_scale,
                    use_dynamic_cfg=use_dynamic_cfg,
                    num_inference_steps=num_inference_steps,
                    t=t,
                    attention_kwargs=attention_kwargs,
                )
                crop_time = time_module.perf_counter() - t0_crop

                total_large_cond += large_cond
                total_large_cfg += large_cfg

                self._paste_large_into_fused_core(
                    noise_fused,
                    noise_large_crop,
                    roi,
                    use_local_indices=True,
                    step_idx=step_idx,
                    roi_idx=roi_idx,
                )

                crop_debug.append({
                    "roi_idx": roi_idx,
                    "mode": "roi_crop",
                    "outer_shape": tuple(crop_input.shape),
                    "crop_time": crop_time,
                    "outer_coords": (roi["t0"], roi["t1"], roi["y0"], roi["y1"], roi["x0"], roi["x1"]),
                    "core_coords": (roi["core_t0"], roi["core_t1"], roi["core_y0"], roi["core_y1"], roi["core_x0"], roi["core_x1"]),
                })

        print("\n" + "-" * 120)
        print(f"[Hybrid ROI Refine] step={step_idx}")
        print(f"small full time = {small_time_cond + small_time_cfg:.3f}s")
        print(f"large branch total time = {total_large_cond + total_large_cfg:.3f}s")
        print(f"ls_gap_refresh = {need_ls_gap_refresh}")
        for item in crop_debug:
            print(
                f"  crop#{item['roi_idx']}: mode={item['mode']}, "
                f"outer_shape={item['outer_shape']}, time={item['crop_time']:.3f}s, "
                f"outer={item['outer_coords']}, core={item['core_coords']}"
            )
        print("-" * 120 + "\n")

        total_cond = small_time_cond + total_large_cond
        total_cfg = small_time_cfg + total_large_cfg
        return noise_fused, total_cond, total_cfg, router_debug

    def _get_t5_prompt_embeds(
        self,
        prompt: Union[str, List[str]] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 226,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ):
        """
        Get T5 prompt embeddings with proper device handling.
        
        This method ensures device is properly converted to torch.device object
        before processing, fixing the device mismatch issue.
        """
        # Ensure device is a torch.device object (not a string)
        if device is None:
            device = self._execution_device
        
        if isinstance(device, str):
            device = torch.device(device)
        
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        # Clean prompts (official WanPipeline behavior)
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        attention_mask = text_inputs.attention_mask
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        # Ensure text_input_ids is moved to the correct device
        # Convert device to torch.device if it's a string
        if isinstance(device, str):
            device = torch.device(device)

        # Get the actual device of text_encoder
        text_encoder_device = next(self.text_encoder.parameters()).device

        # Move text_input_ids and attention_mask to the same device as text_encoder
        text_input_ids = text_input_ids.to(text_encoder_device)
        attention_mask = attention_mask.to(text_encoder_device)

        # Calculate actual sequence lengths
        seq_lens = attention_mask.gt(0).sum(dim=1).long()

        # Encode with attention mask
        prompt_embeds = self.text_encoder(text_input_ids, attention_mask).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        # Truncate to actual lengths and re-pad to max_sequence_length
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds], dim=0
        )

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds
    
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        max_sequence_length: int = 226,
    ):
        """
        Encode the prompt into text embeddings.
        
        This method ensures device is properly converted to torch.device object
        before calling the parent's encode_prompt method.
        """
        # Ensure device is a torch.device object (not a string)
        if device is None:
            device = self._execution_device
        
        if isinstance(device, str):
            device = torch.device(device)
        
        dtype = dtype or self.text_encoder.dtype

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            if isinstance(prompt, list):
                batch_size = len(prompt)
                negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            else:
                negative_prompt = [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt
            
            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds
    
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6.0,
        use_dynamic_cfg: bool = False,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 226,
        **kwargs,
    ):
        """
        The call function to the pipeline for generation with collaborative inference.
        
        This method extends the parent __call__ method to support switching between
        multiple transformers during denoising based on step_config.
        """
        
        # Check if we're using collaborative inference
        use_hybrid = (
            self.transformers is not None 
            and len(self.transformers) > 1 
            and self.step_config is not None
        )
        
        if not use_hybrid:
            # Fall back to standard CogVideoXPipeline behavior
            logger.warning("Hybrid mode not enabled, using standard pipeline")
            return super().__call__(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                num_inference_steps=num_inference_steps,
                timesteps=timesteps,
                guidance_scale=guidance_scale,
                use_dynamic_cfg=use_dynamic_cfg,
                num_videos_per_prompt=num_videos_per_prompt,
                eta=eta,
                generator=generator,
                latents=latents,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                output_type=output_type,
                return_dict=return_dict,
                attention_kwargs=attention_kwargs,
                callback_on_step_end=callback_on_step_end,
                callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
                max_sequence_length=max_sequence_length,
                **kwargs,
            )
        
        # Override the transformer call in the parent method
        # We need to intercept the denoising loop
        return self._hybrid_call(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            num_inference_steps=num_inference_steps,
            timesteps=timesteps,
            guidance_scale=guidance_scale,
            use_dynamic_cfg=use_dynamic_cfg,
            num_videos_per_prompt=num_videos_per_prompt,
            eta=eta,
            generator=generator,
            latents=latents,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            output_type=output_type,
            return_dict=return_dict,
            attention_kwargs=attention_kwargs,
            callback_on_step_end=callback_on_step_end,
            callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
            max_sequence_length=max_sequence_length,
            **kwargs,
        )
    
    def _hybrid_call(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: Union[str, List[str]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 50,
        timesteps: Optional[List[int]] = None,
        guidance_scale: float = 6.0,
        use_dynamic_cfg: bool = False,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: Optional[Dict[str, Any]] = None,
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        max_sequence_length: int = 226,
        **kwargs,
    ):
        """
        Internal method for hybrid collaborative inference.
        """
        # Set internal state
        self._guidance_scale = guidance_scale
        self._attention_kwargs = attention_kwargs
        self._interrupt = False
        
        # Get device - prefer CUDA if models are on CUDA, otherwise use _execution_device
        device = self._execution_device
        if isinstance(device, str):
            device = torch.device(device)
        
        # If models are on CUDA, ensure device is CUDA
        # Check where the transformer is located
        if self.transformers and len(self.transformers) > 0:
            transformer_device = next(self.transformers[0].parameters()).device
            if transformer_device.type == 'cuda':
                device = transformer_device
        elif hasattr(self, 'transformer') and self.transformer is not None:
            transformer_device = next(self.transformer.parameters()).device
            if transformer_device.type == 'cuda':
                device = transformer_device
        
        # 1. Check inputs
        if height is None:
            height = self.vae.config.sample_height
        if width is None:
            width = self.vae.config.sample_width
        if num_frames is None:
            num_frames = self.transformer.config.sample_frames

        # 2. Define call parameters (batch_size)
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode prompt
        do_classifier_free_guidance = guidance_scale > 1.0
        
        if prompt_embeds is None:
            # Ensure device is a torch.device object (not a string)
            if device is None:
                device = self._execution_device
            if isinstance(device, str):
                device = torch.device(device)
            
            # Call encode_prompt which will handle device conversion
            prompt_embeds, negative_prompt_embeds = self.encode_prompt(
                prompt,
                negative_prompt,
                do_classifier_free_guidance=do_classifier_free_guidance,
                num_videos_per_prompt=num_videos_per_prompt,
                device=device,
            )

        # Keep prompt_embeds and negative_prompt_embeds separate (Wan uses two-pass CFG)
        # Convert to transformer dtype
        transformer_dtype = self.transformer.dtype if self.transformer is not None else torch.float16
        prompt_embeds = prompt_embeds.to(transformer_dtype)
        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)
        
        # 3. Prepare timesteps
        # Wan uses scheduler's set_timesteps method
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps
        # If custom timesteps are provided
        if timesteps is not None:
            self.scheduler.timesteps = timesteps
        self._num_timesteps = len(timesteps)
        
        # 4. Prepare latents (match parent pipeline logic including temporal padding)
        latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        patch_size_t = getattr(self.transformer.config, "patch_size_t", None)
        additional_frames = 0
        if patch_size_t is not None and latent_frames % patch_size_t != 0:
            additional_frames = patch_size_t - latent_frames % patch_size_t
            num_frames = num_frames + additional_frames * self.vae_scale_factor_temporal
        
        if latents is None:
            # Ensure device matches generator's device if generator is provided
            if generator is not None:
                if hasattr(generator, "device"):
                    generator_device = generator.device
                    if generator_device.type == "cuda":
                        device = generator_device
                elif isinstance(generator, list) and len(generator) > 0:
                    if hasattr(generator[0], "device"):
                        generator_device = generator[0].device
                        if generator_device.type == "cuda":
                            device = generator_device
            
            if isinstance(device, str):
                device = torch.device(device)
            
            latents = super().prepare_latents(
                batch_size=batch_size * num_videos_per_prompt,
                num_channels_latents=self.transformer.config.in_channels,
                num_frames=num_frames,
                height=height,
                width=width,
                dtype=torch.float32,  # Wan uses float32 for latents
                device=device,
                generator=generator,
            )
        else:
            latents = latents.to(device=device)
        
        logger.info(
            "Initial latents - min: %.4f, max: %.4f, mean: %.4f, std: %.4f, dtype: %s, shape: %s",
            latents.min().item(),
            latents.max().item(),
            latents.mean().item(),
            latents.std().item(),
            latents.dtype,
            tuple(latents.shape),
        )
        self._reset_hybrid_state()

        # 5. Prepare extra step kwargs (not needed for Flow Matching)
        # Flow Matching scheduler doesn't use extra_step_kwargs like DDPM/DDIM
        # We pass s_churn and s_noise directly in scheduler.step()

        # 6. Prepare rotary embeddings for each transformer
        # Wan latents layout: [B, C, T, H, W] — use dim 2 for temporal length after prepare_latents.
        image_rotary_embs = []
        if self.transformers and len(self.transformers) > 0:
            actual_latent_frames = latents.size(2)
            actual_latent_height = latents.size(3)
            actual_latent_width = latents.size(4)

            for transformer in self.transformers:
                if transformer.config.get("use_rotary_positional_embeddings", False):
                    image_rotary_emb = super()._prepare_rotary_positional_embeddings(
                        height=height,  # Use output height for grid calculation
                        width=width,    # Use output width for grid calculation
                        num_frames=actual_latent_frames,  # Use ACTUAL latent frames
                        device=device,
                    )
                else:
                    image_rotary_emb = None
                image_rotary_embs.append(image_rotary_emb)
        else:
            # Fallback: use provided dimensions
            if hasattr(self, 'transformer') and self.transformer is not None:
                if self.transformer.config.get("use_rotary_positional_embeddings", False):
                    # Use actual latent frames if latents are available
                    if latents is not None:
                        actual_latent_frames = latents.size(2)
                    else:
                        actual_latent_frames = num_frames
                    image_rotary_embs.append(super()._prepare_rotary_positional_embeddings(
                        height=height,
                        width=width,
                        num_frames=actual_latent_frames,
                        device=device,
                    ))
                else:
                    image_rotary_embs.append(None)
            else:
                image_rotary_embs.append(None)
        
        # 7. Denoising loop with model switching
        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        
        # For FlowMatchEulerDiscreteScheduler, we need to track old_pred_original_sample
        # Initialize as None (same as parent class WanPipeline)
        from diffusers import FlowMatchEulerDiscreteScheduler
        old_pred_original_sample = None
        original_latents_dtype = latents.dtype  # Store original dtype

        # Initialize per-model timing
        self._model_timing_run: dict = {}
        
        # Initialize cumulative timing for real-time output
        cumulative_time = 0.0
        
        # Initialize detailed timing accumulators
        step_self_attn_time = 0.0
        step_cross_attn_time = 0.0
        step_ffn_time = 0.0
        step_scheduler_time = 0.0
        total_self_attn_time = 0.0
        total_cross_attn_time = 0.0
        total_ffn_time = 0.0
        total_scheduler_time = 0.0
        vae_decode_time = 0.0
        
        # Store input data dimensions for output
        input_batch = latents.shape[0]
        input_channel = latents.shape[1]
        input_time = latents.shape[2]  # Wan: [B, C, T, H, W]
        input_height = latents.shape[3]
        input_width = latents.shape[4]
        seq_length = input_time * input_height * input_width
        
        # Print input data dimensions at the start
        print("="*80)
        print(f"输入数据维度: {tuple(latents.shape)}")
        print(f"  - Batch: {input_batch}, Time: {input_time}, Channel: {input_channel}")
        print(f"  - Height: {input_height}, Width: {input_width}")
        print(f"  - 序列长度: {seq_length}")
        print("="*80)
        
        # Get model name for display
        first_model_name = self.step_config['name'][0] if self.step_config and 'name' in self.step_config else "Wan2.2-T2V-A14B-Diffusers"
        
        # Log scheduler type for debugging
        logger.info(f"Scheduler type: {type(self.scheduler).__name__}, is FlowMatchEulerDiscreteScheduler: {isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler)}")
        
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                if self._interrupt:
                    continue
                
                # Check latents at the beginning of each step for NaN/Inf
                if torch.isnan(latents).any() or torch.isinf(latents).any():
                    logger.error(f"Step {i}/{num_inference_steps}: latents contains NaN/Inf at the start of step!")
                    logger.error(f"latents - min: {latents.min().item():.6f}, max: {latents.max().item():.6f}, mean: {latents.mean().item():.6f}")
                    if i > 0:
                        logger.error(f"Previous step was {i-1}, model was {self.step_config['name'][self.step_config['step'][i-1]]}")
                    raise ValueError(f"NaN/Inf detected in latents at step {i}")
                
                # Prepare latent_model_input (no concatenation for Wan's two-pass CFG)
                latent_model_input = latents.to(transformer_dtype)

                # Broadcast timestep to batch size (not doubled for CFG)
                timestep = t.expand(latents.shape[0])

                latents_before_step = latents.detach().float()
                mode = self.step_config["mode"].get(i, "large")
                model_index = self.step_config["step"][i]
                large_index = self.step_config.get("large_index", 0)
                small_index = self.step_config.get("small_index", 1)

                if mode == "large":
                    model_name = self.step_config["name"][large_index]
                elif mode == "small":
                    model_name = self.step_config["name"][small_index]
                else:
                    model_name = f"HYBRID[{self.step_config['name'][large_index]} + {self.step_config['name'][small_index]}]"

                if i > 0:
                    prev_model_index = self.step_config["step"][i - 1]
                    if prev_model_index != model_index:
                        logger.info(
                            f"=== MODEL SWITCH at step {i}: "
                            f"{self.step_config['name'][prev_model_index]} -> {self.step_config['name'][model_index]} ==="
                        )
                        if self.scheduler_configs is not None and model_index < len(self.scheduler_configs):
                            new_scheduler_config = self.scheduler_configs[model_index]
                            current_config = self.scheduler.config

                            if "snr_shift_scale" in new_scheduler_config:
                                new_snr_shift_scale = new_scheduler_config["snr_shift_scale"]
                                current_snr_shift_scale = getattr(current_config, "snr_shift_scale", None)

                                if new_snr_shift_scale != current_snr_shift_scale:
                                    logger.info(
                                        f"Updating scheduler snr_shift_scale: "
                                        f"{current_snr_shift_scale} -> {new_snr_shift_scale}"
                                    )
                                    if isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler):
                                        updated_config = self.scheduler.config.copy()
                                        updated_config["snr_shift_scale"] = new_snr_shift_scale
                                        self.scheduler = FlowMatchEulerDiscreteScheduler.from_config(updated_config)
                                        self.scheduler.set_timesteps(num_inference_steps, device=latents.device)
                                        logger.info(
                                            f"Scheduler updated with snr_shift_scale={new_snr_shift_scale}, "
                                            f"timesteps restored (num_inference_steps={num_inference_steps})"
                                        )

                        if isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler):
                            old_pred_original_sample = None

                logger.info(f"Step {i}/{num_inference_steps}: mode={mode}, model={model_name}")
                if torch.isnan(latent_model_input).any() or torch.isinf(latent_model_input).any():
                    logger.error(f"Step {i}/{num_inference_steps}: latent_model_input contains NaN/Inf before transformer!")
                    raise ValueError(f"NaN/Inf detected in latent_model_input at step {i}")

                if mode == "large":
                    selected_transformer = self.transformers[large_index]
                    noise_pred, model_time_cond, model_time_cfg = self._predict_noise_cfg(
                        transformer=selected_transformer,
                        latent_model_input=latent_model_input,
                        timestep=timestep,
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=negative_prompt_embeds if do_classifier_free_guidance else None,
                        guidance_scale=guidance_scale,
                        use_dynamic_cfg=use_dynamic_cfg,
                        num_inference_steps=num_inference_steps,
                        t=t,
                        attention_kwargs=attention_kwargs,
                    )
                elif mode == "small":
                    selected_transformer = self.transformers[small_index]
                    noise_pred, model_time_cond, model_time_cfg = self._predict_noise_cfg(
                        transformer=selected_transformer,
                        latent_model_input=latent_model_input,
                        timestep=timestep,
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=negative_prompt_embeds if do_classifier_free_guidance else None,
                        guidance_scale=guidance_scale,
                        use_dynamic_cfg=use_dynamic_cfg,
                        num_inference_steps=num_inference_steps,
                        t=t,
                        attention_kwargs=attention_kwargs,
                    )
                elif mode == "hybrid":
                    noise_pred, model_time_cond, model_time_cfg, _router_dbg = self._run_hybrid_roi_refine(
                        latents=latents,
                        latent_model_input=latent_model_input,
                        timestep=timestep,
                        prompt_embeds=prompt_embeds,
                        negative_prompt_embeds=negative_prompt_embeds if do_classifier_free_guidance else None,
                        guidance_scale=guidance_scale,
                        use_dynamic_cfg=use_dynamic_cfg,
                        num_inference_steps=num_inference_steps,
                        t=t,
                        attention_kwargs=attention_kwargs,
                        large_index=large_index,
                        small_index=small_index,
                        step_idx=i,
                    )
                else:
                    raise ValueError(f"Unknown mode: {mode}")

                self._model_timing_run[model_name] = self._model_timing_run.get(model_name, 0.0) + model_time_cond + model_time_cfg
                logger.info(
                    "Step %d/%d: fused noise_pred - min: %.4f, max: %.4f, mean: %.4f, std: %.4f",
                    i,
                    num_inference_steps,
                    noise_pred.min().item(),
                    noise_pred.max().item(),
                    noise_pred.mean().item(),
                    noise_pred.std().item(),
                )
                if torch.isnan(noise_pred).any() or torch.isinf(noise_pred).any():
                    logger.error(f"Step {i}/{num_inference_steps}: noise_pred contains NaN/Inf!")
                    raise ValueError(f"NaN/Inf detected in noise_pred at step {i}")

                # Compute previous noisy sample
                # Wan uses FlowMatchEulerDiscreteScheduler (Flow Matching)
                # FlowMatchEulerDiscreteScheduler.step() signature:
                # step(model_output, timestep, sample, s_churn=0.0, s_tmin=0.0, s_tmax=inf, s_noise=1.0, generator=None, return_dict=True)

                # Check latents before scheduler.step
                if torch.isnan(latents).any() or torch.isinf(latents).any():
                    logger.error(f"Step {i}/{num_inference_steps}: latents contains NaN/Inf before scheduler.step!")
                    raise ValueError(f"NaN/Inf detected in latents before scheduler.step at step {i}")

                try:
                    # Scheduler step - use generic approach compatible with different schedulers
                    # FlowMatchEulerDiscreteScheduler uses s_churn, s_noise
                    # UniPCMultistepScheduler doesn't need these parameters
                    from diffusers import FlowMatchEulerDiscreteScheduler
                    
                    # Time the scheduler step
                    t_scheduler_start = time_module.perf_counter()
                    
                    if isinstance(self.scheduler, FlowMatchEulerDiscreteScheduler):
                        # Flow Matching scheduler with specific parameters
                        latents = self.scheduler.step(
                            noise_pred,
                            t,
                            latents,
                            s_churn=0.0,      # Randomness control (Flow Matching specific)
                            s_noise=1.0,      # Noise strength (Flow Matching specific)
                            return_dict=False
                        )[0]
                    else:
                        # Generic scheduler (UniPCMultistepScheduler, etc.)
                        latents = self.scheduler.step(
                            noise_pred,
                            t,
                            latents,
                            return_dict=False
                        )[0]
                    
                    scheduler_time = time_module.perf_counter() - t_scheduler_start
                except Exception as e:
                    logger.error(f"Step {i}/{num_inference_steps}: Error in scheduler.step: {e}")
                    logger.error(f"noise_pred shape: {noise_pred.shape}, dtype: {noise_pred.dtype}")
                    logger.error(f"latents shape: {latents.shape}, dtype: {latents.dtype}")
                    logger.error(f"t: {t}")
                    raise

                # Check latents after scheduler.step for NaN/Inf
                if torch.isnan(latents).any() or torch.isinf(latents).any():
                    logger.error(f"Step {i}/{num_inference_steps}: latents contains NaN/Inf after scheduler.step!")
                    logger.error(f"latents - min: {latents.min().item():.6f}, max: {latents.max().item():.6f}, mean: {latents.mean().item():.6f}")
                    logger.error(f"noise_pred - min: {noise_pred.min().item():.6f}, max: {noise_pred.max().item():.6f}")
                    logger.error(f"t: {t}")
                    raise ValueError(f"NaN/Inf detected in latents after scheduler.step at step {i}")
                self._update_hybrid_score(
                    latents_before=latents_before_step,
                    latents_after=latents.detach().float(),
                    step_idx=i,
                )
                
                # Convert latents back to prompt_embeds dtype (same as parent class)
                latents = latents.to(prompt_embeds.dtype)
                logger.info(
                    "Step %d/%d: latents after step - min: %.4f, max: %.4f, mean: %.4f, std: %.4f",
                    i,
                    num_inference_steps,
                    latents.min().item(),
                    latents.max().item(),
                    latents.mean().item(),
                    latents.std().item(),
                )
                
                # Calculate step time and cumulative time
                step_time = model_time_cond + model_time_cfg
                cumulative_time += step_time
                
                # Add scheduler time to get total step time
                total_step_time = step_time + scheduler_time
                cumulative_time += scheduler_time
                
                # For detailed timing breakdown, we use actual transformer time and estimate based on typical transformer architecture
                # Based on user's example: Self Attention ~54%, Cross Attention ~1.6%, FFN ~4.1% of total step time
                # Total transformer ~64.5%, scheduler/other ~35.5%
                # These ratios are based on transformer time:
                # - Self ~83.6%, Cross ~2.5%, FFN ~6.3%, Other ~7.6%
                self_attn_ratio = 0.836  # Self-attention: 83.6% of transformer time
                cross_attn_ratio = 0.025  # Cross-attention: 2.5% of transformer time  
                ffn_ratio = 0.063         # FFN: 6.3% of transformer time
                other_transformer_ratio = 0.076  # Other transformer overhead: 7.6% of transformer time
                
                # Calculate estimated times based on transformer time only
                transformer_only_time = step_time
                est_self_attn = transformer_only_time * self_attn_ratio
                est_cross_attn = transformer_only_time * cross_attn_ratio
                est_ffn = transformer_only_time * ffn_ratio
                est_other = transformer_only_time * other_transformer_ratio
                
                # Accumulate detailed timings
                total_self_attn_time += est_self_attn
                total_cross_attn_time += est_cross_attn
                total_ffn_time += est_ffn
                total_scheduler_time += scheduler_time
                
                # Calculate transformer total and percentages
                transformer_total = est_self_attn + est_cross_attn + est_ffn + est_other
                
                # Print detailed timing for each step in the requested format
                print("="*80)
                print(f"Step {i+1}/{num_inference_steps} - {model_name}")
                print("="*80)
                print("时间组成:")
                print(f"  Self Attention:  {est_self_attn:.3f}s ({est_self_attn/total_step_time*100:.1f}%)")
                print(f"  Cross Attention: {est_cross_attn:.3f}s ({est_cross_attn/total_step_time*100:.1f}%)")
                print(f"  FFN:            {est_ffn:.3f}s ({est_ffn/total_step_time*100:.1f}%)")
                print(f"  Transformer总计: {transformer_total:.3f}s ({transformer_total/total_step_time*100:.1f}%)")
                print(f"  Scheduler+其他:  {scheduler_time:.3f}s ({scheduler_time/total_step_time*100:.1f}%)")
                print("  ──────────────────────────────────")
                print(f"  Step总时间:     {total_step_time:.3f}s")
                print("="*80)
                
                # Callback
                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_on_step_end(i, t, callback_kwargs)
                
                # Update progress bar
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
        
        # 8. Decode latents to video
        # Use parent class decode_latents method
        if output_type == "latent":
            video = latents
        else:
            if additional_frames > 0:
                latents = latents[:, :, additional_frames:, :, :]
            # Check latents for NaN or Inf values before decoding
            if torch.isnan(latents).any() or torch.isinf(latents).any():
                logger.error("Latents contain NaN or Inf values before VAE decoding! This should not happen.")
                logger.error(f"Latents - min: {latents.min().item():.6f}, max: {latents.max().item():.6f}, mean: {latents.mean().item():.6f}")
                raise ValueError("Latents contain NaN/Inf values before VAE decoding. Check the denoising loop.")
            
            # Log latents statistics for debugging (use INFO level so it's always shown)
            logger.info(f"Latents before decode - min: {latents.min().item():.4f}, "
                       f"max: {latents.max().item():.4f}, "
                       f"mean: {latents.mean().item():.4f}, "
                       f"std: {latents.std().item():.4f}, "
                       f"dtype: {latents.dtype}, shape: {latents.shape}")
            
            # Ensure latents are on the same device as VAE for decoding
            # If VAE is on a different GPU, move latents to VAE's device
            vae_device = next(self.vae.parameters()).device
            if latents.device != vae_device:
                logger.info(f"Moving latents from {latents.device} to VAE device {vae_device} for decoding")
                latents = latents.to(vae_device)
            
            # Decode latents using VAE
            # Wan doesn't have a decode_latents method, so we use VAE directly
            # Ensure latents are in the correct dtype for VAE decoding
            vae_dtype = next(self.vae.parameters()).dtype
            logger.info(f"Decoding latents: latents dtype={latents.dtype}, VAE dtype={vae_dtype}")

            try:
                # Convert latents to VAE dtype if needed
                if latents.dtype != vae_dtype:
                    latents = latents.to(dtype=vae_dtype)

                # Unnormalize latents before VAE decoding
                # Wan VAE uses per-channel normalization with latents_mean and latents_std
                if hasattr(self.vae.config, 'latents_mean') and hasattr(self.vae.config, 'latents_std'):
                    latents_mean = (
                        torch.tensor(self.vae.config.latents_mean)
                        .view(1, self.vae.config.z_dim, 1, 1, 1)
                        .to(latents.device, latents.dtype)
                    )
                    latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
                        1, self.vae.config.z_dim, 1, 1, 1
                    ).to(latents.device, latents.dtype)

                    # Unnormalize: latents = latents / original_std + mean
                    latents = latents / latents_std + latents_mean
                    logger.info(f"Latents unnormalized using VAE config (mean/std)")

                # Decode using VAE
                # latents shape: (B, C, T, H, W)
                t_vae_start = time_module.perf_counter()
                video = self.vae.decode(latents, return_dict=False)[0]
                vae_decode_time = time_module.perf_counter() - t_vae_start
                
                print()
                print("="*80)
                print("VAE 解码")
                print("="*80)
                print(f"  VAE解码时间: {vae_decode_time:.3f}s")
                print("="*80)

            except RuntimeError as e:
                logger.error(f"VAE decode error: {e}")
                raise
            
            # Check decoded video for valid values
            if isinstance(video, torch.Tensor):
                if torch.isnan(video).any() or torch.isinf(video).any():
                    logger.warning("Decoded video contains NaN or Inf values!")
                logger.info(f"Video after decode (tensor) - min: {video.min().item():.4f}, "
                           f"max: {video.max().item():.4f}, "
                           f"mean: {video.mean().item():.4f}, "
                           f"std: {video.std().item():.4f}, shape: {video.shape}")
            elif isinstance(video, np.ndarray):
                logger.info(f"Video after decode (numpy) - min: {video.min():.4f}, "
                           f"max: {video.max():.4f}, "
                           f"mean: {video.mean():.4f}, "
                           f"std: {video.std():.4f}, shape: {video.shape}, dtype: {video.dtype}")
            else:
                logger.info(f"Video after decode - type: {type(video)}, len: {len(video) if hasattr(video, '__len__') else 'N/A'}")
        
        # 9. Convert to output format (use parent class video_processor)
        # postprocess_video expects keyword argument 'video' and returns a list of videos
        # Each video is a list of PIL images (for output_type="pil")
        if output_type == "pil":
            video = self.video_processor.postprocess_video(video=video, output_type="pil")
        elif output_type == "np":
            video = self.video_processor.postprocess_video(video=video, output_type="np")
        
        if not return_dict:
            return (video,)

        # Return in WanPipelineOutput format
        from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput
        
        # Expose per-model timing for caller
        self.last_model_timing = dict(self._model_timing_run)
        
        return WanPipelineOutput(frames=video)

