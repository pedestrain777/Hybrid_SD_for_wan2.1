import json
import math
import os
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


def _topk_binary_mask(scores: torch.Tensor, ratio: float) -> torch.Tensor:
    """Return a per-sample top-ratio mask."""
    ratio = float(max(0.0, min(1.0, ratio)))
    bsz = scores.shape[0]
    flat = scores.reshape(bsz, -1)
    total = flat.shape[1]
    k = max(1, min(total, int(math.ceil(total * ratio))))
    vals = torch.topk(flat, k=k, dim=1).values
    thr = vals[:, -1].view(bsz, *([1] * (scores.ndim - 1)))
    return scores >= thr


def _normalize_2d(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    return (x - x.min()) / (x.max() - x.min()).clamp_min(1e-6)


def _normalize_map_per_sample(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if x is None:
        return None
    flat = x.reshape(x.shape[0], -1)
    xmin = flat.min(dim=1).values[:, None, None, None]
    xmax = flat.max(dim=1).values[:, None, None, None]
    return (x - xmin) / (xmax - xmin).clamp_min(1e-6)


def _bool_to_segments(mask_1d: torch.Tensor) -> List[Tuple[int, int]]:
    segments: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for i, v in enumerate(mask_1d.detach().cpu().tolist()):
        if v and start is None:
            start = i
        elif (not v) and start is not None:
            segments.append((start, i))
            start = None
    if start is not None:
        segments.append((start, int(mask_1d.numel())))
    return segments


def _bbox_from_mask(mask_2d: torch.Tensor) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = torch.where(mask_2d)
    if ys.numel() == 0:
        return None
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def _bbox_area(bbox: Tuple[int, int, int, int]) -> int:
    y0, y1, x0, x1 = bbox
    return max(0, y1 - y0) * max(0, x1 - x0)


def _extract_connected_components(mask_2d: torch.Tensor) -> List[Dict[str, Any]]:
    """Small CPU connected-component extractor for a 2D boolean mask."""
    h, w = mask_2d.shape
    mask_cpu = mask_2d.detach().to(torch.bool).cpu()
    visited = torch.zeros((h, w), dtype=torch.bool)
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    comps: List[Dict[str, Any]] = []

    for y in range(h):
        for x in range(w):
            if not bool(mask_cpu[y, x]) or bool(visited[y, x]):
                continue
            q = deque([(y, x)])
            visited[y, x] = True
            pts: List[Tuple[int, int]] = []
            while q:
                cy, cx = q.popleft()
                pts.append((cy, cx))
                for dy, dx in neighbors:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and bool(mask_cpu[ny, nx]) and (not bool(visited[ny, nx])):
                        visited[ny, nx] = True
                        q.append((ny, nx))
            out_mask = torch.zeros((h, w), dtype=torch.bool, device=mask_2d.device)
            for py, px in pts:
                out_mask[py, px] = True
            bbox = _bbox_from_mask(out_mask)
            if bbox is not None:
                comps.append({"mask": out_mask, "bbox": bbox, "pixels": len(pts)})
    return comps


def _compute_frame_diff_map(latents: torch.Tensor) -> torch.Tensor:
    """
    Frame-wise latent difference used by the new temporal router.
    latents: [B,C,T,H,W] -> [B,T,H,W]
    Each latent frame gets the average of its adjacent-frame differences.
    """
    b, c, t, h, w = latents.shape
    out = torch.zeros((b, t, h, w), device=latents.device, dtype=torch.float32)
    if t <= 1:
        return out
    diff = (latents[:, :, 1:] - latents[:, :, :-1]).abs().mean(dim=1).float()  # [B,T-1,H,W]
    count = torch.zeros_like(out)
    out[:, 1:] += diff
    count[:, 1:] += 1
    out[:, :-1] += diff
    count[:, :-1] += 1
    return out / count.clamp_min(1.0)


def _compute_warp_residual(latents: torch.Tensor, max_shift: int) -> torch.Tensor:
    """
    Cheap warp inconsistency: search a small integer translation of z_{t-1}
    that best matches z_t, then use the residual as a spatial score.
    latents: [B,C,T,H,W] -> [B,T,H,W]
    """
    b, c, t, h, w = latents.shape
    out = torch.zeros((b, t, h, w), device=latents.device, dtype=torch.float32)
    if t <= 1:
        return out
    shifts = list(range(-int(max_shift), int(max_shift) + 1))
    for ti in range(1, t):
        prev = latents[:, :, ti - 1]
        curr = latents[:, :, ti]
        best_score: Optional[float] = None
        best_res: Optional[torch.Tensor] = None
        for dy in shifts:
            for dx in shifts:
                rolled = torch.roll(prev, shifts=(dy, dx), dims=(2, 3))
                res = (curr - rolled).abs().mean(dim=1).float()
                score = float(res.mean().item())
                if best_score is None or score < best_score:
                    best_score = score
                    best_res = res
        assert best_res is not None
        out[:, ti] = best_res
    out[:, 0] = out[:, 1]
    return out


def _expand_align_bounds(start: int, end: int, limit: int, margin: int, min_size: int, align: int) -> Tuple[int, int]:
    start = max(0, int(start) - int(margin))
    end = min(int(limit), int(end) + int(margin))
    if end <= start:
        end = min(int(limit), start + 1)
    size = max(end - start, int(min_size))
    if align > 1:
        size = int(math.ceil(size / int(align)) * int(align))
    size = min(size, int(limit))
    center = 0.5 * (start + end)
    new_start = int(round(center - size / 2.0))
    new_start = max(0, min(new_start, int(limit) - size))
    return new_start, new_start + size


class VideoMaskRouter:
    """
    Simplified two-stage spatiotemporal cube router.

    New Hybrid routing policy:
      1) Temporal routing uses frame-wise latent difference to identify hard segments.
      2) Spatial routing uses ONE selected cue inside those segments: cfg / motion / warp.
      3) Each selected segment produces one compact core cube via top-ratio mask + connected component.
      4) The large model sees an expanded outer cube, while only the core cube is pasted back.

    The old rule-heavy fields are intentionally kept as ignored compatibility keys so older scripts
    can still instantiate the pipeline, but the default build_rois path below does not use tube,
    NMS, trajectory curvature, or multi-cue weighted fusion.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config: Dict[str, Any] = {
            # Core new router parameters.
            "routing_mode": "two_stage_cube",
            "temporal_cue": "frame_diff",
            "temporal_top_ratio": 0.15,
            "max_temporal_segments": 2,
            "spatial_cue": "cfg",          # choices: cfg / motion / warp
            "spatial_top_ratio": 0.08,
            "max_cubes": 2,

            # Large-model context crop.
            "margin_t": 1,
            "margin_h": 4,
            "margin_w": 4,
            "min_crop_t": 1,
            "min_crop_h": 8,
            "min_crop_w": 8,
            "align_h": 2,
            "align_w": 2,

            # Spatial cue details kept minimal.
            "warp_max_shift": 2,
            "aux_ema_alpha": 0.0,          # 0 means use current-step cue; >0 enables mild EMA.
            "save_debug_dir": None,
            "debug_every": 1,
            "debug_topk_frames": 5,
            "debug_save_all_cues": True,

            # Backward-compatible aliases/ignored keys from the old router.
            "max_total_rois": 2,
            "max_segments": 2,
            "step_diff_weight": 0.0,
            "cfg_gap_weight": 1.0,
            "warp_weight": 0.0,
            "traj_curv_weight": 0.0,
            "use_warp_cue": True,
            "relative_diff": True,
            "ema_alpha": 0.85,
        }
        if config is not None:
            self.config.update(config)
        # Alias old names to the new concise ones when old scripts pass old config.
        if "max_temporal_segments" not in self.config and "max_segments" in self.config:
            self.config["max_temporal_segments"] = self.config["max_segments"]
        if "max_cubes" not in self.config and "max_total_rois" in self.config:
            self.config["max_cubes"] = self.config["max_total_rois"]
        self.reset()

    def update_config(self, config: Optional[Dict[str, Any]] = None):
        if config is not None:
            self.config.update(config)
        if "max_temporal_segments" not in self.config and "max_segments" in self.config:
            self.config["max_temporal_segments"] = self.config["max_segments"]
        if "max_cubes" not in self.config and "max_total_rois" in self.config:
            self.config["max_cubes"] = self.config["max_total_rois"]

    def reset(self):
        self.step_diff_ema: Optional[torch.Tensor] = None
        self.cfg_gap_map: Optional[torch.Tensor] = None
        self.cfg_gap_ema: Optional[torch.Tensor] = None
        self.warp_map: Optional[torch.Tensor] = None
        self.motion_map: Optional[torch.Tensor] = None
        self.last_ls_gap_step: int = -10**9

        # Old attributes kept for debug/render compatibility.
        self.score_ema: Optional[torch.Tensor] = None
        self.traj_curv_ema: Optional[torch.Tensor] = None
        self.traj_flip_ema: Optional[torch.Tensor] = None
        self.ls_gap_ema: Optional[torch.Tensor] = None
        self.motion_ema: Optional[torch.Tensor] = None
        self.warp_ema: Optional[torch.Tensor] = None
        self.prev_rois: List[Dict[str, Any]] = []

    def _active_cue_names(self) -> List[str]:
        spatial_cue = str(self.config.get("spatial_cue", "cfg")).lower()
        return ["frame_diff_temporal", f"{spatial_cue}_spatial"]

    def compute_diff_map(self, latents_before: torch.Tensor, latents_after: torch.Tensor) -> torch.Tensor:
        diff = (latents_before - latents_after).abs().mean(dim=1).float()
        if bool(self.config.get("relative_diff", True)):
            denom = latents_before.abs().mean(dim=1).clamp_min(1e-6)
            diff = diff / denom
        return diff.float()

    def observe(self, latents_before: torch.Tensor, latents_after: torch.Tensor, step_idx: int) -> torch.Tensor:
        # Kept for compatibility/debug. The new temporal router uses current frame-wise latent difference,
        # not this denoising step-diff score.
        new_score = self.compute_diff_map(latents_before, latents_after)
        alpha = float(self.config.get("ema_alpha", 0.85))
        if self.step_diff_ema is None:
            self.step_diff_ema = new_score
        else:
            self.step_diff_ema = alpha * self.step_diff_ema + (1.0 - alpha) * new_score
        self.score_ema = self.step_diff_ema
        return self.step_diff_ema

    def should_refresh_ls_gap(self, step_idx: int) -> bool:
        # The simplified router no longer uses full large-small gap maps.
        return False

    def observe_aux(
        self,
        latents: torch.Tensor,
        cfg_gap_map: Optional[torch.Tensor] = None,
        ls_gap_map: Optional[torch.Tensor] = None,
        step_idx: int = -1,
    ):
        # Current-step maps used by the spatial cue selector.
        self.motion_map = _compute_frame_diff_map(latents.detach().float())
        self.motion_ema = self.motion_map

        if cfg_gap_map is not None:
            cfg_gap_map = cfg_gap_map.detach().float()
            self.cfg_gap_map = cfg_gap_map
            alpha = float(self.config.get("aux_ema_alpha", 0.0))
            if alpha > 0 and self.cfg_gap_ema is not None and self.cfg_gap_ema.shape == cfg_gap_map.shape:
                self.cfg_gap_ema = alpha * self.cfg_gap_ema + (1.0 - alpha) * cfg_gap_map
            else:
                self.cfg_gap_ema = cfg_gap_map
        else:
            self.cfg_gap_map = None

        spatial_cue = str(self.config.get("spatial_cue", "cfg")).lower()
        save_all_cues = bool(self.config.get("debug_save_all_cues", True))
        if spatial_cue == "warp" or save_all_cues:
            self.warp_map = _compute_warp_residual(latents.detach().float(), int(self.config.get("warp_max_shift", 2)))
        else:
            self.warp_map = None
        self.warp_ema = self.warp_map

        # Explicitly disable old optional maps.
        self.ls_gap_ema = None
        self.traj_curv_ema = None
        self.traj_flip_ema = None

    def _make_roi(
        self,
        t_len: int,
        h: int,
        w: int,
        core_t0: int,
        core_t1: int,
        core_y0: int,
        core_y1: int,
        core_x0: int,
        core_x1: int,
    ) -> Dict[str, Any]:
        outer_t0, outer_t1 = _expand_align_bounds(
            core_t0, core_t1, t_len,
            int(self.config.get("margin_t", 1)), int(self.config.get("min_crop_t", 1)), 1,
        )
        outer_y0, outer_y1 = _expand_align_bounds(
            core_y0, core_y1, h,
            int(self.config.get("margin_h", 4)), int(self.config.get("min_crop_h", 8)), int(self.config.get("align_h", 2)),
        )
        outer_x0, outer_x1 = _expand_align_bounds(
            core_x0, core_x1, w,
            int(self.config.get("margin_w", 4)), int(self.config.get("min_crop_w", 8)), int(self.config.get("align_w", 2)),
        )
        roi: Dict[str, Any] = {
            "t0": outer_t0, "t1": outer_t1,
            "y0": outer_y0, "y1": outer_y1,
            "x0": outer_x0, "x1": outer_x1,
            "core_t0": int(core_t0), "core_t1": int(core_t1),
            "core_y0": int(core_y0), "core_y1": int(core_y1),
            "core_x0": int(core_x0), "core_x1": int(core_x1),
        }
        roi["local_core_t0"] = roi["core_t0"] - roi["t0"]
        roi["local_core_t1"] = roi["core_t1"] - roi["t0"]
        roi["local_core_y0"] = roi["core_y0"] - roi["y0"]
        roi["local_core_y1"] = roi["core_y1"] - roi["y0"]
        roi["local_core_x0"] = roi["core_x0"] - roi["x0"]
        roi["local_core_x1"] = roi["core_x1"] - roi["x0"]
        return roi

    def _select_spatial_source(self, latents: torch.Tensor) -> Tuple[str, torch.Tensor]:
        spatial_cue = str(self.config.get("spatial_cue", "cfg")).lower()
        if spatial_cue == "cfg":
            if self.cfg_gap_ema is not None:
                return "cfg", self.cfg_gap_ema
            # Graceful fallback if CFG is disabled.
            motion = _compute_frame_diff_map(latents)
            return "motion_fallback_no_cfg", motion
        if spatial_cue == "motion":
            motion = self.motion_map if self.motion_map is not None else _compute_frame_diff_map(latents)
            return "motion", motion
        if spatial_cue == "warp":
            warp = self.warp_map
            if warp is None:
                warp = _compute_warp_residual(latents, int(self.config.get("warp_max_shift", 2)))
            return "warp", warp
        raise ValueError(f"Unsupported spatial_cue={spatial_cue!r}; expected cfg/motion/warp.")

    def _candidate_spatial_sources(self, latents: torch.Tensor) -> Dict[str, torch.Tensor]:
        sources: Dict[str, torch.Tensor] = {}
        if self.cfg_gap_ema is not None:
            sources["cfg"] = self.cfg_gap_ema.float()
        motion = self.motion_map if self.motion_map is not None else _compute_frame_diff_map(latents)
        sources["motion"] = motion.float()
        warp = self.warp_map
        if warp is None:
            warp = _compute_warp_residual(latents, int(self.config.get("warp_max_shift", 2)))
        sources["warp"] = warp.float()
        return sources

    def _segment_to_cube(
        self,
        spatial_source: torch.Tensor,
        segment: Tuple[int, int],
        seg_rank: int,
        seg_score: float,
        t_len: int,
        h: int,
        w: int,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], torch.Tensor, torch.Tensor]:
        s, e = segment
        # Average the chosen spatial cue inside the hard temporal segment.
        spatial_2d = spatial_source[0, s:e].mean(dim=0)
        spatial_norm = _normalize_2d(spatial_2d)

        # A tiny fixed smoothing+dilation is kept as an internal stability operation, not exposed as a method parameter.
        if min(h, w) >= 3:
            spatial_norm = F.avg_pool2d(spatial_norm[None, None], kernel_size=3, stride=1, padding=1)[0, 0]
            spatial_norm = _normalize_2d(spatial_norm)

        top_mask = _topk_binary_mask(
            spatial_norm.unsqueeze(0),
            ratio=float(self.config.get("spatial_top_ratio", 0.08)),
        )[0]
        if min(h, w) >= 3:
            top_mask = F.max_pool2d(top_mask.float()[None, None], kernel_size=3, stride=1, padding=1)[0, 0] > 0

        comps = _extract_connected_components(top_mask)
        if not comps:
            bbox = _bbox_from_mask(top_mask)
            source = "top_mask_bbox"
            comp_score = float(spatial_norm[top_mask].mean().item()) if top_mask.any() else 0.0
        else:
            best = None
            best_score = -1.0
            for comp in comps:
                mask = comp["mask"]
                score_sum = float(spatial_norm[mask].sum().item()) if mask.any() else 0.0
                if score_sum > best_score:
                    best_score = score_sum
                    best = comp
            assert best is not None
            bbox = best["bbox"]
            source = "largest_score_component"
            comp_score = best_score

        if bbox is None:
            bbox = (0, h, 0, w)
            source = "full_spatial_fallback"
            comp_score = float(spatial_norm.mean().item())

        y0, y1, x0, x1 = bbox
        roi = self._make_roi(t_len, h, w, s, e, y0, y1, x0, x1)
        roi.update({
            "seg_rank": int(seg_rank),
            "seg_score": float(seg_score),
            "comp_score": float(comp_score),
            "bbox_source": source,
        })
        entry = {
            "seg_rank": int(seg_rank),
            "segment": [int(s), int(e)],
            "seg_score": float(seg_score),
            "spatial_bbox": [int(y0), int(y1), int(x0), int(x1)],
            "bbox_source": source,
            "num_components": len(comps),
            "spatial_score_mean": float(spatial_norm.mean().item()),
            "spatial_score_max": float(spatial_norm.max().item()),
            "mask_ratio": float(top_mask.float().mean().item()),
        }
        return roi, entry, spatial_norm.detach().cpu(), top_mask.detach().cpu()

    def build_rois(self, latents: torch.Tensor, step_idx: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        bsz, _, t_len, h, w = latents.shape
        assert bsz == 1, "当前版本先只支持 batch_size=1。"

        latents = latents.detach().float()
        frame_diff_map = _compute_frame_diff_map(latents)          # [B,T,H,W]
        temporal_score = frame_diff_map.mean(dim=(2, 3))           # [B,T]

        if t_len <= 1 or float(temporal_score.max().item()) <= 1e-8:
            segments = [(0, t_len)]
            temporal_mask = torch.ones((1, t_len), device=latents.device, dtype=torch.bool)
        else:
            temporal_mask = _topk_binary_mask(
                temporal_score,
                ratio=float(self.config.get("temporal_top_ratio", 0.15)),
            )
            segments = _bool_to_segments(temporal_mask[0])
            if not segments:
                segments = [(0, t_len)]

        segment_scores: List[Tuple[float, int, int]] = []
        for s, e in segments:
            segment_scores.append((float(temporal_score[0, s:e].mean().item()), int(s), int(e)))
        segment_scores = sorted(segment_scores, reverse=True)

        max_segments = int(self.config.get("max_temporal_segments", self.config.get("max_segments", 2)))
        max_cubes = int(self.config.get("max_cubes", self.config.get("max_total_rois", 2)))
        selected_segments = segment_scores[: max(1, min(max_segments, max_cubes, len(segment_scores)))]

        spatial_name, spatial_source = self._select_spatial_source(latents)
        spatial_source = spatial_source.float()
        if spatial_source.shape[2:] != (h, w):
            raise RuntimeError(f"spatial_source shape mismatch: {tuple(spatial_source.shape)} vs latent H/W={(h, w)}")

        rois: List[Dict[str, Any]] = []
        spatial_debug: List[Dict[str, Any]] = []
        debug_tensors: Dict[str, Any] = {}
        candidate_spatial_debug: Dict[str, List[Dict[str, Any]]] = {}
        for rank, (seg_score, s, e) in enumerate(selected_segments):
            roi, entry, spatial_norm, top_mask = self._segment_to_cube(
                spatial_source=spatial_source,
                segment=(s, e),
                seg_rank=rank,
                seg_score=seg_score,
                t_len=t_len,
                h=h,
                w=w,
            )
            rois.append(roi)
            spatial_debug.append(entry)
            debug_tensors[f"seg{rank}_spatial_score_norm"] = spatial_norm
            debug_tensors[f"seg{rank}_spatial_top_mask"] = top_mask

        if bool(self.config.get("debug_save_all_cues", True)):
            for cue_name, cue_source in self._candidate_spatial_sources(latents).items():
                if cue_source.shape[2:] != (h, w):
                    continue
                candidate_spatial_debug[cue_name] = []
                for rank, (seg_score, s, e) in enumerate(selected_segments):
                    cand_roi, cand_entry, cand_score, cand_mask = self._segment_to_cube(
                        spatial_source=cue_source,
                        segment=(s, e),
                        seg_rank=rank,
                        seg_score=seg_score,
                        t_len=t_len,
                        h=h,
                        w=w,
                    )
                    cand_entry["cue"] = cue_name
                    cand_entry["roi"] = cand_roi
                    candidate_spatial_debug[cue_name].append(cand_entry)
                    debug_tensors[f"candidate_{cue_name}_seg{rank}_spatial_score_norm"] = cand_score
                    debug_tensors[f"candidate_{cue_name}_seg{rank}_spatial_top_mask"] = cand_mask

        if not rois:
            rois = [self._make_roi(t_len, h, w, 0, t_len, 0, h, 0, w)]

        # Sort for deterministic crop order and keep a copy for external debug tools.
        rois = sorted(rois, key=lambda r: (r["core_t0"], r["core_y0"], r["core_x0"]))
        self.prev_rois = [r.copy() for r in rois]

        total_core = sum(
            (r["core_t1"] - r["core_t0"]) *
            (r["core_y1"] - r["core_y0"]) *
            (r["core_x1"] - r["core_x0"])
            for r in rois
        )
        total_outer = sum(
            (r["t1"] - r["t0"]) *
            (r["y1"] - r["y0"]) *
            (r["x1"] - r["x0"])
            for r in rois
        )
        full_volume = max(1, t_len * h * w)

        topk = min(int(self.config.get("debug_topk_frames", 5)), temporal_score.shape[1])
        top_vals, top_idx = torch.topk(temporal_score[0], k=topk)
        debug: Dict[str, Any] = {
            "step_idx": int(step_idx),
            "router_mode": "two_stage_cube",
            "temporal_cue": "frame_diff",
            "spatial_cue": spatial_name,
            "temporal_top_ratio": float(self.config.get("temporal_top_ratio", 0.15)),
            "max_temporal_segments": int(max_segments),
            "spatial_top_ratio": float(self.config.get("spatial_top_ratio", 0.08)),
            "max_cubes": int(max_cubes),
            "segments": [(int(s), int(e)) for _, s, e in selected_segments],
            "segment_scores": [float(x[0]) for x in selected_segments],
            "temporal_top_frames": [int(x) for x in top_idx.tolist()],
            "temporal_top_values": [float(v) for v in top_vals.tolist()],
            "rois": rois,
            "core_ratio": float(total_core / full_volume),
            "outer_ratio": float(total_outer / full_volume),
            "spatial_debug": spatial_debug,
            "candidate_spatial_debug": candidate_spatial_debug,
        }

        save_dir = self.config.get("save_debug_dir", None)
        debug_every = int(self.config.get("debug_every", 1))
        if save_dir is not None and (step_idx % max(1, debug_every) == 0):
            os.makedirs(save_dir, exist_ok=True)
            payload: Dict[str, Any] = {
                "debug": debug,
                "frame_diff_map": frame_diff_map.detach().cpu(),
                "temporal_score": temporal_score.detach().cpu(),
                "temporal_mask": temporal_mask.detach().cpu(),
                "spatial_source_name": spatial_name,
                "spatial_source": spatial_source.detach().cpu(),
                "cfg_gap_ema": None if self.cfg_gap_ema is None else self.cfg_gap_ema.detach().cpu(),
                "motion_map": None if self.motion_map is None else self.motion_map.detach().cpu(),
                "warp_map": None if self.warp_map is None else self.warp_map.detach().cpu(),
                "step_diff_ema": None if self.step_diff_ema is None else self.step_diff_ema.detach().cpu(),
            }
            payload.update(debug_tensors)
            torch.save(payload, os.path.join(save_dir, f"router_step_{step_idx:03d}.pt"))
            with open(os.path.join(save_dir, f"router_step_{step_idx:03d}.json"), "w", encoding="utf-8") as f:
                json.dump(debug, f, ensure_ascii=False, indent=2, default=str)

        return rois, debug
