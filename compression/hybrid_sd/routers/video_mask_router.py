import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


def _topk_binary_mask(scores: torch.Tensor, ratio: float) -> torch.Tensor:
    """
    scores: [B, T] or [B, H, W]
    return: bool mask with same shape
    """
    bsz = scores.shape[0]
    flat = scores.reshape(bsz, -1)
    total = flat.shape[1]
    k = max(1, min(total, int(math.ceil(total * ratio))))
    topk_vals = torch.topk(flat, k=k, dim=1).values
    threshold = topk_vals[:, -1].unsqueeze(1)
    return (flat >= threshold).reshape_as(scores)


def _bool_to_segments(mask_1d: torch.Tensor) -> List[Tuple[int, int]]:
    """
    输入 bool 1D mask，输出连续区间，右端点是 exclusive
    例如 [0,1,1,0,1] -> [(1,3), (4,5)]
    """
    segments = []
    start = None
    values = mask_1d.tolist()
    for i, v in enumerate(values):
        if v and start is None:
            start = i
        elif (not v) and start is not None:
            segments.append((start, i))
            start = None
    if start is not None:
        segments.append((start, len(values)))
    return segments


def _bbox_from_mask(mask_2d: torch.Tensor) -> Optional[Tuple[int, int, int, int]]:
    """
    mask_2d: [H, W] bool
    return: (y0, y1, x0, x1), right/bottom exclusive
    """
    ys, xs = torch.where(mask_2d)
    if ys.numel() == 0:
        return None
    y0 = int(ys.min().item())
    y1 = int(ys.max().item()) + 1
    x0 = int(xs.min().item())
    x1 = int(xs.max().item()) + 1
    return y0, y1, x0, x1


def _expand_align_bounds(
    start: int,
    end: int,
    limit: int,
    margin: int,
    min_size: int,
    align: int,
) -> Tuple[int, int]:
    """
    对 [start, end) 做扩张，并对齐到 align。
    """
    start = max(0, start - margin)
    end = min(limit, end + margin)

    if end <= start:
        end = min(limit, start + 1)

    size = end - start
    size = max(size, min_size)

    if align > 1:
        size = int(math.ceil(size / align) * align)

    size = min(size, limit)

    center = 0.5 * (start + end)
    new_start = int(round(center - size / 2.0))
    new_start = max(0, min(new_start, limit - size))
    new_end = new_start + size
    return new_start, new_end


def _interval_iou(a0: int, a1: int, b0: int, b1: int) -> float:
    inter = max(0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    if union <= 0:
        return 0.0
    return inter / union


def _roi_iou(a: Dict[str, int], b: Dict[str, int]) -> float:
    it = _interval_iou(a["core_t0"], a["core_t1"], b["core_t0"], b["core_t1"])
    iy = _interval_iou(a["core_y0"], a["core_y1"], b["core_y0"], b["core_y1"])
    ix = _interval_iou(a["core_x0"], a["core_x1"], b["core_x0"], b["core_x1"])
    return it * iy * ix


class VideoMaskRouter:
    """
    思路二对应的 ROI router：
    1) 用 step-to-step latent diff 更新 score_ema
    2) 先按时间选 hard segments
    3) 再在每个 hard segment 内按空间选 hard bbox
    4) bbox + temporal segment -> tube / ROI volume
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config: Dict[str, Any] = {
            # score update
            "ema_alpha": 0.85,
            "relative_diff": True,
            # temporal routing
            "temporal_top_ratio": 0.15,
            "temporal_dilate": 1,
            "max_segments": 2,
            # spatial routing
            "spatial_top_ratio": 0.08,
            "spatial_dilate": 1,
            # outer crop margins
            "margin_t": 1,
            "margin_h": 4,
            "margin_w": 4,
            # min crop size in latent space
            "min_crop_t": 1,
            "min_crop_h": 8,
            "min_crop_w": 8,
            # Wan patchify alignment
            "align_h": 2,
            "align_w": 2,
            # temporal smoothing
            "smooth_iou_thresh": 0.25,
            "smooth_momentum": 0.6,
            # debug
            "debug_every": 1,
            "debug_topk_frames": 5,
            "save_debug_dir": None,
        }
        if config is not None:
            self.config.update(config)
        self.reset()

    def update_config(self, config: Optional[Dict[str, Any]] = None):
        if config is not None:
            self.config.update(config)

    def reset(self):
        self.score_ema: Optional[torch.Tensor] = None  # [B, T, H, W]
        self.prev_rois: List[Dict[str, int]] = []

    def compute_diff_map(self, latents_before: torch.Tensor, latents_after: torch.Tensor) -> torch.Tensor:
        """
        latents shape: [B, C, T, H, W]
        return diff_map: [B, T, H, W]
        """
        diff = (latents_before - latents_after).abs().mean(dim=1)

        if self.config.get("relative_diff", True):
            denom = latents_before.abs().mean(dim=1).clamp_min(1e-6)
            diff = diff / denom

        return diff.float()

    def observe(self, latents_before: torch.Tensor, latents_after: torch.Tensor, step_idx: int) -> torch.Tensor:
        """
        在每个 denoising step 结束后更新 score_ema。
        """
        new_score = self.compute_diff_map(latents_before, latents_after)

        if self.score_ema is None:
            self.score_ema = new_score
        else:
            alpha = float(self.config.get("ema_alpha", 0.85))
            self.score_ema = alpha * self.score_ema + (1.0 - alpha) * new_score

        return self.score_ema

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
    ) -> Dict[str, int]:
        """
        core 是真正要回填的区域
        outer 是带 margin 的 crop 区域（给大模型更多上下文）
        """
        outer_t0, outer_t1 = _expand_align_bounds(
            core_t0, core_t1, t_len,
            self.config["margin_t"],
            self.config["min_crop_t"],
            1,
        )
        outer_y0, outer_y1 = _expand_align_bounds(
            core_y0, core_y1, h,
            self.config["margin_h"],
            self.config["min_crop_h"],
            self.config["align_h"],
        )
        outer_x0, outer_x1 = _expand_align_bounds(
            core_x0, core_x1, w,
            self.config["margin_w"],
            self.config["min_crop_w"],
            self.config["align_w"],
        )

        roi = {
            "t0": outer_t0, "t1": outer_t1,
            "y0": outer_y0, "y1": outer_y1,
            "x0": outer_x0, "x1": outer_x1,
            "core_t0": core_t0, "core_t1": core_t1,
            "core_y0": core_y0, "core_y1": core_y1,
            "core_x0": core_x0, "core_x1": core_x1,
        }

        roi["local_core_t0"] = roi["core_t0"] - roi["t0"]
        roi["local_core_t1"] = roi["core_t1"] - roi["t0"]
        roi["local_core_y0"] = roi["core_y0"] - roi["y0"]
        roi["local_core_y1"] = roi["core_y1"] - roi["y0"]
        roi["local_core_x0"] = roi["core_x0"] - roi["x0"]
        roi["local_core_x1"] = roi["core_x1"] - roi["x0"]

        return roi

    def _smooth_rois(self, rois: List[Dict[str, int]]) -> List[Dict[str, int]]:
        """
        用上一 step 的 ROI 做简单时间平滑，减少 bbox 抖动。
        这里只平滑 outer crop，不改当前 core。
        """
        if not self.prev_rois or not rois:
            return rois

        momentum = float(self.config.get("smooth_momentum", 0.6))
        iou_thresh = float(self.config.get("smooth_iou_thresh", 0.25))

        smoothed_rois = []
        used_prev = set()

        for roi in rois:
            best_idx = -1
            best_iou = 0.0
            for j, prev_roi in enumerate(self.prev_rois):
                if j in used_prev:
                    continue
                iou = _roi_iou(roi, prev_roi)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = j

            if best_idx >= 0 and best_iou >= iou_thresh:
                prev_roi = self.prev_rois[best_idx]
                used_prev.add(best_idx)

                out = roi.copy()
                for k in ["t0", "t1", "y0", "y1", "x0", "x1"]:
                    out[k] = int(round(momentum * prev_roi[k] + (1.0 - momentum) * roi[k]))

                # outer 不能侵入 core
                out["t0"] = min(out["t0"], out["core_t0"])
                out["y0"] = min(out["y0"], out["core_y0"])
                out["x0"] = min(out["x0"], out["core_x0"])

                out["t1"] = max(out["t1"], out["core_t1"])
                out["y1"] = max(out["y1"], out["core_y1"])
                out["x1"] = max(out["x1"], out["core_x1"])

                out["local_core_t0"] = out["core_t0"] - out["t0"]
                out["local_core_t1"] = out["core_t1"] - out["t0"]
                out["local_core_y0"] = out["core_y0"] - out["y0"]
                out["local_core_y1"] = out["core_y1"] - out["y0"]
                out["local_core_x0"] = out["core_x0"] - out["x0"]
                out["local_core_x1"] = out["core_x1"] - out["x0"]

                smoothed_rois.append(out)
            else:
                smoothed_rois.append(roi)

        return smoothed_rois

    def build_rois(self, latents: torch.Tensor, step_idx: int) -> Tuple[List[Dict[str, int]], Dict[str, Any]]:
        """
        根据上一轮累计好的 score_ema 选 ROI。
        latents: [B, C, T, H, W]
        """
        bsz, _, t_len, h, w = latents.shape
        assert bsz == 1, "当前 ROI refine 版本先只支持 batch_size=1。"

        if self.score_ema is None:
            # warm start fallback
            full_roi = self._make_roi(t_len, h, w, 0, t_len, 0, h, 0, w)
            debug = {
                "step_idx": step_idx,
                "router_warm_start": True,
                "segments": [(0, t_len)],
                "rois": [full_roi],
            }
            self.prev_rois = [full_roi]
            return [full_roi], debug

        score = self.score_ema  # [1, T, H, W]

        # -------- 1) temporal routing --------
        temporal_score = score.mean(dim=(2, 3))  # [1, T]
        temporal_mask = _topk_binary_mask(
            temporal_score,
            ratio=float(self.config["temporal_top_ratio"]),
        )

        dilate_t = int(self.config["temporal_dilate"])
        if dilate_t > 0:
            temporal_mask = F.max_pool1d(
                temporal_mask.float().unsqueeze(1),
                kernel_size=2 * dilate_t + 1,
                stride=1,
                padding=dilate_t,
            ).squeeze(1) > 0

        segments = _bool_to_segments(temporal_mask[0])

        segment_scores = []
        for s, e in segments:
            seg_score = float(temporal_score[0, s:e].mean().item())
            segment_scores.append((seg_score, s, e))

        segment_scores.sort(reverse=True, key=lambda x: x[0])
        segment_scores = segment_scores[: int(self.config["max_segments"])]

        # -------- 2) spatial routing inside each segment --------
        rois = []
        spatial_debug = []

        for seg_rank, (seg_score, s, e) in enumerate(segment_scores):
            spatial_score = score[0, s:e].mean(dim=0)  # [H, W]

            spatial_mask = _topk_binary_mask(
                spatial_score.unsqueeze(0),
                ratio=float(self.config["spatial_top_ratio"]),
            )[0]

            dilate_hw = int(self.config["spatial_dilate"])
            if dilate_hw > 0:
                spatial_mask = F.max_pool2d(
                    spatial_mask.float().unsqueeze(0).unsqueeze(0),
                    kernel_size=2 * dilate_hw + 1,
                    stride=1,
                    padding=dilate_hw,
                ).squeeze(0).squeeze(0) > 0

            bbox = _bbox_from_mask(spatial_mask)
            if bbox is None:
                continue

            y0, y1, x0, x1 = bbox
            roi = self._make_roi(
                t_len=t_len,
                h=h,
                w=w,
                core_t0=s,
                core_t1=e,
                core_y0=y0,
                core_y1=y1,
                core_x0=x0,
                core_x1=x1,
            )
            roi["seg_rank"] = seg_rank
            roi["seg_score"] = seg_score
            rois.append(roi)

            spatial_debug.append({
                "segment": [s, e],
                "seg_score": seg_score,
                "bbox": [y0, y1, x0, x1],
            })

        if len(rois) == 0:
            full_roi = self._make_roi(t_len, h, w, 0, t_len, 0, h, 0, w)
            rois = [full_roi]

        rois = sorted(rois, key=lambda r: (r["core_t0"], r["core_y0"], r["core_x0"]))
        rois = self._smooth_rois(rois)
        self.prev_rois = [r.copy() for r in rois]

        # -------- debug summary --------
        topk_frames = min(int(self.config.get("debug_topk_frames", 5)), temporal_score.shape[1])
        top_vals, top_idx = torch.topk(temporal_score[0], k=topk_frames)

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

        debug = {
            "step_idx": step_idx,
            "router_warm_start": False,
            "score_mean": float(score.mean().item()),
            "score_std": float(score.std().item()),
            "score_max": float(score.max().item()),
            "temporal_top_frames": top_idx.tolist(),
            "temporal_top_values": [float(v) for v in top_vals.tolist()],
            "segments": [(s, e) for _, s, e in segment_scores],
            "rois": rois,
            "core_ratio": float(total_core / full_volume),
            "outer_ratio": float(total_outer / full_volume),
            "spatial_debug": spatial_debug,
        }

        save_dir = self.config.get("save_debug_dir", None)
        debug_every = int(self.config.get("debug_every", 1))
        if save_dir is not None and (step_idx % max(1, debug_every) == 0):
            os.makedirs(save_dir, exist_ok=True)
            torch.save({
                "score_ema": score.detach().cpu(),
                "temporal_score": temporal_score.detach().cpu(),
                "temporal_mask": temporal_mask.detach().cpu(),
                "debug": debug,
            }, os.path.join(save_dir, f"router_step_{step_idx:03d}.pt"))

            with open(os.path.join(save_dir, f"router_step_{step_idx:03d}.json"), "w", encoding="utf-8") as f:
                json.dump(debug, f, ensure_ascii=False, indent=2, default=str)

        return rois, debug
