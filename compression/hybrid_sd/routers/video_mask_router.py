import json
import math
import os
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


def _topk_binary_mask(scores: torch.Tensor, ratio: float) -> torch.Tensor:
    bsz = scores.shape[0]
    flat = scores.reshape(bsz, -1)
    total = flat.shape[1]
    k = max(1, min(total, int(math.ceil(total * ratio))))
    topk_vals = torch.topk(flat, k=k, dim=1).values
    threshold = topk_vals[:, -1].unsqueeze(1)
    return (flat >= threshold).reshape_as(scores)


def _bool_to_segments(mask_1d: torch.Tensor) -> List[Tuple[int, int]]:
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
    ys, xs = torch.where(mask_2d)
    if ys.numel() == 0:
        return None
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def _bbox_area(bbox: Tuple[int, int, int, int]) -> int:
    y0, y1, x0, x1 = bbox
    return max(0, y1 - y0) * max(0, x1 - x0)


def _bbox_iou_2d(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ay0, ay1, ax0, ax1 = a
    by0, by1, bx0, bx1 = b
    iy = max(0, min(ay1, by1) - max(ay0, by0))
    ix = max(0, min(ax1, bx1) - max(ax0, bx0))
    inter = iy * ix
    if inter <= 0:
        return 0.0
    ua = _bbox_area(a) + _bbox_area(b) - inter
    return inter / max(1, ua)


def _union_bboxes(bbs: List[Tuple[int, int, int, int]]) -> Tuple[int, int, int, int]:
    y0 = min(b[0] for b in bbs)
    y1 = max(b[1] for b in bbs)
    x0 = min(b[2] for b in bbs)
    x1 = max(b[3] for b in bbs)
    return y0, y1, x0, x1


def _temporal_topq_mean(x: torch.Tensor, top_ratio: float) -> torch.Tensor:
    """x: [B,T,H,W] -> temporal score [B,T]：每帧取空间 top-q 像素均值。"""
    b, t, h, w = x.shape
    flat = x.reshape(b, t, -1)
    hw = flat.shape[2]
    k = max(1, int(math.ceil(hw * float(top_ratio))))
    vals, _ = torch.topk(flat, k=k, dim=2)
    return vals.mean(dim=2)


def _compute_warp_residual(latents: torch.Tensor, max_shift: int) -> torch.Tensor:
    """
    便宜 warp：对 z_{t-1} 做整数平移搜索，使与 z_t 的 L1 最小；残差 |z_t - W(z_{t-1})| 通道均值。
    latents [B,C,T,H,W] -> [B,T,H,W]
    """
    b, c, t, h, w = latents.shape
    out = torch.zeros(b, t, h, w, device=latents.device, dtype=torch.float32)
    if t <= 1:
        return out
    shifts = list(range(-max_shift, max_shift + 1))
    for ti in range(1, t):
        prev = latents[:, :, ti - 1]
        curr = latents[:, :, ti]
        best: Optional[torch.Tensor] = None
        for dy in shifts:
            for dx in shifts:
                rolled = torch.roll(prev, shifts=(dy, dx), dims=(2, 3))
                res = (curr - rolled).abs().mean(dim=1)
                m = float(res.mean().item())
                if best is None or m < best[0]:
                    best = (m, res)
        assert best is not None
        out[:, ti] = best[1]
    out[:, 0] = out[:, 1]
    return out


def _expand_align_bounds(
    start: int, end: int, limit: int, margin: int, min_size: int, align: int
) -> Tuple[int, int]:
    start = max(0, start - margin)
    end = min(limit, end + margin)
    if end <= start:
        end = min(limit, start + 1)
    size = max(end - start, min_size)
    if align > 1:
        size = int(math.ceil(size / align) * align)
    size = min(size, limit)
    center = 0.5 * (start + end)
    new_start = int(round(center - size / 2.0))
    new_start = max(0, min(new_start, limit - size))
    return new_start, new_start + size


def _interval_iou(a0: int, a1: int, b0: int, b1: int) -> float:
    inter = max(0, min(a1, b1) - max(a0, b0))
    union = max(a1, b1) - min(a0, b0)
    if union <= 0:
        return 0.0
    return inter / union


def _roi_iou(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    it = _interval_iou(a["core_t0"], a["core_t1"], b["core_t0"], b["core_t1"])
    iy = _interval_iou(a["core_y0"], a["core_y1"], b["core_y0"], b["core_y1"])
    ix = _interval_iou(a["core_x0"], a["core_x1"], b["core_x0"], b["core_x1"])
    return it * iy * ix


def _smooth2d(x: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 1:
        return x
    if k % 2 == 0:
        k += 1
    return F.avg_pool2d(x[None, None], kernel_size=k, stride=1, padding=k // 2)[0, 0]


def _smooth1d(x: torch.Tensor, k: int) -> torch.Tensor:
    if k <= 1:
        return x
    if k % 2 == 0:
        k += 1
    return F.avg_pool1d(x[None, None], kernel_size=k, stride=1, padding=k // 2)[0, 0]


def _normalize_map_per_sample(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if x is None:
        return None
    if x.ndim != 4:
        raise ValueError(f"Expected [B,T,H,W], got shape={tuple(x.shape)}")
    flat = x.reshape(x.shape[0], -1)
    xmin = flat.min(dim=1).values[:, None, None, None]
    xmax = flat.max(dim=1).values[:, None, None, None]
    return (x - xmin) / (xmax - xmin).clamp_min(1e-6)


def _extract_connected_components(mask_2d: torch.Tensor, min_area: int = 1) -> List[Dict[str, Any]]:
    h, w = mask_2d.shape
    mask_cpu = mask_2d.detach().to(torch.bool).cpu()
    visited = torch.zeros((h, w), dtype=torch.bool)
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    comps = []
    for y in range(h):
        for x in range(w):
            if not mask_cpu[y, x] or visited[y, x]:
                continue

            q = deque([(y, x)])
            visited[y, x] = True
            pts = []

            while q:
                cy, cx = q.popleft()
                pts.append((cy, cx))
                for dy, dx in neighbors:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and mask_cpu[ny, nx] and (not visited[ny, nx]):
                        visited[ny, nx] = True
                        q.append((ny, nx))

            if len(pts) < min_area:
                continue

            out_mask = torch.zeros((h, w), dtype=torch.bool, device=mask_2d.device)
            for py, px in pts:
                out_mask[py, px] = True

            bbox = _bbox_from_mask(out_mask)
            comps.append({
                "mask": out_mask,
                "bbox": bbox,
                "pixels": len(pts),
            })

    return comps


def _mass_window_1d(v: torch.Tensor, keep_ratio: float) -> Tuple[int, int]:
    n = v.numel()
    if n == 0:
        return 0, 0

    total = float(v.sum().item())
    if total <= 1e-8:
        peak = int(torch.argmax(v).item())
        return peak, min(peak + 1, n)

    target = keep_ratio * total
    peak = int(torch.argmax(v).item())

    left = peak
    right = peak
    cur = float(v[peak].item())

    while cur < target and (left > 0 or right < n - 1):
        left_val = float(v[left - 1].item()) if left > 0 else -1.0
        right_val = float(v[right + 1].item()) if right < n - 1 else -1.0

        if right_val >= left_val and right < n - 1:
            right += 1
            cur += float(v[right].item())
        elif left > 0:
            left -= 1
            cur += float(v[left].item())
        else:
            break

    return left, right + 1


def _bbox_from_projection_mass(
    score_2d: torch.Tensor, keep_ratio_h: float, keep_ratio_w: float, blur_kernel: int
) -> Tuple[int, int, int, int]:
    row_energy = _smooth1d(score_2d.sum(dim=1), blur_kernel)
    col_energy = _smooth1d(score_2d.sum(dim=0), blur_kernel)
    y0, y1 = _mass_window_1d(row_energy, keep_ratio_h)
    x0, x1 = _mass_window_1d(col_energy, keep_ratio_w)
    return int(y0), int(y1), int(x0), int(x1)


class VideoMaskRouter:
    """
    step_diff + cfg_gap + ls_gap +（可选）motion / warp + 轨迹曲率/翻转；
    时域 top-q 池化；空域可选 tube（逐帧 CC + IoU 链）替代段均值大框。
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config: Dict[str, Any] = {
            "ema_alpha": 0.85,
            "aux_ema_alpha": 0.70,
            "relative_diff": True,

            # 只保留 4 个实际使用的 cue
            "step_diff_weight": 1.0,
            "cfg_gap_weight": 0.8,
            "warp_weight": 0.8,
            "traj_curv_weight": 0.6,

            # 这些保留字段，但默认不使用
            "ls_gap_weight": 0.0,
            "motion_weight": 0.0,
            "traj_flip_weight": 0.0,

            "use_warp_cue": True,
            "warp_max_shift": 2,

            "ls_gap_every": 0,
            "force_ls_gap_first": False,

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

        if config is not None:
            self.config.update(config)

        self.reset()

    def update_config(self, config: Optional[Dict[str, Any]] = None):
        if config is not None:
            self.config.update(config)

    def reset(self):
        self.score_ema: Optional[torch.Tensor] = None
        self.traj_curv_ema: Optional[torch.Tensor] = None
        self.traj_flip_ema: Optional[torch.Tensor] = None
        self.cfg_gap_ema: Optional[torch.Tensor] = None
        self.ls_gap_ema: Optional[torch.Tensor] = None
        self.motion_ema: Optional[torch.Tensor] = None
        self.warp_ema: Optional[torch.Tensor] = None
        self._prev_latents_delta: Optional[torch.Tensor] = None
        self.prev_rois: List[Dict[str, Any]] = []
        self.last_ls_gap_step: int = -10**9

    def _active_cue_names(self) -> List[str]:
        """
        当前版本只实际使用 4 个 cue：
          step_diff + cfg_gap + warp + traj_curv
        """
        return ["step_diff", "cfg_gap", "warp", "traj_curv"]

    def _repair_roi_geometry(
        self,
        roi: Dict[str, Any],
        t_len: int,
        h: int,
        w: int,
    ) -> Dict[str, Any]:
        """
        对 smoothing 后的 ROI 做几何修复：
        - core 先 clamp 到合法范围
        - outer 不再直接相信 smoothing 后的坐标
        - 而是重新基于 core + margin + align 生成合法 outer
        这样能彻底避免 outer 过紧 / 不对齐，导致 paste clipped。
        """
        core_t0 = max(0, min(int(roi["core_t0"]), t_len - 1))
        core_t1 = max(core_t0 + 1, min(int(roi["core_t1"]), t_len))

        core_y0 = max(0, min(int(roi["core_y0"]), h - 1))
        core_y1 = max(core_y0 + 1, min(int(roi["core_y1"]), h))

        core_x0 = max(0, min(int(roi["core_x0"]), w - 1))
        core_x1 = max(core_x0 + 1, min(int(roi["core_x1"]), w))

        repaired = self._make_roi(
            t_len=t_len,
            h=h,
            w=w,
            core_t0=core_t0,
            core_t1=core_t1,
            core_y0=core_y0,
            core_y1=core_y1,
            core_x0=core_x0,
            core_x1=core_x1,
        )

        # 保留非几何元信息
        geom_keys = {
            "t0", "t1", "y0", "y1", "x0", "x1",
            "core_t0", "core_t1", "core_y0", "core_y1", "core_x0", "core_x1",
            "local_core_t0", "local_core_t1", "local_core_y0", "local_core_y1",
            "local_core_x0", "local_core_x1",
        }
        for k, v in roi.items():
            if k not in geom_keys:
                repaired[k] = v

        repaired["geometry_repaired"] = True
        return repaired

    def _repair_all_rois(
        self,
        rois: List[Dict[str, Any]],
        t_len: int,
        h: int,
        w: int,
    ) -> List[Dict[str, Any]]:
        return [self._repair_roi_geometry(r, t_len=t_len, h=h, w=w) for r in rois]

    def compute_diff_map(self, latents_before: torch.Tensor, latents_after: torch.Tensor) -> torch.Tensor:
        diff = (latents_before - latents_after).abs().mean(dim=1)
        if self.config.get("relative_diff", True):
            denom = latents_before.abs().mean(dim=1).clamp_min(1e-6)
            diff = diff / denom
        return diff.float()

    def compute_motion_map(self, latents: torch.Tensor) -> torch.Tensor:
        if latents.shape[2] <= 1:
            return torch.zeros(
                (latents.shape[0], 1, latents.shape[3], latents.shape[4]),
                device=latents.device,
                dtype=torch.float32,
            )

        diff_t = (latents[:, :, 1:] - latents[:, :, :-1]).abs().mean(dim=1)

        motion = torch.zeros(
            (latents.shape[0], latents.shape[2], latents.shape[3], latents.shape[4]),
            device=latents.device,
            dtype=diff_t.dtype,
        )
        count = torch.zeros_like(motion)

        motion[:, 1:] += diff_t
        count[:, 1:] += 1
        motion[:, :-1] += diff_t
        count[:, :-1] += 1

        motion = motion / count.clamp_min(1.0)

        k = int(self.config.get("motion_blur_kernel", 3))
        if k > 1:
            if k % 2 == 0:
                k += 1
            b, t, h, w = motion.shape
            # [B*T,H,W] 不能复用 _smooth2d（会对 3D 多叠一维变成 5D）；用 [N,1,H,W] 逐帧平滑
            x = motion.reshape(b * t, 1, h, w)
            motion = F.avg_pool2d(x, kernel_size=k, stride=1, padding=k // 2).reshape(b, t, h, w)

        return motion.float()

    def observe(self, latents_before: torch.Tensor, latents_after: torch.Tensor, step_idx: int) -> torch.Tensor:
        new_score = self.compute_diff_map(latents_before, latents_after)

        alpha = float(self.config.get("ema_alpha", 0.85))
        if self.score_ema is None:
            self.score_ema = new_score
        else:
            self.score_ema = alpha * self.score_ema + (1.0 - alpha) * new_score

        raw_delta = (latents_before - latents_after).float()
        b, c, t, h, w = raw_delta.shape
        zero_like = torch.zeros(b, t, h, w, device=raw_delta.device, dtype=torch.float32)
        prev = self._prev_latents_delta
        if prev is None or prev.shape != raw_delta.shape:
            curv_map = zero_like
            flip_map = zero_like
        else:
            dcur = raw_delta - prev
            curv_map = dcur.abs().mean(dim=1)
            a = raw_delta
            b_ = prev
            cos_map = (a * b_).sum(dim=1) / (
                a.norm(dim=1).clamp_min(1e-6) * b_.norm(dim=1).clamp_min(1e-6)
            )
            flip_map = (1.0 - cos_map.clamp(-1.0, 1.0)).clamp_min(0.0)

        self._prev_latents_delta = raw_delta.detach()

        if self.traj_curv_ema is None:
            self.traj_curv_ema = curv_map
        else:
            self.traj_curv_ema = alpha * self.traj_curv_ema + (1.0 - alpha) * curv_map

        if self.traj_flip_ema is None:
            self.traj_flip_ema = flip_map
        else:
            self.traj_flip_ema = alpha * self.traj_flip_ema + (1.0 - alpha) * flip_map

        return self.score_ema

    def should_refresh_ls_gap(self, step_idx: int) -> bool:
        every = int(self.config.get("ls_gap_every", 0))
        force_first = bool(self.config.get("force_ls_gap_first", True))

        if self.ls_gap_ema is None and force_first:
            return True

        if every > 0 and (step_idx - self.last_ls_gap_step) >= every:
            return True

        return False

    def observe_aux(
        self,
        latents: torch.Tensor,
        cfg_gap_map: Optional[torch.Tensor] = None,
        ls_gap_map: Optional[torch.Tensor] = None,
        step_idx: int = -1,
    ):
        alpha = float(self.config.get("aux_ema_alpha", 0.70))

        # warp 是 motion 的升级版：当前只保留 warp，不再实际使用 motion
        use_warp = bool(self.config.get("use_warp_cue", True)) and float(
            self.config.get("warp_weight", 0.0)
        ) > 0.0
        if use_warp:
            max_shift = int(self.config.get("warp_max_shift", 2))
            warp_map = _compute_warp_residual(latents, max_shift)
            if self.warp_ema is None:
                self.warp_ema = warp_map
            else:
                self.warp_ema = alpha * self.warp_ema + (1.0 - alpha) * warp_map
        else:
            self.warp_ema = None

        # 当前版本默认不再使用 motion
        self.motion_ema = None

        if cfg_gap_map is not None:
            cfg_gap_map = cfg_gap_map.float()
            if self.cfg_gap_ema is None:
                self.cfg_gap_ema = cfg_gap_map
            else:
                self.cfg_gap_ema = alpha * self.cfg_gap_ema + (1.0 - alpha) * cfg_gap_map

        # 当前版本默认不再使用 ls_gap，但接口保留
        if ls_gap_map is not None and float(self.config.get("ls_gap_weight", 0.0)) > 0.0:
            ls_gap_map = ls_gap_map.float()
            if self.ls_gap_ema is None:
                self.ls_gap_ema = ls_gap_map
            else:
                self.ls_gap_ema = alpha * self.ls_gap_ema + (1.0 - alpha) * ls_gap_map
            self.last_ls_gap_step = step_idx
        else:
            self.ls_gap_ema = None

    def _compose_score_maps(self) -> Dict[str, Optional[torch.Tensor]]:
        # 当前版本只实际使用 4 个 cue
        cues = {
            "step_diff": _normalize_map_per_sample(self.score_ema),
            "cfg_gap": _normalize_map_per_sample(self.cfg_gap_ema),
            "warp": _normalize_map_per_sample(self.warp_ema),
            "traj_curv": _normalize_map_per_sample(self.traj_curv_ema),
        }

        weights = {
            "step_diff": float(self.config.get("step_diff_weight", 1.0)),
            "cfg_gap": float(self.config.get("cfg_gap_weight", 0.8)),
            "warp": float(self.config.get("warp_weight", 0.8)),
            "traj_curv": float(self.config.get("traj_curv_weight", 0.6)),
        }

        combined: Optional[torch.Tensor] = None
        wsum = 0.0
        for name in self._active_cue_names():
            m = cues.get(name, None)
            if m is None:
                continue
            w = max(0.0, weights[name])
            if w <= 0:
                continue
            combined = m * w if combined is None else combined + m * w
            wsum += w

        if combined is None:
            if self.score_ema is not None:
                combined = torch.zeros_like(self.score_ema)
            else:
                raise RuntimeError("No active cue available to compose score.")
            wsum = 1.0

        combined = combined / max(wsum, 1e-6)
        cues["combined"] = combined
        return cues

    def _frame_best_component_bbox(
        self,
        spatial_2d: torch.Tensor,
        full_area: int,
    ) -> Tuple[Optional[Tuple[int, int, int, int]], float, List[Dict[str, Any]]]:
        spatial_score_blur = _smooth2d(
            spatial_2d.float(),
            int(self.config["spatial_blur_kernel"]),
        )
        smin = float(spatial_score_blur.min().item())
        smax = float(spatial_score_blur.max().item())
        spatial_score_norm = (spatial_score_blur - smin) / max(1e-6, (smax - smin))

        seed_topk = _topk_binary_mask(
            spatial_score_norm.unsqueeze(0),
            ratio=float(self.config["spatial_top_ratio"]),
        )[0]

        peak_ratio = float(self.config["spatial_peak_ratio"])
        maxv = float(spatial_score_norm.max().item())
        peak_gate = spatial_score_norm >= (peak_ratio * maxv) if maxv > 0 else spatial_score_norm > 0

        seed_mask = seed_topk & peak_gate
        if seed_mask.sum().item() == 0:
            seed_mask = seed_topk

        dilate_hw = int(self.config["spatial_dilate"])
        if dilate_hw > 0:
            seed_mask = F.max_pool2d(
                seed_mask.float().unsqueeze(0).unsqueeze(0),
                kernel_size=2 * dilate_hw + 1,
                stride=1,
                padding=dilate_hw,
            ).squeeze(0).squeeze(0) > 0

        comps = _extract_connected_components(
            seed_mask,
            min_area=int(self.config["spatial_cc_min_area"]),
        )

        min_ratio = float(self.config["spatial_min_bbox_ratio"])
        max_ratio = float(self.config["spatial_max_bbox_ratio"])

        comp_infos: List[Dict[str, Any]] = []
        best_bbox: Optional[Tuple[int, int, int, int]] = None
        best_sum = 0.0

        for comp_id, comp in enumerate(comps):
            bbox = comp["bbox"]
            if bbox is None:
                continue
            area_ratio = _bbox_area(bbox) / full_area
            mask = comp["mask"]

            score_mean = float(spatial_score_norm[mask].mean().item()) if mask.any() else 0.0
            score_sum = float(spatial_score_norm[mask].sum().item()) if mask.any() else 0.0

            valid = bool(min_ratio <= area_ratio <= max_ratio)
            comp_infos.append({
                "comp_id": comp_id,
                "bbox": list(bbox),
                "pixels": int(comp["pixels"]),
                "bbox_area_ratio": float(area_ratio),
                "score_mean": score_mean,
                "score_sum": score_sum,
                "valid": valid,
            })

            if valid and score_sum > best_sum:
                best_sum = score_sum
                best_bbox = bbox

        return best_bbox, float(best_sum), comp_infos

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
            self.config["margin_t"], self.config["min_crop_t"], 1,
        )
        outer_y0, outer_y1 = _expand_align_bounds(
            core_y0, core_y1, h,
            self.config["margin_h"], self.config["min_crop_h"], self.config["align_h"],
        )
        outer_x0, outer_x1 = _expand_align_bounds(
            core_x0, core_x1, w,
            self.config["margin_w"], self.config["min_crop_w"], self.config["align_w"],
        )

        roi: Dict[str, Any] = {
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

    def _smooth_rois(
        self,
        rois: List[Dict[str, Any]],
        t_len: int,
        h: int,
        w: int,
    ) -> List[Dict[str, Any]]:
        if not self.prev_rois or not rois:
            return self._repair_all_rois(rois, t_len=t_len, h=h, w=w)

        momentum = float(self.config.get("smooth_momentum", 0.6))
        iou_thresh = float(self.config.get("smooth_iou_thresh", 0.25))

        smoothed_rois: List[Dict[str, Any]] = []
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

                # 这里只是先保留平滑后的 outer 草案，
                # 真正合法 outer 后面统一通过 repair 重新基于 core 生成
                out = self._repair_roi_geometry(out, t_len=t_len, h=h, w=w)
                smoothed_rois.append(out)
            else:
                smoothed_rois.append(self._repair_roi_geometry(roi, t_len=t_len, h=h, w=w))

        return smoothed_rois

    def build_rois(self, latents: torch.Tensor, step_idx: int) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        bsz, _, t_len, h, w = latents.shape
        assert bsz == 1, "当前版本先只支持 batch_size=1。"

        if self.score_ema is None:
            full_roi = self._make_roi(t_len, h, w, 0, t_len, 0, h, 0, w)
            debug = {
                "step_idx": step_idx,
                "router_warm_start": True,
                "segments": [(0, t_len)],
                "rois": [full_roi],
            }
            self.prev_rois = [full_roi]
            return [full_roi], debug

        cue_maps = self._compose_score_maps()
        combined = cue_maps["combined"]
        assert combined is not None
        full_area = max(1, h * w)

        temporal_pool = str(self.config.get("temporal_pool", "topq")).lower()
        if temporal_pool == "mean":
            temporal_score = combined.mean(dim=(2, 3))
        else:
            temporal_score = _temporal_topq_mean(
                combined,
                float(self.config.get("temporal_topq_ratio", 0.10)),
            )
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

        candidates: List[Dict[str, Any]] = []
        spatial_debug: List[Dict[str, Any]] = []
        debug_tensors: Dict[str, Any] = {}

        use_tube = bool(self.config.get("use_tube_spatial", True))
        tube_iou = float(self.config.get("tube_link_iou_thresh", 0.15))
        max_dbg_frames = int(self.config.get("tube_debug_max_frames", 8))

        for seg_rank, (seg_score, s, e) in enumerate(segment_scores):
            spatial_score_raw = combined[0, s:e].mean(dim=0)
            spatial_score_blur = _smooth2d(
                spatial_score_raw,
                int(self.config["spatial_blur_kernel"]),
            )
            smin = float(spatial_score_blur.min().item())
            smax = float(spatial_score_blur.max().item())
            spatial_score_norm = (spatial_score_blur - smin) / max(1e-6, (smax - smin))

            seg_entry: Dict[str, Any] = {
                "segment": [s, e],
                "seg_score": seg_score,
                "temporal_pool": temporal_pool,
            }

            if use_tube and (e - s) >= 1:
                seg_entry["spatial_routing"] = "tube"
                records: List[Tuple[int, Optional[Tuple[int, int, int, int]], float, List[Dict[str, Any]]]] = []
                for t in range(s, e):
                    bb, sc_sum, finfos = self._frame_best_component_bbox(combined[0, t], full_area)
                    records.append((t, bb, sc_sum, finfos))

                chains: List[List[Tuple[int, Optional[Tuple[int, int, int, int]], float, List[Dict[str, Any]]]]] = []
                cur_chain: List[Tuple[int, Optional[Tuple[int, int, int, int]], float, List[Dict[str, Any]]]] = []
                prev_bb: Optional[Tuple[int, int, int, int]] = None
                for t, bb, sc_sum, finfos in records:
                    if bb is None:
                        if cur_chain:
                            chains.append(cur_chain)
                            cur_chain = []
                        prev_bb = None
                        continue
                    if prev_bb is None:
                        cur_chain = [(t, bb, sc_sum, finfos)]
                    else:
                        if _bbox_iou_2d(prev_bb, bb) >= tube_iou:
                            cur_chain.append((t, bb, sc_sum, finfos))
                        else:
                            if cur_chain:
                                chains.append(cur_chain)
                            cur_chain = [(t, bb, sc_sum, finfos)]
                    prev_bb = bb
                if cur_chain:
                    chains.append(cur_chain)

                min_ratio = float(self.config["spatial_min_bbox_ratio"])
                max_ratio = float(self.config["spatial_max_bbox_ratio"])
                tube_summaries: List[Dict[str, Any]] = []
                valid_tube_rois = 0

                for ci, chain in enumerate(chains):
                    bbs = [c[1] for c in chain if c[1] is not None]
                    if not bbs:
                        continue
                    u = _union_bboxes(bbs)
                    u_ratio = _bbox_area(u) / full_area
                    comp_score = float(sum(c[2] for c in chain))
                    tube_summaries.append({
                        "chain_id": ci,
                        "frames": [int(c[0]) for c in chain],
                        "union_bbox": list(u),
                        "union_area_ratio": float(u_ratio),
                        "comp_score_sum": comp_score,
                        "valid_union": bool(min_ratio <= u_ratio <= max_ratio),
                    })
                    if min_ratio <= u_ratio <= max_ratio:
                        valid_tube_rois += 1
                        y0, y1, x0, x1 = u
                        roi = self._make_roi(t_len, h, w, s, e, y0, y1, x0, x1)
                        roi["seg_rank"] = seg_rank
                        roi["seg_score"] = seg_score
                        roi["comp_score"] = comp_score
                        roi["bbox_source"] = "tube_union"
                        roi["tube_chain_id"] = ci
                        candidates.append(roi)

                per_frame_dbg: List[Dict[str, Any]] = []
                for t, bb, sc_sum, finfos in records[:max_dbg_frames]:
                    per_frame_dbg.append({
                        "t": int(t),
                        "bbox": None if bb is None else list(bb),
                        "best_score_sum": float(sc_sum),
                        "num_components": len(finfos),
                    })

                seg_entry.update({
                    "num_tube_chains": len(chains),
                    "tube_chain_summaries": tube_summaries,
                    "per_frame_top_bbox_preview": per_frame_dbg,
                    "num_tube_rois_valid": valid_tube_rois,
                })

                if valid_tube_rois == 0:
                    proj_bbox = _bbox_from_projection_mass(
                        spatial_score_norm,
                        keep_ratio_h=float(self.config["projection_keep_ratio_h"]),
                        keep_ratio_w=float(self.config["projection_keep_ratio_w"]),
                        blur_kernel=int(self.config["projection_blur_kernel"]),
                    )
                    y0, y1, x0, x1 = proj_bbox
                    roi = self._make_roi(t_len, h, w, s, e, y0, y1, x0, x1)
                    roi["seg_rank"] = seg_rank
                    roi["seg_score"] = seg_score
                    roi["comp_score"] = float(spatial_score_norm[y0:y1, x0:x1].mean().item())
                    roi["bbox_source"] = "tube_fallback_projection"
                    candidates.append(roi)
                    seg_entry["tube_fallback"] = "projection"

                flat_comps: List[Dict[str, Any]] = []
                for t, bb, sc_sum, finfos in records:
                    for fi in finfos:
                        fc = dict(fi)
                        fc["frame_t"] = int(t)
                        flat_comps.append(fc)
                seg_entry["num_components_total"] = len(flat_comps)
                seg_entry["num_components_valid"] = sum(1 for x in flat_comps if x.get("valid"))
                seg_entry["proj_bbox"] = list(
                    _bbox_from_projection_mass(
                        spatial_score_norm,
                        keep_ratio_h=float(self.config["projection_keep_ratio_h"]),
                        keep_ratio_w=float(self.config["projection_keep_ratio_w"]),
                        blur_kernel=int(self.config["projection_blur_kernel"]),
                    )
                )
                seg_entry["proj_area_ratio"] = float(_bbox_area(tuple(seg_entry["proj_bbox"])) / full_area)
                seg_entry["components"] = flat_comps[:48]

                spatial_debug.append(seg_entry)
                debug_tensors[f"seg{seg_rank}_segment_combined_stack"] = combined[0, s:e].detach().cpu()
            else:
                seg_entry["spatial_routing"] = "segment_mean"
                seed_topk = _topk_binary_mask(
                    spatial_score_norm.unsqueeze(0),
                    ratio=float(self.config["spatial_top_ratio"]),
                )[0]
                peak_ratio = float(self.config["spatial_peak_ratio"])
                peak_gate = spatial_score_norm >= (peak_ratio * float(spatial_score_norm.max().item()))
                seed_mask = seed_topk & peak_gate
                if seed_mask.sum().item() == 0:
                    seed_mask = seed_topk
                dilate_hw = int(self.config["spatial_dilate"])
                if dilate_hw > 0:
                    seed_mask = F.max_pool2d(
                        seed_mask.float().unsqueeze(0).unsqueeze(0),
                        kernel_size=2 * dilate_hw + 1,
                        stride=1,
                        padding=dilate_hw,
                    ).squeeze(0).squeeze(0) > 0

                comps = _extract_connected_components(
                    seed_mask,
                    min_area=int(self.config["spatial_cc_min_area"]),
                )

                comp_infos = []
                valid_comp_count = 0
                min_ratio = float(self.config["spatial_min_bbox_ratio"])
                max_ratio = float(self.config["spatial_max_bbox_ratio"])

                for comp_id, comp in enumerate(comps):
                    bbox = comp["bbox"]
                    if bbox is None:
                        continue
                    area_ratio = _bbox_area(bbox) / full_area
                    mask = comp["mask"]
                    score_mean = float(spatial_score_norm[mask].mean().item()) if mask.any() else 0.0
                    score_sum = float(spatial_score_norm[mask].sum().item()) if mask.any() else 0.0
                    comp_infos.append({
                        "comp_id": comp_id,
                        "bbox": list(bbox),
                        "pixels": int(comp["pixels"]),
                        "bbox_area_ratio": float(area_ratio),
                        "score_mean": score_mean,
                        "score_sum": score_sum,
                        "valid": bool(min_ratio <= area_ratio <= max_ratio),
                    })
                    if min_ratio <= area_ratio <= max_ratio:
                        valid_comp_count += 1
                        y0, y1, x0, x1 = bbox
                        roi = self._make_roi(t_len, h, w, s, e, y0, y1, x0, x1)
                        roi["seg_rank"] = seg_rank
                        roi["seg_score"] = seg_score
                        roi["comp_score"] = score_sum
                        roi["bbox_source"] = "multi_cc"
                        candidates.append(roi)

                proj_bbox = _bbox_from_projection_mass(
                    spatial_score_norm,
                    keep_ratio_h=float(self.config["projection_keep_ratio_h"]),
                    keep_ratio_w=float(self.config["projection_keep_ratio_w"]),
                    blur_kernel=int(self.config["projection_blur_kernel"]),
                )
                proj_area_ratio = _bbox_area(proj_bbox) / full_area

                if valid_comp_count == 0:
                    y0, y1, x0, x1 = proj_bbox
                    roi = self._make_roi(t_len, h, w, s, e, y0, y1, x0, x1)
                    roi["seg_rank"] = seg_rank
                    roi["seg_score"] = seg_score
                    roi["comp_score"] = float(spatial_score_norm[y0:y1, x0:x1].mean().item())
                    roi["bbox_source"] = "projection_fallback_no_valid_cc"
                    candidates.append(roi)

                seg_entry.update({
                    "num_components_total": len(comps),
                    "num_components_valid": valid_comp_count,
                    "proj_bbox": list(proj_bbox),
                    "proj_area_ratio": float(proj_area_ratio),
                    "components": comp_infos,
                })
                spatial_debug.append(seg_entry)
                debug_tensors[f"seg{seg_rank}_seed_mask"] = seed_mask.detach().cpu()

            debug_tensors[f"seg{seg_rank}_spatial_score_raw"] = spatial_score_raw.detach().cpu()
            debug_tensors[f"seg{seg_rank}_spatial_score_blur"] = spatial_score_blur.detach().cpu()
            debug_tensors[f"seg{seg_rank}_spatial_score_norm"] = spatial_score_norm.detach().cpu()

        if len(candidates) == 0:
            full_roi = self._make_roi(t_len, h, w, 0, t_len, 0, h, 0, w)
            candidates = [full_roi]

        candidates = sorted(
            candidates,
            key=lambda r: (float(r.get("comp_score", 0.0)) + 0.5 * float(r.get("seg_score", 0.0))),
            reverse=True
        )

        kept: List[Dict[str, Any]] = []
        nms_thresh = float(self.config.get("roi_nms_iou_thresh", 0.12))
        max_total = int(self.config.get("max_total_rois", 4))

        for cand in candidates:
            if len(kept) >= max_total:
                break

            overlap = False
            for old in kept:
                if _roi_iou(cand, old) > nms_thresh:
                    overlap = True
                    break

            if not overlap:
                kept.append(cand)

        max_per_seg = int(self.config.get("max_rois_per_segment", 2))
        seg_counter: Dict[int, int] = {}
        rois: List[Dict[str, Any]] = []
        for roi in kept:
            seg_rank = int(roi.get("seg_rank", -1))
            seg_counter.setdefault(seg_rank, 0)
            if seg_rank >= 0 and seg_counter[seg_rank] >= max_per_seg:
                continue
            seg_counter[seg_rank] += 1
            rois.append(roi)

        rois = sorted(rois, key=lambda r: (r["core_t0"], r["core_y0"], r["core_x0"]))
        rois = self._smooth_rois(rois, t_len=t_len, h=h, w=w)
        self.prev_rois = [r.copy() for r in rois]

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

        active_cues = self._active_cue_names()
        debug = {
            "step_idx": step_idx,
            "router_warm_start": False,
            "temporal_pool": temporal_pool,
            "temporal_topq_ratio": float(self.config.get("temporal_topq_ratio", 0.10)),
            "use_tube_spatial": bool(self.config.get("use_tube_spatial", True)),
            "tube_link_iou_thresh": float(self.config.get("tube_link_iou_thresh", 0.15)),
            "score_mean": float(combined.mean().item()),
            "score_std": float(combined.std().item()),
            "score_max": float(combined.max().item()),
            "cue_weights": {
                "step_diff": float(self.config.get("step_diff_weight", 1.0)),
                "cfg_gap": float(self.config.get("cfg_gap_weight", 0.8)),
                "warp": float(self.config.get("warp_weight", 0.8)),
                "traj_curv": float(self.config.get("traj_curv_weight", 0.6)),
            },
            "cue_means": {
                name: (None if cue_maps.get(name) is None else float(cue_maps[name].mean().item()))
                for name in active_cues
            },
            "ls_gap_last_step": int(self.last_ls_gap_step),
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

            save_payload: Dict[str, Any] = {
                "step_diff_ema": None if self.score_ema is None else self.score_ema.detach().cpu(),
                "traj_curv_ema": None if self.traj_curv_ema is None else self.traj_curv_ema.detach().cpu(),
                "cfg_gap_ema": None if self.cfg_gap_ema is None else self.cfg_gap_ema.detach().cpu(),
                "warp_ema": None if self.warp_ema is None else self.warp_ema.detach().cpu(),
                "combined_score": combined.detach().cpu(),
                "temporal_score": temporal_score.detach().cpu(),
                "temporal_mask": temporal_mask.detach().cpu(),
                "debug": debug,
            }

            for k, v in cue_maps.items():
                if v is not None:
                    save_payload[k] = v.detach().cpu()

            save_payload.update(debug_tensors)

            torch.save(save_payload, os.path.join(save_dir, f"router_step_{step_idx:03d}.pt"))

            with open(os.path.join(save_dir, f"router_step_{step_idx:03d}.json"), "w", encoding="utf-8") as f:
                json.dump(debug, f, ensure_ascii=False, indent=2, default=str)

        return rois, debug
