#!/usr/bin/env python3
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from compression.hybrid_sd.routers.video_mask_router import VideoMaskRouter


def _bbox_iou(a, b):
    ay0, ay1, ax0, ax1 = a
    by0, by1, bx0, bx1 = b
    iy0, iy1 = max(ay0, by0), min(ay1, by1)
    ix0, ix1 = max(ax0, bx0), min(ax1, bx1)
    inter = max(0, iy1 - iy0) * max(0, ix1 - ix0)
    area_a = max(0, ay1 - ay0) * max(0, ax1 - ax0)
    area_b = max(0, by1 - by0) * max(0, bx1 - bx0)
    return inter / max(1, area_a + area_b - inter)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="router_debug_smoke")
    parser.add_argument("--min-iou", type=float, default=0.45)
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if out_dir.exists() and not args.keep:
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Synthetic moving subject in latent space. The true subject box is intentionally
    # compact and off-center so a full-frame or background-biased router fails the IoU gate.
    latents = torch.zeros(1, 4, 8, 16, 16)
    latents[:, :, 3:5, 5:11, 6:12] = 2.0
    latents[:, :, 4, 7:10, 8:13] += 3.0
    target_bbox = (4, 12, 5, 13)  # y0, y1, x0, x1 after smoothing/dilation tolerance

    cfg_gap = torch.zeros(1, 8, 16, 16)
    cfg_gap[:, 3:5, 5:11, 6:12] = 5.0

    router = VideoMaskRouter({
        "spatial_cue": "cfg",
        "temporal_top_ratio": 0.25,
        "spatial_top_ratio": 0.08,
        "max_cubes": 2,
        "save_debug_dir": str(out_dir),
        "debug_save_all_cues": True,
    })
    router.observe_aux(latents, cfg_gap_map=cfg_gap, step_idx=34)
    rois, debug = router.build_rois(latents, step_idx=34)

    candidates = debug.get("candidate_spatial_debug", {})
    if not rois:
        raise SystemExit("router produced no selected ROIs")
    missing = {"cfg", "motion", "warp"} - set(candidates)
    if missing:
        raise SystemExit(f"missing candidate cues: {sorted(missing)}")

    cue_ious = {}
    for cue, entries in candidates.items():
        if not entries:
            cue_ious[cue] = 0.0
            continue
        cue_ious[cue] = max(_bbox_iou(entry["spatial_bbox"], target_bbox) for entry in entries)

    best_cue = max(cue_ious, key=cue_ious.get)
    best_iou = cue_ious[best_cue]
    print(f"selected_rois={rois}")
    print(f"candidate_ious={cue_ious}")
    print(f"best_candidate={best_cue} iou={best_iou:.3f}")

    if best_iou < args.min_iou:
        raise SystemExit(f"best candidate IoU {best_iou:.3f} < required {args.min_iou:.3f}")

    render_script = ROOT / "scripts" / "hybrid_sd" / "render_hybrid_router_debug.py"
    subprocess.run(
        [
            sys.executable,
            str(render_script),
            "--debug_dir",
            str(out_dir),
            "--out_dir",
            str(out_dir / "vis"),
            "--max_frames",
            "3",
        ],
        check=True,
    )
    print(f"smoke debug visualization: {out_dir / 'vis' / 'router_step_034.png'}")


if __name__ == "__main__":
    main()
