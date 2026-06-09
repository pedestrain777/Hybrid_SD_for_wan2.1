#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch


def _load_pt(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _iter_step_files(debug_dir: Path):
    return sorted(debug_dir.glob("router_step_*.pt"))


def _normalize_2d(x: torch.Tensor) -> torch.Tensor:
    x = x.float()
    return (x - x.min()) / (x.max() - x.min()).clamp_min(1e-6)


def _score_volume(payload: dict, key: str) -> Optional[torch.Tensor]:
    tensor = payload.get(key)
    if tensor is None:
        return None
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    return tensor.float()


def _candidate_score(payload: dict, cue: str, seg_rank: int) -> Optional[torch.Tensor]:
    direct = payload.get(f"candidate_{cue}_seg{seg_rank}_spatial_score_norm")
    if direct is not None:
        return direct.float()

    key = {"cfg": "cfg_gap_ema", "motion": "motion_map", "warp": "warp_map"}.get(cue)
    vol = _score_volume(payload, key) if key is not None else None
    if vol is None:
        return None

    debug = payload["debug"]
    segments = debug.get("segments", [])
    if seg_rank >= len(segments):
        return _normalize_2d(vol[0].mean(dim=0))
    s, e = segments[seg_rank]
    return _normalize_2d(vol[0, int(s):int(e)].mean(dim=0))


def _frames_to_show(payload: dict, debug: dict, max_frames: int = 4) -> List[int]:
    frames: List[int] = []
    for f in debug.get("temporal_top_frames", []):
        if int(f) not in frames:
            frames.append(int(f))
        if len(frames) >= max_frames:
            return frames

    for roi in debug.get("rois", []):
        for f in [int(roi["core_t0"]), int(max(roi["core_t0"], roi["core_t1"] - 1))]:
            if f not in frames:
                frames.append(f)
            if len(frames) >= max_frames:
                return frames

    t_len = int(payload["temporal_score"].shape[1])
    while len(frames) < min(max_frames, t_len):
        frames.append(len(frames))
    return frames[:max_frames]


def _draw_roi(ax, roi: dict, label: str, edgecolor: str, scale_xy: Tuple[float, float] = (1.0, 1.0)):
    sx, sy = scale_xy
    rect = patches.Rectangle(
        (roi["core_x0"] * sx, roi["core_y0"] * sy),
        (roi["core_x1"] - roi["core_x0"]) * sx,
        (roi["core_y1"] - roi["core_y0"]) * sy,
        linewidth=2.2,
        edgecolor=edgecolor,
        facecolor="none",
    )
    ax.add_patch(rect)
    ax.text(
        roi["core_x0"] * sx,
        max(0, roi["core_y0"] * sy - 3),
        label,
        color="white",
        fontsize=8,
        bbox=dict(facecolor="black", alpha=0.55, pad=1),
    )


def _draw_selected_frame(ax, score_2d: torch.Tensor, rois: list, frame_idx: int, title: str):
    ax.imshow(_normalize_2d(score_2d), cmap="viridis", vmin=0.0, vmax=1.0)
    for roi_idx, roi in enumerate(rois):
        if not (roi["core_t0"] <= frame_idx < roi["core_t1"]):
            continue
        outer = patches.Rectangle(
            (roi["x0"], roi["y0"]),
            roi["x1"] - roi["x0"],
            roi["y1"] - roi["y0"],
            linewidth=1.6,
            edgecolor="cyan",
            facecolor="none",
            linestyle="--",
        )
        ax.add_patch(outer)
        _draw_roi(ax, roi, f"selected R{roi_idx}", "red")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def _load_video_frame(video_path: Optional[Path], latent_idx: int, latent_t: int):
    if video_path is None:
        return None, None
    try:
        import imageio.v3 as iio
    except Exception:
        return None, "imageio is not available"

    try:
        meta = iio.immeta(video_path)
        nframes = int(meta.get("nframes") or meta.get("duration", 0) * meta.get("fps", 0) or 0)
        if nframes <= 0:
            frames = iio.imread(video_path)
            nframes = len(frames)
            video_idx = int(round(latent_idx * (nframes - 1) / max(1, latent_t - 1)))
            return frames[video_idx], None
        video_idx = int(round(latent_idx * (nframes - 1) / max(1, latent_t - 1)))
        return iio.imread(video_path, index=video_idx), None
    except Exception as exc:
        return None, str(exc)


def _render_video_overlay(payload: dict, debug: dict, video_path: Optional[Path], out_dir: Path, step_idx: int):
    if video_path is None:
        return
    spatial = payload.get("spatial_source")
    if spatial is None:
        return
    latent_t = int(spatial.shape[1])
    frame_idx = _frames_to_show(payload, debug, max_frames=1)[0]
    frame, err = _load_video_frame(video_path, frame_idx, latent_t)
    if frame is None:
        print(f"[render] skip video overlay: {err}")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(frame)
    frame_h, frame_w = frame.shape[0], frame.shape[1]
    latent_h, latent_w = int(spatial.shape[-2]), int(spatial.shape[-1])
    scale_xy = (frame_w / max(1, latent_w), frame_h / max(1, latent_h))

    colors = {"cfg": "lime", "motion": "yellow", "warp": "magenta"}
    candidates = debug.get("candidate_spatial_debug", {})
    for cue, entries in candidates.items():
        if not entries:
            continue
        roi = entries[0].get("roi")
        if roi is not None and roi["core_t0"] <= frame_idx < roi["core_t1"]:
            _draw_roi(ax, roi, cue, colors.get(cue, "white"), scale_xy=scale_xy)

    for i, roi in enumerate(debug.get("rois", [])):
        if roi["core_t0"] <= frame_idx < roi["core_t1"]:
            _draw_roi(ax, roi, f"selected R{i}", "red", scale_xy=scale_xy)

    ax.set_title(f"video overlay | step={step_idx} | latent frame={frame_idx}")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    out_path = out_dir / f"router_step_{step_idx:03d}_video_overlay.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[render] saved: {out_path}")


def render_step(step_file: Path, out_dir: Path, max_frames: int = 4, video_path: Optional[Path] = None):
    payload = _load_pt(step_file)
    debug = payload["debug"]
    step_idx = int(debug["step_idx"])
    rois = debug.get("rois", [])

    spatial_source = payload["spatial_source"][0].float()
    temporal_score = payload["temporal_score"][0].float()
    temporal_mask = payload["temporal_mask"][0].float()
    frames = _frames_to_show(payload, debug, max_frames=max_frames)

    cue_order = [cue for cue in ("cfg", "motion", "warp") if cue in debug.get("candidate_spatial_debug", {})]
    ncols = max(len(frames), len(cue_order), 3)
    fig = plt.figure(figsize=(4.6 * ncols, 11.5))
    gs = fig.add_gridspec(3, ncols, height_ratios=[1.0, 1.2, 1.2])

    ax0 = fig.add_subplot(gs[0, :])
    xs = list(range(len(temporal_score)))
    ax0.plot(xs, temporal_score.numpy(), marker="o", linewidth=1.8)
    for idx, flag in enumerate(temporal_mask.numpy() > 0.5):
        if flag:
            ax0.axvspan(idx - 0.5, idx + 0.5, alpha=0.15)
    for roi_idx, roi in enumerate(rois):
        ax0.axvspan(roi["core_t0"] - 0.5, roi["core_t1"] - 0.5, alpha=0.20)
        ax0.text((roi["core_t0"] + roi["core_t1"] - 1) / 2.0, float(temporal_score.max()) * 1.02, f"R{roi_idx}", ha="center", fontsize=9)
    ax0.set_title(
        f"step={step_idx} | temporal=frame_diff | selected_spatial={debug.get('spatial_cue')} | "
        f"core_ratio={debug.get('core_ratio', 0.0):.4f} | outer_ratio={debug.get('outer_ratio', 0.0):.4f}"
    )
    ax0.set_xlabel("latent frame index")
    ax0.set_ylabel("temporal score")
    ax0.grid(alpha=0.25)

    for j, frame_idx in enumerate(frames):
        ax = fig.add_subplot(gs[1, j])
        _draw_selected_frame(
            ax,
            spatial_source[frame_idx],
            rois,
            frame_idx,
            f"selected cue map @ latent frame {frame_idx}",
        )
    for j in range(len(frames), ncols):
        fig.add_subplot(gs[1, j]).axis("off")

    candidates = debug.get("candidate_spatial_debug", {})
    for j, cue in enumerate(cue_order):
        ax = fig.add_subplot(gs[2, j])
        entries = candidates.get(cue, [])
        seg_rank = int(entries[0].get("seg_rank", 0)) if entries else 0
        score = _candidate_score(payload, cue, seg_rank)
        if score is None:
            ax.axis("off")
            continue
        ax.imshow(score.numpy(), cmap="viridis", vmin=0.0, vmax=1.0)
        for idx, entry in enumerate(entries[:2]):
            roi = entry.get("roi")
            if roi is not None:
                _draw_roi(ax, roi, f"{cue} R{idx}", "red" if cue == debug.get("spatial_cue") else "white")
        ax.set_title(f"{cue} candidate bbox")
        ax.set_xticks([])
        ax.set_yticks([])
    for j in range(len(cue_order), ncols):
        fig.add_subplot(gs[2, j]).axis("off")

    fig.tight_layout()
    out_path = out_dir / f"router_step_{step_idx:03d}.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[render] saved: {out_path}")

    _render_video_overlay(payload, debug, video_path, out_dir, step_idx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--max_frames", type=int, default=4)
    parser.add_argument("--video", type=str, default=None, help="Optional generated mp4 for scaled bbox overlay.")
    args = parser.parse_args()

    debug_dir = Path(args.debug_dir)
    out_dir = Path(args.out_dir) if args.out_dir is not None else debug_dir / "vis"
    out_dir.mkdir(parents=True, exist_ok=True)
    video_path = Path(args.video) if args.video else None

    step_files = _iter_step_files(debug_dir)
    if not step_files:
        raise SystemExit(f"No router_step_*.pt found in {debug_dir}")

    for step_file in step_files:
        render_step(step_file, out_dir=out_dir, max_frames=args.max_frames, video_path=video_path)
