#!/usr/bin/env python3
import argparse
from pathlib import Path

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


def _frames_to_show(payload, debug, max_frames=4):
    temporal_top = debug.get("temporal_top_frames", [])
    rois = debug.get("rois", [])
    frames = []
    for f in temporal_top:
        if f not in frames:
            frames.append(int(f))
        if len(frames) >= max_frames:
            return frames
    for roi in rois:
        for f in [int(roi["core_t0"]), int(max(roi["core_t0"], roi["core_t1"] - 1))]:
            if f not in frames:
                frames.append(f)
            if len(frames) >= max_frames:
                return frames
    combined = payload["combined_score"]
    t_len = int(combined.shape[1])
    while len(frames) < min(max_frames, t_len):
        frames.append(len(frames))
    return frames[:max_frames]


def _draw_frame_panel(ax, score_2d, rois, frame_idx, title):
    ax.imshow(score_2d, cmap="viridis", vmin=0.0, vmax=1.0)
    for roi_idx, roi in enumerate(rois):
        if not (roi["core_t0"] <= frame_idx < roi["core_t1"]):
            continue

        rect_outer = patches.Rectangle(
            (roi["x0"], roi["y0"]),
            roi["x1"] - roi["x0"],
            roi["y1"] - roi["y0"],
            linewidth=2.0,
            edgecolor="cyan",
            facecolor="none",
            linestyle="--",
        )
        rect_core = patches.Rectangle(
            (roi["core_x0"], roi["core_y0"]),
            roi["core_x1"] - roi["core_x0"],
            roi["core_y1"] - roi["core_y0"],
            linewidth=2.0,
            edgecolor="red",
            facecolor="none",
        )
        ax.add_patch(rect_outer)
        ax.add_patch(rect_core)

        ax.text(
            roi["core_x0"],
            max(0, roi["core_y0"] - 1),
            f"R{roi_idx}",
            color="white",
            fontsize=8,
            bbox=dict(facecolor="black", alpha=0.5, pad=1),
        )

    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])


def render_step(step_file: Path, out_dir: Path, max_frames: int = 4):
    payload = _load_pt(step_file)
    debug = payload["debug"]
    step_idx = int(debug["step_idx"])
    rois = debug.get("rois", [])

    combined = payload["combined_score"][0]
    temporal_score = payload["temporal_score"][0]
    temporal_mask = payload["temporal_mask"][0].float()

    cue_names = ["step_diff", "cfg_gap", "motion", "ls_gap"]
    cue_maps = {name: payload.get(name, None) for name in cue_names}

    frames = _frames_to_show(payload, debug, max_frames=max_frames)

    ncols = max(2, len(frames))
    fig = plt.figure(figsize=(4.8 * ncols, 10.5))
    gs = fig.add_gridspec(3, ncols, height_ratios=[1.0, 1.0, 1.15])

    ax0 = fig.add_subplot(gs[0, :])
    xs = list(range(len(temporal_score)))
    ax0.plot(xs, temporal_score.numpy(), marker="o", linewidth=1.8)

    active = temporal_mask.numpy() > 0.5
    for idx, flag in enumerate(active):
        if flag:
            ax0.axvspan(idx - 0.5, idx + 0.5, alpha=0.15)

    for roi_idx, roi in enumerate(rois):
        ax0.axvspan(roi["core_t0"] - 0.5, roi["core_t1"] - 0.5, alpha=0.20)
        ax0.text(
            (roi["core_t0"] + roi["core_t1"] - 1) / 2.0,
            float(temporal_score.max()) * 1.02,
            f"R{roi_idx}",
            ha="center",
            fontsize=9,
        )

    ax0.set_title(
        f"step={step_idx} | temporal hard frames / segments | "
        f"core_ratio={debug.get('core_ratio', 0.0):.4f} | "
        f"outer_ratio={debug.get('outer_ratio', 0.0):.4f}"
    )
    ax0.set_xlabel("frame index (latent time)")
    ax0.set_ylabel("temporal score")
    ax0.grid(alpha=0.25)

    for j, frame_idx in enumerate(frames):
        ax = fig.add_subplot(gs[1, j])
        _draw_frame_panel(
            ax,
            combined[frame_idx].numpy(),
            rois,
            frame_idx,
            f"combined score @ frame {frame_idx}",
        )

    for j in range(len(frames), ncols):
        fig.add_subplot(gs[1, j]).axis("off")

    show_cues = []
    for name in cue_names:
        tensor = cue_maps.get(name)
        if tensor is not None:
            show_cues.append((name, tensor[0].mean(dim=0).numpy()))
    show_cues = show_cues[:ncols]

    for j, (name, arr) in enumerate(show_cues):
        ax = fig.add_subplot(gs[2, j])
        ax.imshow(arr, cmap="viridis")
        ax.set_title(f"{name} mean over time")
        ax.set_xticks([])
        ax.set_yticks([])

    for j in range(len(show_cues), ncols):
        fig.add_subplot(gs[2, j]).axis("off")

    cue_means = debug.get("cue_means", {})
    fig.suptitle(f"router debug: step {step_idx} | cue_means={cue_means}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    out_path = out_dir / f"router_step_{step_idx:03d}.png"
    fig.savefig(out_path, dpi=180)
    plt.close(fig)
    print(f"[render] saved: {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug_dir", type=str, required=True)
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--max_frames", type=int, default=4)
    args = parser.parse_args()

    debug_dir = Path(args.debug_dir)
    out_dir = Path(args.out_dir) if args.out_dir is not None else debug_dir / "vis"
    out_dir.mkdir(parents=True, exist_ok=True)

    step_files = _iter_step_files(debug_dir)
    if not step_files:
        raise SystemExit(f"No router_step_*.pt found in {debug_dir}")

    for step_file in step_files:
        render_step(step_file, out_dir=out_dir, max_frames=args.max_frames)
