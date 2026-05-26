#!/usr/bin/env python3
"""Observation 2(b): spatial sparsity inside a hard temporal segment on Wan2.1.

Question answered:
    Even when a temporal segment is hard, is the whole frame equally hard?

Figure:
    x-axis: Latent Width
    y-axis: Latent Height
    color: Spatial step_diff / large-model update magnitude

Preferred usage:
    First run obs2a_temporal_update_curve.py with --save_middle_maps, then pass
    its obs2a_temporal_update_stats.pt to this script. This avoids recomputing
    large trajectories and uses the same objectively selected representative
    prompt and hard temporal segment.
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from observation_diagnostics import (
    array_stats,
    ensure_dir as ensure_diag_dir,
    plot_topk_mass_curve,
    save_json,
    write_markdown,
)

from wan_observation_common import (
    cfg_noise_prediction,
    cleanup_pipeline,
    compute_seq_len,
    compute_target_shape,
    create_t2v_pipeline,
    encode_prompt,
    ensure_dir,
    initial_latent,
    load_prompt_items,
    make_scheduler,
    normalized_update_map,
    parse_size,
    scheduler_step,
    segment_vector,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stats_file", default=None, help="obs2a_temporal_update_stats.pt from obs2a script. Recommended.")
    parser.add_argument("--large_ckpt_dir", default=None, help="Only needed if --stats_file is not provided.")
    parser.add_argument("--large_task", default="t2v-14B")
    parser.add_argument("--prompt_file", default=None, help="Only used when recomputing without --stats_file.")
    parser.add_argument("--output_dir", default="analysis_outputs/obs2b_spatial_heatmap")
    parser.add_argument("--size", default="832*480")
    parser.add_argument("--frame_num", type=int, default=81)
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--sample_shift", type=float, default=5.0)
    parser.add_argument("--sample_solver", default="unipc", choices=["unipc", "dpm++"])
    parser.add_argument("--guide_scale", type=float, default=5.0)
    parser.add_argument("--middle_step", type=int, default=25)
    parser.add_argument("--segment_len", type=int, default=1)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--base_seed", type=int, default=0)
    parser.add_argument("--t5_cpu", action="store_true")
    parser.add_argument("--prompt_index", type=int, default=-1, help="Force a prompt index when recomputing or using stats. Default uses representative prompt from stats.")
    parser.add_argument("--clip_percentile", type=float, default=99.0, help="Clip heatmap color scale to this percentile for readability.")
    parser.add_argument("--topk_percent", type=float, default=20.0, help="Report how much update mass is covered by top-k spatial positions.")
    parser.add_argument("--debug_topk_percents", default="5,10,20,30,40,50",
                        help="Reserved for future top-k diagnostics; common values are shown in the debug top-k curve.")
    return parser.parse_args()


def plot_heatmap(path_pdf: str, path_png: str, heatmap: np.ndarray, title: str, clip_percentile: float):
    vmax = float(np.percentile(heatmap, clip_percentile)) if heatmap.size > 0 else 1.0
    vmax = max(vmax, 1e-8)
    plt.figure(figsize=(5.2, 4.2))
    im = plt.imshow(heatmap, origin="upper", aspect="auto", vmin=0, vmax=vmax)
    plt.xlabel("Latent Width")
    plt.ylabel("Latent Height")
    plt.title(title, fontsize=10)
    cbar = plt.colorbar(im)
    cbar.set_label("Spatial Update Magnitude")
    plt.tight_layout()
    plt.savefig(path_pdf, bbox_inches="tight")
    plt.savefig(path_png, dpi=300, bbox_inches="tight")
    plt.close()


def plot_topk_mask(path_pdf: str, path_png: str, heatmap: np.ndarray, topk_percent: float, title: str):
    flat = heatmap.reshape(-1)
    k = max(1, int(round(flat.size * topk_percent / 100.0)))
    threshold = np.partition(flat, -k)[-k]
    mask = (heatmap >= threshold).astype(float)
    plt.figure(figsize=(5.2, 4.2))
    im = plt.imshow(mask, origin="upper", aspect="auto", vmin=0, vmax=1)
    plt.xlabel("Latent Width")
    plt.ylabel("Latent Height")
    plt.title(title, fontsize=10)
    cbar = plt.colorbar(im)
    cbar.set_label(f"Top {topk_percent:g}% mask")
    plt.tight_layout()
    plt.savefig(path_pdf, bbox_inches="tight")
    plt.savefig(path_png, dpi=300, bbox_inches="tight")
    plt.close()


def save_heatmap_csv(path: str, heatmap: np.ndarray):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["latent_h", "latent_w", "spatial_update_magnitude"])
        h, w = heatmap.shape
        for i in range(h):
            for j in range(w):
                writer.writerow([i, j, float(heatmap[i, j])])


def topk_mass_ratio(heatmap: np.ndarray, topk_percent: float) -> float:
    flat = heatmap.reshape(-1).astype(np.float64)
    if flat.size == 0 or flat.sum() <= 0:
        return 0.0
    k = max(1, int(round(flat.size * topk_percent / 100.0)))
    idx = np.argpartition(flat, -k)[-k:]
    return float(flat[idx].sum() / flat.sum() * 100.0)


def from_stats_file(args):
    obj = torch.load(args.stats_file, map_location="cpu")
    middle_maps = obj.get("middle_maps", None)
    if not middle_maps:
        raise RuntimeError(
            "The stats file does not contain middle_maps. Re-run obs2a_temporal_update_curve.py with --save_middle_maps."
        )
    rep_idx = int(obj.get("representative_prompt_index", 0))
    prompt_index = args.prompt_index if args.prompt_index >= 0 else rep_idx
    item = None
    for m in middle_maps:
        if int(m["prompt_index"]) == prompt_index:
            item = m
            break
    if item is None:
        raise RuntimeError(f"prompt_index {prompt_index} not found in middle_maps.")

    update_map = item["update_map"].float()  # [F,H,W]
    segment_len = int(item.get("segment_len", obj.get("segment_len", args.segment_len)))
    temporal_curve = np.asarray(item["temporal_curve"], dtype=np.float64)
    hard_seg = int(np.argmax(temporal_curve))
    start = hard_seg * segment_len
    end = min((hard_seg + 1) * segment_len, update_map.shape[0])
    spatial = update_map[start:end].mean(dim=0).numpy()
    meta = {
        "prompt_index": prompt_index,
        "prompt": item["prompt"],
        "middle_step": int(item.get("middle_step", obj.get("middle_step", args.middle_step))),
        "hard_segment": hard_seg,
        "segment_start_latent_frame": start,
        "segment_end_latent_frame_exclusive": end,
    }
    return spatial, meta


def recompute(args):
    if args.large_ckpt_dir is None:
        raise ValueError("--large_ckpt_dir is required when --stats_file is not provided.")
    size = parse_size(args.size)
    prompt_items = load_prompt_items(args.prompt_file, base_seed=args.base_seed)
    pipe = create_t2v_pipeline(args.large_task, args.large_ckpt_dir, args.device_id, t5_cpu=args.t5_cpu)
    target_shape = compute_target_shape(pipe, size, args.frame_num)
    seq_len = compute_seq_len(pipe, target_shape)

    chosen_prompt_idx = args.prompt_index if args.prompt_index >= 0 else 0
    item = prompt_items[chosen_prompt_idx]
    context, context_null = encode_prompt(pipe, item.prompt)
    latents, seed_g = initial_latent(target_shape, pipe.device, item.seed)
    scheduler, timesteps = make_scheduler(args.sample_solver, pipe.num_train_timesteps, args.sample_steps, args.sample_shift, pipe.device)

    update_map_cpu = None
    for step_id, t in enumerate(tqdm(timesteps, desc="large trajectory"), start=1):
        latent_cur = latents[0]
        pred = cfg_noise_prediction(pipe, latent_cur, t, context, context_null, seq_len, args.guide_scale)
        latent_next = scheduler_step(scheduler, pred, t, latent_cur, seed_g)
        if step_id == args.middle_step:
            update_map_cpu = normalized_update_map(latent_next, latent_cur).detach().cpu().float()
            break
        latents = [latent_next]

    cleanup_pipeline(pipe)
    if update_map_cpu is None:
        raise RuntimeError(f"middle_step {args.middle_step} not reached.")

    frame_scores = update_map_cpu.mean(dim=(1, 2))
    seg_scores = segment_vector(frame_scores, args.segment_len).numpy()
    hard_seg = int(np.argmax(seg_scores))
    start = hard_seg * args.segment_len
    end = min((hard_seg + 1) * args.segment_len, update_map_cpu.shape[0])
    spatial = update_map_cpu[start:end].mean(dim=0).numpy()
    meta = {
        "prompt_index": chosen_prompt_idx,
        "prompt": item.prompt,
        "middle_step": args.middle_step,
        "hard_segment": hard_seg,
        "segment_start_latent_frame": start,
        "segment_end_latent_frame_exclusive": end,
    }
    return spatial, meta


def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    if args.stats_file:
        heatmap, meta = from_stats_file(args)
    else:
        heatmap, meta = recompute(args)

    ratio = topk_mass_ratio(heatmap, args.topk_percent)
    title = f"Prompt #{meta['prompt_index']}, step {meta['middle_step']}, hard segment {meta['hard_segment']}"
    plot_heatmap(
        os.path.join(args.output_dir, "fig_obs2b_spatial_update_heatmap.pdf"),
        os.path.join(args.output_dir, "fig_obs2b_spatial_update_heatmap.png"),
        heatmap,
        title,
        args.clip_percentile,
    )
    save_heatmap_csv(os.path.join(args.output_dir, "obs2b_spatial_heatmap.csv"), heatmap)
    np.save(os.path.join(args.output_dir, "obs2b_spatial_heatmap.npy"), heatmap)

    diag_dir = os.path.join(args.output_dir, "diagnostics")
    ensure_diag_dir(diag_dir)
    topk_curve_summary = plot_topk_mass_curve(
        os.path.join(diag_dir, "fig_debug_topk_spatial_mass_curve.pdf"),
        os.path.join(diag_dir, "fig_debug_topk_spatial_mass_curve.png"),
        heatmap,
        "Spatial concentration of update magnitude",
    )
    plot_topk_mask(
        os.path.join(diag_dir, f"fig_debug_top{args.topk_percent:g}_spatial_mask.pdf"),
        os.path.join(diag_dir, f"fig_debug_top{args.topk_percent:g}_spatial_mask.png"),
        heatmap,
        args.topk_percent,
        title=f"Top {args.topk_percent:g}% spatial update mask",
    )
    heatmap_stats = array_stats(heatmap.reshape(-1), thresholds=[])
    save_json(os.path.join(diag_dir, "summary.json"), {
        "description": "Observation 2b spatial heatmap diagnostics. A high top-k mass indicates spatial sparsity.",
        "meta": meta,
        "topk_percent_main": args.topk_percent,
        "topk_mass_main_percent": ratio,
        "topk_mass_curve": topk_curve_summary,
        "heatmap_stats": heatmap_stats,
    })

    with open(os.path.join(args.output_dir, "obs2b_meta.txt"), "w", encoding="utf-8") as f:
        for k, v in meta.items():
            f.write(f"{k}: {v}\n")
        f.write(f"top_{args.topk_percent:.1f}_percent_spatial_update_mass_percent: {ratio:.4f}\n")
        for k, v in topk_curve_summary.items():
            f.write(f"{k}: {v:.4f}\n")

    md_lines = [
        "This file is for debugging whether Observation 2(b) supports spatial sparsity inside a hard temporal segment.",
        "",
        "Expected useful pattern: a small percentage of spatial locations covers a large percentage of total update mass.",
        "",
        f"Prompt index: {meta.get('prompt_index')}",
        f"Middle step: {meta.get('middle_step')}",
        f"Hard segment: {meta.get('hard_segment')}",
        f"Top {args.topk_percent:g}% spatial positions cover {ratio:.2f}% of total update mass.",
        "",
        "## Additional top-k mass values",
    ]
    for k, v in topk_curve_summary.items():
        md_lines.append(f"- {k}: {v:.2f}%")
    write_markdown(os.path.join(diag_dir, "debug_report.md"), "Obs2b Spatial Heatmap Debug Report", md_lines)

    print(f"[Obs2b] Top {args.topk_percent:.1f}% spatial positions cover {ratio:.2f}% of total update mass.")
    print(f"[Obs2b] Done. Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
