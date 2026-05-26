#!/usr/bin/env python3
"""Observation 2(a): temporal non-uniformity of large-model updates on Wan2.1.

Question answered:
    Why introduce a hybrid stage? In the same denoising step, the large model
    changes different temporal segments by different amounts.

Figure:
    x-axis: Temporal Segment Index / latent frame segment index
    y-axis: Normalized Large-model Update Magnitude (step_diff)
    curves: selected denoising steps, e.g., step 10, 25, 40

The script chooses a representative prompt automatically: the prompt whose
middle-step temporal variation CV is closest to the median CV among prompts.
This avoids manually cherry-picking a visually favorable case.
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from observation_diagnostics import (
    curve_metrics,
    ensure_dir as ensure_diag_dir,
    group_numeric_rows,
    plot_metric_errorbar_by_step,
    plot_temporal_overlay,
    save_json,
    write_csv as write_rows_csv,
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
    parse_steps,
    scheduler_step,
    segment_vector,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--large_ckpt_dir", required=True, help="Path to Wan2.1-T2V-14B checkpoint dir.")
    parser.add_argument("--large_task", default="t2v-14B")
    parser.add_argument("--prompt_file", default=None, help="Each line: prompt, or seed<TAB>prompt.")
    parser.add_argument("--output_dir", default="analysis_outputs/obs2a_temporal_update")
    parser.add_argument("--size", default="832*480")
    parser.add_argument("--frame_num", type=int, default=81)
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--sample_shift", type=float, default=5.0)
    parser.add_argument("--sample_solver", default="unipc", choices=["unipc", "dpm++"])
    parser.add_argument("--guide_scale", type=float, default=5.0)
    parser.add_argument("--selected_steps", default="10,25,40", help="1-based denoising step ids.")
    parser.add_argument("--middle_step", type=int, default=25, help="Step used for representative-prompt selection.")
    parser.add_argument("--segment_len", type=int, default=1, help="Number of latent frames per temporal segment.")
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--base_seed", type=int, default=0)
    parser.add_argument("--t5_cpu", action="store_true")
    parser.add_argument("--plot_prompt_index", type=int, default=-1, help="If >=0, force plotting this prompt index instead of median-CV prompt.")
    parser.add_argument("--save_middle_maps", action="store_true", help="Also save middle-step update maps for spatial heatmap script.")
    parser.add_argument("--normalize_each_curve_by_mean", action="store_true",
                        help="If set, divide each temporal curve by its own mean for shape-only visualization. "
                             "Default keeps raw step_diff magnitude so early/middle/late levels remain comparable.")
    return parser.parse_args()


def save_temporal_csv(path: str, selected_steps: List[int], step_to_curve: Dict[int, np.ndarray]):
    max_len = max(len(v) for v in step_to_curve.values())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["temporal_segment_index"] + [f"step_{s}_normalized_update" for s in selected_steps])
        for i in range(max_len):
            row = [i]
            for s in selected_steps:
                curve = step_to_curve[s]
                row.append(float(curve[i]) if i < len(curve) else "")
            writer.writerow(row)


def plot_temporal(path_pdf: str, path_png: str, selected_steps: List[int], step_to_curve: Dict[int, np.ndarray], prompt_label: str):
    plt.figure(figsize=(6.4, 4.0))
    for s in selected_steps:
        y = step_to_curve[s]
        x = np.arange(len(y))
        plt.plot(x, y, linewidth=2, marker="o", markersize=3, label=f"Step {s}")
    plt.xlabel("Temporal Segment Index")
    plt.ylabel("Normalized Large-model Update Magnitude")
    plt.title(prompt_label, fontsize=10)
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(path_pdf, bbox_inches="tight")
    plt.savefig(path_png, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    selected_steps = parse_steps(args.selected_steps)
    if args.middle_step not in selected_steps:
        selected_steps.append(args.middle_step)
        selected_steps = sorted(set(selected_steps))
    selected_set = set(selected_steps)
    size = parse_size(args.size)
    prompt_items = load_prompt_items(args.prompt_file, base_seed=args.base_seed)

    pipe = create_t2v_pipeline(args.large_task, args.large_ckpt_dir, args.device_id, t5_cpu=args.t5_cpu)
    target_shape = compute_target_shape(pipe, size, args.frame_num)
    seq_len = compute_seq_len(pipe, target_shape)

    all_prompt_curves: List[Dict[int, np.ndarray]] = []
    all_prompt_raw_curves: List[Dict[int, np.ndarray]] = []
    all_prompt_cv = []
    middle_maps = []
    curve_metric_rows = []

    print("[Obs2a] Running large-only trajectories and collecting step_diff temporal curves.")
    for pidx, item in enumerate(tqdm(prompt_items, desc="large trajectories")):
        context, context_null = encode_prompt(pipe, item.prompt)
        latents, seed_g = initial_latent(target_shape, pipe.device, item.seed)
        scheduler, timesteps = make_scheduler(
            args.sample_solver, pipe.num_train_timesteps, args.sample_steps, args.sample_shift, pipe.device)

        step_to_curve = {}
        step_to_raw_curve = {}
        middle_update_map_cpu = None

        for step_id, t in enumerate(timesteps, start=1):
            latent_cur = latents[0]
            pred = cfg_noise_prediction(pipe, latent_cur, t, context, context_null, seq_len, args.guide_scale)
            latent_next = scheduler_step(scheduler, pred, t, latent_cur, seed_g)

            if step_id in selected_set:
                update_map = normalized_update_map(latent_next, latent_cur)  # [F,H,W]
                frame_scores = update_map.mean(dim=(1, 2))
                seg_scores_raw = segment_vector(frame_scores, args.segment_len)
                step_to_raw_curve[step_id] = seg_scores_raw.detach().cpu().numpy()
                if args.normalize_each_curve_by_mean:
                    seg_scores = seg_scores_raw / seg_scores_raw.mean().clamp_min(1e-6)
                else:
                    seg_scores = seg_scores_raw
                curve_np = seg_scores.detach().cpu().numpy()
                step_to_curve[step_id] = curve_np
                metric = curve_metrics(step_to_raw_curve[step_id])
                metric.update({
                    "prompt_index": pidx,
                    "seed": item.seed,
                    "step_id": step_id,
                    "prompt": item.prompt,
                })
                curve_metric_rows.append(metric)
                if step_id == args.middle_step and args.save_middle_maps:
                    middle_update_map_cpu = update_map.detach().cpu().to(torch.float16)

            latents = [latent_next]

        if args.middle_step not in step_to_curve:
            raise RuntimeError(f"middle_step {args.middle_step} was not collected.")
        mid = step_to_curve[args.middle_step]
        cv = float(np.std(mid) / (np.mean(mid) + 1e-8))
        all_prompt_cv.append(cv)
        all_prompt_curves.append(step_to_curve)
        all_prompt_raw_curves.append(step_to_raw_curve)
        if args.save_middle_maps:
            middle_maps.append({
                "prompt_index": pidx,
                "prompt": item.prompt,
                "seed": item.seed,
                "middle_step": args.middle_step,
                "segment_len": args.segment_len,
                "temporal_curve": mid,
                "temporal_curve_raw": step_to_raw_curve[args.middle_step],
                "update_map": middle_update_map_cpu,
                "cv": cv,
            })

        del context, context_null, latents, scheduler, timesteps
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    cleanup_pipeline(pipe)

    cvs = np.asarray(all_prompt_cv)
    if args.plot_prompt_index >= 0:
        rep_idx = args.plot_prompt_index
    else:
        median_cv = float(np.median(cvs))
        rep_idx = int(np.argmin(np.abs(cvs - median_cv)))

    rep_curves = all_prompt_curves[rep_idx]
    prompt_label = f"Representative prompt #{rep_idx}, CV={all_prompt_cv[rep_idx]:.3f}"
    plot_temporal(
        os.path.join(args.output_dir, "fig_obs2a_temporal_update_curve.pdf"),
        os.path.join(args.output_dir, "fig_obs2a_temporal_update_curve.png"),
        selected_steps,
        rep_curves,
        prompt_label,
    )
    save_temporal_csv(os.path.join(args.output_dir, "obs2a_representative_temporal_curve.csv"), selected_steps, rep_curves)

    with open(os.path.join(args.output_dir, "obs2a_prompt_cv.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt_index", "seed", "cv_middle_step", "prompt"])
        for i, (item, cv) in enumerate(zip(prompt_items, all_prompt_cv)):
            writer.writerow([i, item.seed, cv, item.prompt])

    diag_dir = os.path.join(args.output_dir, "diagnostics")
    ensure_diag_dir(diag_dir)
    write_rows_csv(os.path.join(diag_dir, "per_prompt_step_curve_metrics.csv"), curve_metric_rows)
    step_summary = group_numeric_rows(curve_metric_rows, "step_id", numeric_keys=["mean", "cv", "max", "peak_to_mean", "dynamic_range"])
    save_json(os.path.join(diag_dir, "summary.json"), {
        "description": "Observation 2a temporal step_diff diagnostics. Metrics are computed from raw temporal curves before optional display normalization.",
        "selected_steps": selected_steps,
        "middle_step": args.middle_step,
        "representative_prompt_index": rep_idx,
        "normalize_each_curve_by_mean_for_main_plot": bool(args.normalize_each_curve_by_mean),
        "step_summary": step_summary,
    })
    plot_metric_errorbar_by_step(
        os.path.join(diag_dir, "fig_debug_raw_mean_update_by_step.pdf"),
        os.path.join(diag_dir, "fig_debug_raw_mean_update_by_step.png"),
        curve_metric_rows, "mean", "Mean Temporal step_diff", "Across-prompt raw update magnitude",
    )
    plot_metric_errorbar_by_step(
        os.path.join(diag_dir, "fig_debug_temporal_cv_by_step.pdf"),
        os.path.join(diag_dir, "fig_debug_temporal_cv_by_step.png"),
        curve_metric_rows, "cv", "Temporal CV", "Across-prompt temporal non-uniformity",
    )
    for s in selected_steps:
        curves_s = [curves[s] for curves in all_prompt_raw_curves if s in curves]
        plot_temporal_overlay(
            os.path.join(diag_dir, f"fig_debug_all_prompts_step{s}_raw_curves.pdf"),
            os.path.join(diag_dir, f"fig_debug_all_prompts_step{s}_raw_curves.png"),
            curves_s, "Raw Temporal step_diff", f"All prompts, step {s}",
        )
    md_lines = [
        "This file is for debugging whether Observation 2(a) supports temporal non-uniformity and the hybrid-stage story.",
        "",
        "Expected useful pattern: early step has high raw update, middle step has larger temporal CV / peaks, late step has lower raw update.",
        "",
        f"Representative prompt index: {rep_idx}",
        "",
        "## Step-level summary",
    ]
    for s in selected_steps:
        ss = step_summary.get(str(s), {})
        md_lines.append(f"- Step {s}: raw_mean={ss.get('mean_mean', float('nan')):.4f}, temporal_CV={ss.get('cv_mean', float('nan')):.4f}, peak_to_mean={ss.get('peak_to_mean_mean', float('nan')):.4f}")
    write_markdown(os.path.join(diag_dir, "debug_report.md"), "Obs2a Temporal Update Debug Report", md_lines)

    torch.save({
        "selected_steps": selected_steps,
        "middle_step": args.middle_step,
        "segment_len": args.segment_len,
        "representative_prompt_index": rep_idx,
        "prompt_items": [item.__dict__ for item in prompt_items],
        "prompt_cv": all_prompt_cv,
        "curves": all_prompt_curves,
        "raw_curves": all_prompt_raw_curves,
        "curve_metrics": curve_metric_rows,
        "middle_maps": middle_maps if args.save_middle_maps else None,
    }, os.path.join(args.output_dir, "obs2a_temporal_update_stats.pt"))

    print(f"[Obs2a] Representative prompt index: {rep_idx}")
    print(f"[Obs2a] Done. Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
