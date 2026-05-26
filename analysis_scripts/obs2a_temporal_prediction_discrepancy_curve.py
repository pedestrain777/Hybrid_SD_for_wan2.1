#!/usr/bin/env python3
"""Observation 2(a-alt): temporal large-small prediction discrepancy on Wan2.1.

Question answered:
    Why introduce a hybrid stage from the temporal dimension?

Diagnostic idea:
    At selected denoising steps, we take the SAME latent x_t from a large-only
    reference trajectory and feed it to both the large and small denoisers. We
    then measure the large-small prediction discrepancy for each latent frame /
    temporal segment.

Expected figure:
    x-axis: Temporal Segment Index / latent frame segment index
    y-axis: Mean Large-Small Prediction Difference
    curves: selected denoising steps, e.g., step 10, 25, 40

Interpretation:
    - Step 10 high over most temporal segments: small model is not ready to
      replace the large model in the early global formation stage.
    - Step 25 mixed high/low: some temporal segments can be handled by the small
      model, while hard segments still need the large model, motivating a hybrid
      stage.
    - Step 40 low over most temporal segments: late denoising is more stable and
      the small model can replace the large model more broadly.

This is a controlled comparison. It does NOT compare two independently rolled
out large/small trajectories, which would include cumulative drift.
"""

from __future__ import annotations

import argparse
import csv
import gc
import os
import tempfile
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
    normalized_prediction_difference,
    parse_size,
    parse_steps,
    scheduler_step,
    segment_vector,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--large_ckpt_dir", required=True, help="Path to Wan2.1-T2V-14B checkpoint dir.")
    parser.add_argument("--small_ckpt_dir", required=True, help="Path to Wan2.1-T2V-1.3B checkpoint dir.")
    parser.add_argument("--large_task", default="t2v-14B")
    parser.add_argument("--small_task", default="t2v-1.3B")
    parser.add_argument("--prompt_file", default=None, help="Each line: prompt, or seed<TAB>prompt.")
    parser.add_argument("--output_dir", default="analysis_outputs/obs2a_temporal_prediction_discrepancy")
    parser.add_argument("--size", default="832*480", help="Wan size, e.g., 832*480. Use 480P for 1.3B compatibility.")
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
    parser.add_argument("--plot_prompt_index", type=int, default=-1,
                        help="If >=0, force plotting this prompt index instead of auto representative prompt.")
    parser.add_argument("--normalize_plot_by_prompt_max", action="store_true",
                        help="Normalize all plotted curves of the chosen prompt by their shared maximum. "
                             "Default keeps raw normalized discrepancy so step magnitudes remain comparable.")
    parser.add_argument("--keep_tmp", action="store_true", help="Keep temporary large trajectory entries for debugging.")
    return parser.parse_args()


def save_temporal_csv(path: str, selected_steps: List[int], step_to_curve: Dict[int, np.ndarray]):
    max_len = max(len(v) for v in step_to_curve.values())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["temporal_segment_index"] + [f"step_{s}_prediction_discrepancy" for s in selected_steps])
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
    plt.ylabel("Mean Large-Small Prediction Difference")
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

    tmp_dir_obj = tempfile.TemporaryDirectory(prefix="obs2a_lg_latents_", dir=args.output_dir)
    tmp_dir = tmp_dir_obj.name
    entries: List[dict] = []

    print("[Obs2a-alt] Pass 1/2: run large-only trajectories and save selected latents/predictions.")
    large_pipe = create_t2v_pipeline(args.large_task, args.large_ckpt_dir, args.device_id, t5_cpu=args.t5_cpu)
    target_shape = compute_target_shape(large_pipe, size, args.frame_num)
    seq_len = compute_seq_len(large_pipe, target_shape)

    for pidx, item in enumerate(tqdm(prompt_items, desc="large trajectories")):
        context, context_null = encode_prompt(large_pipe, item.prompt)
        latents, seed_g = initial_latent(target_shape, large_pipe.device, item.seed)
        scheduler, timesteps = make_scheduler(
            args.sample_solver, large_pipe.num_train_timesteps, args.sample_steps, args.sample_shift, large_pipe.device)

        for step_id, t in enumerate(timesteps, start=1):
            latent_cur = latents[0]
            pred_large = cfg_noise_prediction(
                large_pipe, latent_cur, t, context, context_null, seq_len, args.guide_scale)

            if step_id in selected_set:
                entry_path = os.path.join(tmp_dir, f"prompt{pidx:04d}_step{step_id:03d}.pt")
                torch.save({
                    "prompt_index": pidx,
                    "prompt": item.prompt,
                    "seed": item.seed,
                    "step_id": step_id,
                    "timestep": t.detach().cpu(),
                    "latent": latent_cur.detach().cpu().to(torch.float16),
                    "pred_large": pred_large.detach().cpu().to(torch.float16),
                }, entry_path)
                entries.append({"path": entry_path, "prompt_index": pidx, "step_id": step_id})

            latent_next = scheduler_step(scheduler, pred_large, t, latent_cur, seed_g)
            latents = [latent_next]

        del context, context_null, latents, scheduler, timesteps
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    cleanup_pipeline(large_pipe)

    print("[Obs2a-alt] Pass 2/2: load small model and compute temporal discrepancy curves.")
    small_pipe = create_t2v_pipeline(args.small_task, args.small_ckpt_dir, args.device_id, t5_cpu=args.t5_cpu)
    small_target_shape = compute_target_shape(small_pipe, size, args.frame_num)
    if small_target_shape != target_shape:
        raise RuntimeError(f"Large/small target shapes differ: {target_shape} vs {small_target_shape}")
    small_seq_len = compute_seq_len(small_pipe, small_target_shape)

    # prompt_index -> {step_id -> temporal curve}
    all_prompt_curves: List[Dict[int, np.ndarray]] = [dict() for _ in prompt_items]
    curve_metric_rows = []
    small_context_cache = {}

    for entry in tqdm(entries, desc="small predictions"):
        obj = torch.load(entry["path"], map_location="cpu")
        pidx = int(obj["prompt_index"])
        if pidx not in small_context_cache:
            small_context_cache[pidx] = encode_prompt(small_pipe, obj["prompt"])
        context, context_null = small_context_cache[pidx]

        latent = obj["latent"].to(small_pipe.device, dtype=torch.float32)
        pred_large = obj["pred_large"].to(small_pipe.device, dtype=torch.float32)
        timestep = obj["timestep"].to(small_pipe.device)
        pred_small = cfg_noise_prediction(
            small_pipe, latent, timestep, context, context_null, small_seq_len, args.guide_scale)

        # [F,H,W], then average spatial dimensions to get one value per latent frame.
        diff_map = normalized_prediction_difference(pred_large, pred_small)
        frame_scores = diff_map.mean(dim=(1, 2))
        seg_scores = segment_vector(frame_scores, args.segment_len)
        curve_np = seg_scores.detach().cpu().numpy()
        all_prompt_curves[pidx][int(obj["step_id"])] = curve_np
        metric = curve_metrics(curve_np)
        metric.update({
            "prompt_index": pidx,
            "seed": int(obj["seed"]),
            "step_id": int(obj["step_id"]),
            "prompt": obj["prompt"],
        })
        curve_metric_rows.append(metric)

        del obj, latent, pred_large, pred_small, diff_map, frame_scores, seg_scores
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    cleanup_pipeline(small_pipe)

    # Pick a representative prompt: middle-step temporal CV closest to median.
    # If the user provides --plot_prompt_index, use that prompt directly.
    prompt_cv = []
    valid_indices = []
    for pidx, curves in enumerate(all_prompt_curves):
        if args.middle_step not in curves:
            continue
        mid = curves[args.middle_step]
        cv = float(np.std(mid) / (np.mean(mid) + 1e-8))
        prompt_cv.append(cv)
        valid_indices.append(pidx)
    if not valid_indices:
        raise RuntimeError(f"No prompt has middle_step={args.middle_step} curve.")

    if args.plot_prompt_index >= 0:
        rep_idx = args.plot_prompt_index
        if rep_idx < 0 or rep_idx >= len(all_prompt_curves):
            raise ValueError(f"Invalid --plot_prompt_index {rep_idx}")
    else:
        cvs = np.asarray(prompt_cv)
        median_cv = float(np.median(cvs))
        rep_idx = int(valid_indices[int(np.argmin(np.abs(cvs - median_cv)))])

    rep_curves = {s: all_prompt_curves[rep_idx][s] for s in selected_steps if s in all_prompt_curves[rep_idx]}
    if args.normalize_plot_by_prompt_max:
        vmax = max(float(np.max(v)) for v in rep_curves.values())
        vmax = max(vmax, 1e-8)
        rep_curves = {s: v / vmax for s, v in rep_curves.items()}

    rep_cv = float(np.std(all_prompt_curves[rep_idx][args.middle_step]) /
                   (np.mean(all_prompt_curves[rep_idx][args.middle_step]) + 1e-8))
    prompt_label = f"Representative prompt #{rep_idx}, middle-step CV={rep_cv:.3f}"
    plot_temporal(
        os.path.join(args.output_dir, "fig_obs2a_temporal_prediction_discrepancy.pdf"),
        os.path.join(args.output_dir, "fig_obs2a_temporal_prediction_discrepancy.png"),
        list(rep_curves.keys()),
        rep_curves,
        prompt_label,
    )
    save_temporal_csv(os.path.join(args.output_dir, "obs2a_representative_temporal_prediction_discrepancy.csv"),
                      list(rep_curves.keys()), rep_curves)

    with open(os.path.join(args.output_dir, "obs2a_prompt_middle_cv.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["prompt_index", "seed", "cv_middle_step", "prompt"])
        cv_map = {idx: cv for idx, cv in zip(valid_indices, prompt_cv)}
        for i, item in enumerate(prompt_items):
            writer.writerow([i, item.seed, cv_map.get(i, ""), item.prompt])

    diag_dir = os.path.join(args.output_dir, "diagnostics")
    ensure_diag_dir(diag_dir)
    write_rows_csv(os.path.join(diag_dir, "per_prompt_step_curve_metrics.csv"), curve_metric_rows)
    step_summary = group_numeric_rows(curve_metric_rows, "step_id", numeric_keys=["mean", "cv", "max", "peak_to_mean", "dynamic_range"])
    save_json(os.path.join(diag_dir, "summary.json"), {
        "description": "Temporal large-small prediction-discrepancy diagnostics. High mean = small model deviates from large model more for that temporal segment/step.",
        "selected_steps": selected_steps,
        "middle_step": args.middle_step,
        "representative_prompt_index": rep_idx,
        "step_summary": step_summary,
    })
    plot_metric_errorbar_by_step(
        os.path.join(diag_dir, "fig_debug_mean_temporal_discrepancy_by_step.pdf"),
        os.path.join(diag_dir, "fig_debug_mean_temporal_discrepancy_by_step.png"),
        curve_metric_rows, "mean", "Mean Temporal Prediction Difference", "Across-prompt temporal discrepancy level",
    )
    plot_metric_errorbar_by_step(
        os.path.join(diag_dir, "fig_debug_temporal_cv_by_step.pdf"),
        os.path.join(diag_dir, "fig_debug_temporal_cv_by_step.png"),
        curve_metric_rows, "cv", "Temporal CV", "Across-prompt temporal non-uniformity",
    )
    for s in selected_steps:
        curves_s = [curves[s] for curves in all_prompt_curves if s in curves]
        plot_temporal_overlay(
            os.path.join(diag_dir, f"fig_debug_all_prompts_step{s}_temporal_pred_diff.pdf"),
            os.path.join(diag_dir, f"fig_debug_all_prompts_step{s}_temporal_pred_diff.png"),
            curves_s, "Mean Large-Small Prediction Difference", f"All prompts, step {s}",
        )
    md_lines = [
        "This file is for debugging whether the temporal prediction-discrepancy version supports the hybrid-stage story.",
        "",
        "Expected useful pattern: step 10 high over most segments, step 25 mixed/high-CV, step 40 low over most segments.",
        "",
        f"Representative prompt index: {rep_idx}",
        "",
        "## Step-level summary",
    ]
    for s in selected_steps:
        ss = step_summary.get(str(s), {})
        md_lines.append(f"- Step {s}: mean={ss.get('mean_mean', float('nan')):.4f}, temporal_CV={ss.get('cv_mean', float('nan')):.4f}, peak_to_mean={ss.get('peak_to_mean_mean', float('nan')):.4f}")
    write_markdown(os.path.join(diag_dir, "debug_report.md"), "Obs2a Temporal Prediction Discrepancy Debug Report", md_lines)

    torch.save({
        "selected_steps": selected_steps,
        "middle_step": args.middle_step,
        "segment_len": args.segment_len,
        "representative_prompt_index": rep_idx,
        "prompt_items": [item.__dict__ for item in prompt_items],
        "prompt_cv": {idx: cv for idx, cv in zip(valid_indices, prompt_cv)},
        "curves": all_prompt_curves,
        "curve_metrics": curve_metric_rows,
    }, os.path.join(args.output_dir, "obs2a_temporal_prediction_discrepancy_stats.pt"))

    if args.keep_tmp:
        print(f"[Obs2a-alt] Temporary entries kept at: {tmp_dir}")
    else:
        tmp_dir_obj.cleanup()

    print(f"[Obs2a-alt] Representative prompt index: {rep_idx}")
    print(f"[Obs2a-alt] Done. Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
