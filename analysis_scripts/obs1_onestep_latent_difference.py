#!/usr/bin/env python3
"""Observation 1 alternative: one-step latent difference between large/small denoisers.

Question answered:
    Why can/should we do large-small collaborative denoising?

Figure:
    x-axis: Normalized One-step Latent Difference
    y-axis: Token Percentage (%)
    curves: selected denoising steps, e.g., step 10, 25, 40

Diagnostic design:
    This is still a controlled comparison. We do NOT compare two independently
    rolled-out large/small trajectories. Instead, we run a large-only reference
    trajectory, save the same latent x_t at selected steps, and compare the
    one-step updated latents produced from that exact x_t:

        x_{t-1}^L = scheduler(x_t, pred_large)
        x_{t-1}^S = scheduler(x_t, pred_small)

    Then we plot the distribution of ||x_{t-1}^L - x_{t-1}^S|| over all
    spatiotemporal latent tokens.

Note:
    For multistep samplers such as UniPC/DPM++, the scheduler has internal
    history. This script saves a scheduler snapshot right before each selected
    step, then reuses that same scheduler state when stepping with the small
    model prediction. This keeps the comparison controlled and scheduler-aware.
"""

from __future__ import annotations

import argparse
import copy
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
    array_stats,
    ensure_dir as ensure_diag_dir,
    group_numeric_rows,
    parse_float_list,
    plot_metric_errorbar_by_step,
    plot_two_metric_by_step,
    save_json,
    weighted_hist_stats,
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
    parse_size,
    parse_steps,
    scheduler_step,
    token_l2,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--large_ckpt_dir", required=True, help="Path to Wan2.1-T2V-14B checkpoint dir.")
    parser.add_argument("--small_ckpt_dir", required=True, help="Path to Wan2.1-T2V-1.3B checkpoint dir.")
    parser.add_argument("--large_task", default="t2v-14B")
    parser.add_argument("--small_task", default="t2v-1.3B")
    parser.add_argument("--prompt_file", default=None, help="Each line: prompt, or seed<TAB>prompt.")
    parser.add_argument("--output_dir", default="analysis_outputs/obs1_onestep_latent_difference")
    parser.add_argument("--size", default="832*480", help="Wan size, e.g., 832*480. Use 480P for 1.3B compatibility.")
    parser.add_argument("--frame_num", type=int, default=81)
    parser.add_argument("--sample_steps", type=int, default=50)
    parser.add_argument("--sample_shift", type=float, default=5.0)
    parser.add_argument("--sample_solver", default="unipc", choices=["unipc", "dpm++"])
    parser.add_argument("--guide_scale", type=float, default=5.0)
    parser.add_argument("--selected_steps", default="10,25,40", help="1-based denoising step ids.")
    parser.add_argument("--bins", type=int, default=80)
    parser.add_argument("--max_diff", type=float, default=1.0)
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--base_seed", type=int, default=0)
    parser.add_argument("--t5_cpu", action="store_true")
    parser.add_argument("--keep_tmp", action="store_true", help="Keep temporary entries for debugging.")
    parser.add_argument("--diagnostic_thresholds", default="0.05,0.10,0.20,0.30",
                        help="Comma-separated thresholds used for debug token fractions.")
    return parser.parse_args()


def _torch_load(path: str):
    """Compatible torch.load wrapper across PyTorch versions."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _move_list_tensors(xs, device: torch.device):
    out = []
    for x in xs:
        if isinstance(x, torch.Tensor):
            out.append(x.to(device=device, dtype=torch.float32))
        else:
            out.append(x)
    return out


def move_scheduler_history_to_device(scheduler, device: torch.device):
    """Move only the scheduler history tensors needed by multistep stepping.

    Wan's UniPC scheduler intentionally keeps sigmas on CPU. We therefore avoid
    moving every tensor blindly and only move full latent-sized history tensors.
    """
    if hasattr(scheduler, "model_outputs") and scheduler.model_outputs is not None:
        scheduler.model_outputs = _move_list_tensors(scheduler.model_outputs, device)
    if hasattr(scheduler, "timestep_list") and scheduler.timestep_list is not None:
        scheduler.timestep_list = _move_list_tensors(scheduler.timestep_list, device)
    if hasattr(scheduler, "last_sample") and isinstance(scheduler.last_sample, torch.Tensor):
        scheduler.last_sample = scheduler.last_sample.to(device=device, dtype=torch.float32)
    return scheduler


def normalized_latent_pair_difference(
    x_large_next: torch.Tensor,
    x_small_next: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Per-token normalized one-step latent difference [F,H,W].

    Symmetric normalization keeps values mostly within [0,1]:
        ||x_L - x_S|| / (||x_L|| + ||x_S|| + eps)
    """
    num = token_l2(x_large_next - x_small_next)
    denom = token_l2(x_large_next) + token_l2(x_small_next) + eps
    return num / denom


def save_csv(path: str, bin_centers: np.ndarray, percentages: Dict[int, np.ndarray]):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["bin_center"] + [f"step_{s}_token_percentage" for s in percentages.keys()])
        for i, x in enumerate(bin_centers):
            writer.writerow([float(x)] + [float(percentages[s][i]) for s in percentages.keys()])


def plot_distribution(path_pdf: str, path_png: str, bin_centers: np.ndarray, percentages: Dict[int, np.ndarray]):
    plt.figure(figsize=(6.2, 4.0))
    for step, vals in percentages.items():
        plt.plot(bin_centers, vals, linewidth=2, label=f"Step {step}")
    plt.xlabel("Normalized One-step Latent Difference")
    plt.ylabel("Token Percentage (%)")
    plt.xlim(float(bin_centers[0]), float(bin_centers[-1]))
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
    selected_set = set(selected_steps)
    size = parse_size(args.size)
    prompt_items = load_prompt_items(args.prompt_file, base_seed=args.base_seed)

    tmp_dir_obj = tempfile.TemporaryDirectory(prefix="obs1_onestep_latents_", dir=args.output_dir)
    tmp_dir = tmp_dir_obj.name
    entries: List[dict] = []

    print("[Obs1-alt] Pass 1/2: run large-only trajectories, save selected x_t, x_{t-1}^L, and scheduler snapshots.")
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
                # Snapshot scheduler BEFORE the current step. The snapshot will be
                # reused with pred_small in Pass 2, so the one-step update starts
                # from the exact same sampler state as the large update.
                scheduler_snapshot = copy.deepcopy(scheduler)
                latent_next_large = scheduler_step(scheduler, pred_large, t, latent_cur, seed_g)

                entry_path = os.path.join(tmp_dir, f"prompt{pidx:04d}_step{step_id:03d}.pt")
                torch.save({
                    "prompt_index": pidx,
                    "prompt": item.prompt,
                    "seed": item.seed,
                    "step_id": step_id,
                    "timestep": t.detach().cpu(),
                    "latent": latent_cur.detach().cpu().to(torch.float16),
                    "x_next_large": latent_next_large.detach().cpu().to(torch.float16),
                    "scheduler_snapshot": scheduler_snapshot,
                }, entry_path)
                entries.append({"path": entry_path, "prompt_index": pidx, "step_id": step_id})
                latents = [latent_next_large]
            else:
                latent_next = scheduler_step(scheduler, pred_large, t, latent_cur, seed_g)
                latents = [latent_next]

        del context, context_null, latents, scheduler, timesteps
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    cleanup_pipeline(large_pipe)

    print("[Obs1-alt] Pass 2/2: load small model and compare one-step updated latents from the same x_t.")
    small_pipe = create_t2v_pipeline(args.small_task, args.small_ckpt_dir, args.device_id, t5_cpu=args.t5_cpu)
    small_target_shape = compute_target_shape(small_pipe, size, args.frame_num)
    if small_target_shape != target_shape:
        raise RuntimeError(f"Large/small target shapes differ: {target_shape} vs {small_target_shape}")
    small_seq_len = compute_seq_len(small_pipe, small_target_shape)

    hist_edges = np.linspace(0.0, args.max_diff, args.bins + 1, dtype=np.float64)
    hist_counts = {s: np.zeros(args.bins, dtype=np.float64) for s in selected_steps}
    diagnostic_thresholds = parse_float_list(args.diagnostic_thresholds)
    per_entry_stats = []

    small_context_cache = {}
    for entry in tqdm(entries, desc="small one-step updates"):
        obj = _torch_load(entry["path"])
        pidx = int(obj["prompt_index"])
        if pidx not in small_context_cache:
            small_context_cache[pidx] = encode_prompt(small_pipe, obj["prompt"])
        context, context_null = small_context_cache[pidx]

        latent = obj["latent"].to(small_pipe.device, dtype=torch.float32)
        x_next_large = obj["x_next_large"].to(small_pipe.device, dtype=torch.float32)
        timestep = obj["timestep"].to(small_pipe.device)
        scheduler_snapshot = move_scheduler_history_to_device(obj["scheduler_snapshot"], small_pipe.device)

        pred_small = cfg_noise_prediction(
            small_pipe, latent, timestep, context, context_null, small_seq_len, args.guide_scale)
        x_next_small = scheduler_step(scheduler_snapshot, pred_small, timestep, latent, seed_g=None)

        diff = normalized_latent_pair_difference(x_next_large, x_next_small)
        diff_np = diff.detach().float().clamp(0, args.max_diff).cpu().numpy().reshape(-1)
        stats = array_stats(diff_np, thresholds=diagnostic_thresholds)
        stats.update({
            "prompt_index": int(obj["prompt_index"]),
            "seed": int(obj["seed"]),
            "step_id": int(obj["step_id"]),
            "prompt": obj["prompt"],
        })
        per_entry_stats.append(stats)
        counts, _ = np.histogram(diff_np, bins=hist_edges)
        hist_counts[int(obj["step_id"])] += counts.astype(np.float64)

        del obj, latent, x_next_large, pred_small, x_next_small, diff, diff_np, scheduler_snapshot
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    cleanup_pipeline(small_pipe)

    percentages = {}
    for s in selected_steps:
        total = hist_counts[s].sum()
        percentages[s] = hist_counts[s] / max(total, 1.0) * 100.0

    bin_centers = (hist_edges[:-1] + hist_edges[1:]) / 2.0
    save_csv(os.path.join(args.output_dir, "obs1_onestep_latent_difference.csv"), bin_centers, percentages)
    np.savez(os.path.join(args.output_dir, "obs1_onestep_latent_difference_hist.npz"),
             bin_centers=bin_centers, **{f"step_{s}": percentages[s] for s in selected_steps})
    plot_distribution(
        os.path.join(args.output_dir, "fig_obs1_onestep_latent_difference.pdf"),
        os.path.join(args.output_dir, "fig_obs1_onestep_latent_difference.png"),
        bin_centers,
        percentages,
    )

    diag_dir = os.path.join(args.output_dir, "diagnostics")
    ensure_diag_dir(diag_dir)
    write_rows_csv(os.path.join(diag_dir, "per_prompt_step_stats.csv"), per_entry_stats)
    hist_summary = {str(s): weighted_hist_stats(bin_centers, hist_counts[s], diagnostic_thresholds) for s in selected_steps}
    frac_keys = [k for k in per_entry_stats[0].keys() if k.startswith("frac_")] if per_entry_stats else []
    per_prompt_summary = group_numeric_rows(
        per_entry_stats, "step_id", numeric_keys=["mean", "p50", "p90", "p95", "p99"] + frac_keys
    ) if per_entry_stats else {}
    save_json(os.path.join(diag_dir, "summary.json"), {
        "description": "Observation 1 one-step latent-difference diagnostics. Fractions are percentages of spatiotemporal tokens.",
        "selected_steps": selected_steps,
        "diagnostic_thresholds": diagnostic_thresholds,
        "histogram_summary_by_step": hist_summary,
        "per_prompt_summary_by_step": per_prompt_summary,
    })
    if diagnostic_thresholds:
        low_th = diagnostic_thresholds[1] if len(diagnostic_thresholds) > 1 else diagnostic_thresholds[0]
        high_th = diagnostic_thresholds[-1]
        low_tag = f"{low_th:.3f}".rstrip("0").rstrip(".").replace(".", "p")
        high_tag = f"{high_th:.3f}".rstrip("0").rstrip(".").replace(".", "p")
        steps_sorted = sorted(selected_steps)
        low_vals = [hist_summary[str(s)].get(f"frac_le_{low_tag}", 0.0) for s in steps_sorted]
        high_vals = [hist_summary[str(s)].get(f"frac_ge_{high_tag}", 0.0) for s in steps_sorted]
        plot_two_metric_by_step(
            os.path.join(diag_dir, "fig_debug_token_fraction_by_step.pdf"),
            os.path.join(diag_dir, "fig_debug_token_fraction_by_step.png"),
            steps_sorted, low_vals, high_vals,
            f"diff <= {low_th:g}", f"diff >= {high_th:g}",
            "Token Percentage (%)",
            "Low- and high-difference token fractions",
        )
    plot_metric_errorbar_by_step(
        os.path.join(diag_dir, "fig_debug_mean_difference_by_step.pdf"),
        os.path.join(diag_dir, "fig_debug_mean_difference_by_step.png"),
        per_entry_stats, "mean", "Mean One-step Latent Difference",
        "Across-prompt mean one-step latent difference",
    )
    md_lines = [
        "This file is for debugging whether the one-step latent version of Observation 1 supports the paper story.",
        "",
        "Expected useful pattern: many tokens have small large-small one-step latent difference, but a non-trivial tail remains.",
        "",
        "## Histogram-level summary",
    ]
    for s in selected_steps:
        hs = hist_summary[str(s)]
        md_lines.append(f"- Step {s}: mean={hs.get('mean', float('nan')):.4f}, p90={hs.get('p90', float('nan')):.4f}, p95={hs.get('p95', float('nan')):.4f}, p99={hs.get('p99', float('nan')):.4f}")
    write_markdown(os.path.join(diag_dir, "debug_report.md"), "Obs1 One-step Latent Difference Debug Report", md_lines)

    if args.keep_tmp:
        print(f"[Obs1-alt] Temporary entries kept at: {tmp_dir}")
    else:
        tmp_dir_obj.cleanup()
    print(f"[Obs1-alt] Done. Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
