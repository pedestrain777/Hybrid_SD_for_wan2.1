# ST-Hybrid Observation Scripts for Wan2.1

These scripts are diagnostic motivation experiments. They do **not** run ST-Hybrid, do **not** decode final videos, and do **not** evaluate VBench/FID/latency.

## Observation 1: why large-small collaboration is feasible/necessary

```bash
python analysis_scripts/obs1_prediction_discrepancy.py \
  --large_ckpt_dir /path/to/Wan2.1-T2V-14B \
  --small_ckpt_dir /path/to/Wan2.1-T2V-1.3B \
  --prompt_file analysis_scripts/prompts_observation_demo.txt \
  --size 832*480 \
  --frame_num 81 \
  --sample_steps 50 \
  --sample_shift 5.0 \
  --guide_scale 5.0 \
  --selected_steps 10,25,40 \
  --output_dir analysis_outputs/obs1
```

Output figure:

- `fig_obs1_prediction_discrepancy.pdf/png`

Meaning:

- x-axis: normalized prediction difference between large and small denoisers.
- y-axis: token percentage.
- curves: denoising steps 10, 25, 40.

## Observation 2(a): temporal non-uniformity of large-model updates

```bash
python analysis_scripts/obs2a_temporal_update_curve.py \
  --large_ckpt_dir /path/to/Wan2.1-T2V-14B \
  --prompt_file analysis_scripts/prompts_observation_demo.txt \
  --size 832*480 \
  --frame_num 81 \
  --sample_steps 50 \
  --sample_shift 5.0 \
  --guide_scale 5.0 \
  --selected_steps 10,25,40 \
  --middle_step 25 \
  --segment_len 1 \
  --save_middle_maps \
  --output_dir analysis_outputs/obs2a
```

Output figure:

- `fig_obs2a_temporal_update_curve.pdf/png`

Meaning:

- x-axis: temporal segment index.
- y-axis: normalized large-model update magnitude, i.e., step_diff.
- curves: denoising steps 10, 25, 40.

## Observation 2(b): spatial sparsity inside a hard temporal segment

Recommended: use the statistics from Observation 2(a).

```bash
python analysis_scripts/obs2b_spatial_update_heatmap.py \
  --stats_file analysis_outputs/obs2a/obs2a_temporal_update_stats.pt \
  --output_dir analysis_outputs/obs2b
```

Output figure:

- `fig_obs2b_spatial_update_heatmap.pdf/png`

Meaning:

- x-axis: latent width.
- y-axis: latent height.
- color: spatial update magnitude inside the hardest temporal segment at the middle step.

The script also reports how much total update magnitude is covered by the top-k% spatial positions. This number can be used to support spatial sparsity quantitatively.

## Observation 1 alternative: one-step latent difference

This is an alternative version of Observation 1. It is still a controlled comparison, but it compares the **one-step updated latent** instead of the direct model prediction.

```bash
python analysis_scripts/obs1_onestep_latent_difference.py \
  --large_ckpt_dir /path/to/Wan2.1-T2V-14B \
  --small_ckpt_dir /path/to/Wan2.1-T2V-1.3B \
  --prompt_file analysis_scripts/prompts_observation_demo.txt \
  --size 832*480 \
  --frame_num 81 \
  --sample_steps 50 \
  --sample_shift 5.0 \
  --guide_scale 5.0 \
  --selected_steps 10,25,40 \
  --output_dir analysis_outputs/obs1_onestep
```

Output figure:

- `fig_obs1_onestep_latent_difference.pdf/png`

Meaning:

- x-axis: normalized one-step latent difference between large and small denoisers.
- y-axis: token percentage.
- curves: denoising steps 10, 25, 40.

This script uses the same latent `x_t` for both models, then compares:

```text
x_{t-1}^L = scheduler(x_t, pred_large)
x_{t-1}^S = scheduler(x_t, pred_small)
```

It does **not** compare two independently generated large/small trajectories, so it avoids accumulated trajectory drift.

## Observation 2(a) alternative: temporal large-small prediction discrepancy

This is an alternative temporal observation. Instead of measuring the large model's own `step_diff`, it measures **how much the small model deviates from the large model at each temporal segment** under the same latent state.

It is useful for showing the expected stage pattern:

- early step: high discrepancy over most temporal segments, so the small model is not ready to replace the large model;
- middle step: mixed high/low discrepancy, so only some temporal segments need the large model;
- late step: low discrepancy over most temporal segments, so the small model can replace the large model more broadly.

```bash
python analysis_scripts/obs2a_temporal_prediction_discrepancy_curve.py \
  --large_ckpt_dir /path/to/Wan2.1-T2V-14B \
  --small_ckpt_dir /path/to/Wan2.1-T2V-1.3B \
  --prompt_file analysis_scripts/prompts_observation_demo.txt \
  --size 832*480 \
  --frame_num 81 \
  --sample_steps 50 \
  --sample_shift 5.0 \
  --guide_scale 5.0 \
  --selected_steps 10,25,40 \
  --middle_step 25 \
  --segment_len 1 \
  --output_dir analysis_outputs/obs2a_temporal_pred_diff
```

Output figure:

- `fig_obs2a_temporal_prediction_discrepancy.pdf/png`

Meaning:

- x-axis: temporal segment index.
- y-axis: mean large-small prediction difference.
- curves: denoising steps 10, 25, 40.

By default, the y-axis keeps the raw normalized discrepancy so different steps remain comparable. If the absolute scale makes the curves hard to view, add `--normalize_plot_by_prompt_max`, but use the raw version for paper interpretation whenever possible.

## Added diagnostic outputs

All five scripts now create a `diagnostics/` subfolder in their `--output_dir`. These files are intended for checking whether the observation figures actually support the paper story before deciding which version to put into the manuscript.

Typical diagnostic files include:

- `diagnostics/debug_report.md`: short human-readable summary and expected pattern.
- `diagnostics/summary.json`: machine-readable summary statistics.
- `diagnostics/per_prompt_step_stats.csv` or `diagnostics/per_prompt_step_curve_metrics.csv`: prompt-level statistics so outlier prompts can be inspected.
- `diagnostics/fig_debug_*.png/pdf`: auxiliary plots such as token fractions, across-prompt means, temporal CV, all-prompt temporal overlays, and top-k spatial mass curves.

Recommended quick checks:

1. For `obs1_prediction_discrepancy.py` and `obs1_onestep_latent_difference.py`, open:
   - `diagnostics/debug_report.md`
   - `diagnostics/fig_debug_token_fraction_by_step.png`
   - `diagnostics/fig_debug_mean_discrepancy_by_step.png` or `fig_debug_mean_difference_by_step.png`

   Useful pattern: many low-difference tokens plus a non-trivial high-difference tail.

2. For `obs2a_temporal_update_curve.py`, open:
   - `diagnostics/fig_debug_raw_mean_update_by_step.png`
   - `diagnostics/fig_debug_temporal_cv_by_step.png`
   - `diagnostics/fig_debug_all_prompts_step25_raw_curves.png`

   Useful pattern: early step has high raw update, middle step has strong temporal non-uniformity, and late step has lower update.

   Note: the main plot now keeps raw `step_diff` magnitude by default. Add `--normalize_each_curve_by_mean` only if you want a shape-only visualization.

3. For `obs2b_spatial_update_heatmap.py`, open:
   - `diagnostics/fig_debug_topk_spatial_mass_curve.png`
   - `diagnostics/fig_debug_top20_spatial_mask.png` if `--topk_percent 20`
   - `diagnostics/debug_report.md`

   Useful pattern: top 10% or 20% spatial locations cover a large fraction of update mass.

4. For `obs2a_temporal_prediction_discrepancy_curve.py`, open:
   - `diagnostics/fig_debug_mean_temporal_discrepancy_by_step.png`
   - `diagnostics/fig_debug_temporal_cv_by_step.png`
   - `diagnostics/fig_debug_all_prompts_step25_temporal_pred_diff.png`

   Useful pattern: step 10 is generally high, step 25 has mixed temporal peaks, and step 40 is lower.
