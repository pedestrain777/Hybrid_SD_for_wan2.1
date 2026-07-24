# Hybrid SD for Wan2.1 - V2

Research implementation of two-stage Wan2.1 14B/1.3B collaborative video generation.

## Active pipeline

1. Run the 14B model on the full latent video.
2. Dynamically switch between large steps 28 and 38 after two consecutive stable clean-sample estimates.
3. In the Hybrid stage, run the 1.3B model on the full latent and the 14B model on at most two routed spatiotemporal crops.
4. Preserve full-video Wan RoPE coordinates and feather the large-model correction into the small-model prediction.

The default spatial cue is the small-model CFG gap. Temporal routing uses adjacent latent-frame differences.
The default dynamic-switch threshold is `0.15`, selected from a 20-prompt VBench-1.0 calibration sweep.

## Environment

The validated server environment is:

```bash
conda activate minyu_lee
```

Model paths can be provided with `--model-large` and `--model-small`, or the environment variables `WAN_HYBRID_MODEL_LARGE` and `WAN_HYBRID_MODEL_SMALL`.

## Generate one video

On the validated H200 server, the checked model paths make the default command directly runnable:

```bash
python run_hybrid_complex_landscape.py 0 "a black dog running on green grass"
```

For another machine, override the model paths explicitly:

```bash
python run_hybrid_complex_landscape.py 0 \
  --model-large /path/to/Wan2.1-T2V-14B-Diffusers \
  --model-small /path/to/Wan2.1-T2V-1.3B-Diffusers \
  --output-dir results/generated \
  "a black dog running on green grass"
```

The production default disables router debug logging, debug files, and unused cue computation. For a routing investigation, enable them explicitly:

```bash
python run_hybrid_complex_landscape.py 0 --debug-router "prompt"
python run_hybrid_complex_landscape.py 0 --debug-router --debug-all-cues "prompt"
```

Use `--fixed-switch --stages 30,20` for a fixed 30-step Large / 20-step Hybrid ablation. Use `--stages 50,0` for Large-only.

## Relevant source files

- `compression/hybrid_sd/diffusers/pipeline_wan.py`: denoising, dynamic switching, RoPE offset, and fusion.
- `compression/hybrid_sd/routers/video_mask_router.py`: temporal/spatial routing and ROI construction.
- `compression/hybrid_sd/inference_pipeline.py`: model loading and stage configuration.
- `scripts/hybrid_sd/smoke_*.py`: fast regressions.
- `vbench1_minimal_v2_pack/`: internal small-sample evaluation helper, not an official benchmark submission.

## Fast regressions

```bash
python scripts/hybrid_sd/smoke_router_debug.py --out-dir /tmp/router_smoke
python scripts/hybrid_sd/smoke_rope_offset.py
python scripts/hybrid_sd/smoke_dynamic_switch.py
```

Generated videos, logs, router dumps, PID files, model weights, and full VBench installations are intentionally excluded from Git.
