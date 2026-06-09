# Hybrid Router Simplification Notes

This version replaces the previous rule-heavy Hybrid ROI router with a simplified two-stage cube router.

## Main Hybrid routing flow

1. **Temporal routing** uses frame-wise latent difference:
   `mean(|z_t - z_{t-1}|)` is used to locate hard temporal segments.

2. **Spatial routing** uses exactly one selectable spatial cue inside the selected temporal segments:
   - `cfg`: CFG gap, i.e. `|noise_cond - noise_uncond|`
   - `motion`: frame-wise latent difference map
   - `warp`: cheap warp inconsistency with small integer shift search

3. **Cube construction**:
   The selected spatial score map is thresholded by `spatial_top_ratio`, converted to connected components, and the strongest component becomes the core cube.

4. **Large-model correction**:
   The small model predicts the full-video noise. The large model only predicts selected outer cube crops. Only the core cube region is pasted back into the small model's full noise prediction before the scheduler update.

## Main parameters

- `stage_steps`: `[large_steps, hybrid_steps, small_steps]`
- `hybrid_spatial_cue`: `cfg`, `motion`, or `warp`
- `hybrid_temporal_top_ratio`: how many hard temporal positions to select
- `hybrid_max_temporal_segments`: maximum hard temporal segments per Hybrid step
- `hybrid_spatial_top_ratio`: how many high-score spatial latent positions to select
- `hybrid_max_cubes`: maximum large-model cubes per Hybrid step
- `hybrid_margin_t/h/w`: context margin for the large-model outer crop

## CLI additions in `run_hybrid_complex_landscape.py`

```bash
python run_hybrid_complex_landscape.py 0 --stages 30,10,10 --spatial-cue cfg
python run_hybrid_complex_landscape.py 0 --stages 30,10,10 --spatial-cue motion
python run_hybrid_complex_landscape.py 0 --stages 30,10,10 --spatial-cue warp
```

Optional environment variables:

- `WAN_HYBRID_SPATIAL_CUE`
- `WAN_HYBRID_TEMPORAL_TOP_RATIO`
- `WAN_HYBRID_SPATIAL_TOP_RATIO`
- `WAN_HYBRID_MAX_CUBES`

The old tube/NMS/smoothing/trajectory-curvature routing code has been removed from the active router path.
