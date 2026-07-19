#!/usr/bin/env python3
"""CPU regression for bounded two-stage dynamic switching."""

import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from compression.hybrid_sd.diffusers.pipeline_wan import HybridWanPipeline
from compression.hybrid_sd.inference_pipeline import HybridVideoInferencePipeline


class SwitchHarness:
    _dynamic_switch_enabled = HybridWanPipeline._dynamic_switch_enabled
    _update_dynamic_switch = HybridWanPipeline._update_dynamic_switch


def make_pipe(threshold=0.20, min_step=2, max_step=4):
    pipe = SwitchHarness()
    pipe.step_config = {
        "dynamic_switch": True,
        "dynamic_switch_min_step": min_step,
        "dynamic_switch_max_step": max_step,
        "dynamic_switch_threshold": threshold,
        "dynamic_switch_patience": 2,
    }
    pipe.dynamic_switch_trace = []
    pipe.dynamic_switch_step = None
    pipe._dynamic_switch_previous_x0 = None
    return pipe


def main():
    noise = torch.zeros(1, 1, 1, 2, 2)
    pipe = make_pipe()
    pipe._update_dynamic_switch(0, torch.ones_like(noise), noise, 0.0)
    pipe._update_dynamic_switch(1, torch.full_like(noise, 1.10), noise, 0.0)
    assert pipe.dynamic_switch_step is None
    pipe._update_dynamic_switch(2, torch.full_like(noise, 1.15), noise, 0.0)
    assert pipe.dynamic_switch_step == 3

    forced = make_pipe(threshold=0.0, min_step=2, max_step=2)
    forced._update_dynamic_switch(0, torch.ones_like(noise), noise, 0.0)
    forced._update_dynamic_switch(1, torch.full_like(noise, 2.0), noise, 0.0)
    assert forced.dynamic_switch_step == 2

    args = SimpleNamespace(
        stage_steps=[30, 20],
        steps=[30, 20],
        hybrid_dynamic_switch=True,
        hybrid_dynamic_switch_min_step=28,
        hybrid_dynamic_switch_max_step=38,
        hybrid_dynamic_switch_threshold=0.20,
        hybrid_dynamic_switch_patience=2,
    )
    runner = HybridVideoInferencePipeline(["/large", "/small"], 0, "cpu", args)
    total, config = runner.get_step_config(args)
    assert total == 50 and config["dynamic_switch"]
    assert config["mode"][37] == "large" and config["mode"][38] == "hybrid"
    args.stage_steps = [50, 0]
    _, baseline_config = runner.get_step_config(args)
    assert not baseline_config["dynamic_switch"]
    print("dynamic switch regression passed: stable and forced paths obey bounds")


if __name__ == "__main__":
    main()
