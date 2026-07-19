#!/usr/bin/env python3
"""Fast CPU regression for patch-aligned ROI bounds and global Wan RoPE offsets."""

import sys
from pathlib import Path

import torch
from diffusers.models.transformers.transformer_wan import WanRotaryPosEmbed

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from compression.hybrid_sd.diffusers.pipeline_wan import HybridWanPipeline
from compression.hybrid_sd.routers.video_mask_router import _expand_align_bounds


def main():
    start, end = _expand_align_bounds(
        start=7,
        end=19,
        limit=30,
        margin=4,
        min_size=8,
        align=2,
    )
    assert start <= 3 and end >= 23
    assert start % 2 == 0 and (end - start) % 2 == 0

    rope = WanRotaryPosEmbed(
        attention_head_dim=12,
        patch_size=(1, 2, 2),
        max_seq_len=64,
    )
    full = torch.zeros(1, 4, 5, 30, 52)
    crop = full[:, :, 1:4, 6:22, 12:38]
    full_cos, full_sin = rope(full)
    crop_cos, crop_sin = HybridWanPipeline._wan_rotary_emb_with_offset(
        rope,
        crop,
        position_offset=(1, 6, 12),
    )

    full_cos = full_cos.reshape(5, 15, 26, 1, 12)[1:4, 3:11, 6:19].reshape_as(crop_cos)
    full_sin = full_sin.reshape(5, 15, 26, 1, 12)[1:4, 3:11, 6:19].reshape_as(crop_sin)
    torch.testing.assert_close(crop_cos, full_cos, rtol=0, atol=0)
    torch.testing.assert_close(crop_sin, full_sin, rtol=0, atol=0)

    try:
        HybridWanPipeline._wan_rotary_emb_with_offset(rope, crop, (1, 5, 12))
    except ValueError:
        pass
    else:
        raise AssertionError("unaligned ROI origin must be rejected")

    print("RoPE offset regression passed: crop embeddings exactly match the full-video slice")


if __name__ == "__main__":
    main()
