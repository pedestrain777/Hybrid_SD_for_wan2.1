#!/usr/bin/env python3
"""Compare full-large videos against hybrid videos with frame-level metrics."""

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--large_root", required=True)
    p.add_argument("--hybrid_root", required=True)
    p.add_argument("--output_csv", required=True)
    p.add_argument("--frame_dir", default=None)
    p.add_argument("--prompts", nargs="+", required=True)
    p.add_argument("--sample_idx", type=int, default=0)
    p.add_argument("--max_frames", type=int, default=0, help="0 means all frames")
    return p.parse_args()


def read_video(path: Path, max_frames: int = 0):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames decoded: {path}")
    return frames


def ssim_gray(a: np.ndarray, b: np.ndarray):
    a = cv2.cvtColor(a, cv2.COLOR_RGB2GRAY).astype(np.float32)
    b = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float32)
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    mu_a = cv2.GaussianBlur(a, (11, 11), 1.5)
    mu_b = cv2.GaussianBlur(b, (11, 11), 1.5)
    mu_a2 = mu_a * mu_a
    mu_b2 = mu_b * mu_b
    mu_ab = mu_a * mu_b
    sigma_a2 = cv2.GaussianBlur(a * a, (11, 11), 1.5) - mu_a2
    sigma_b2 = cv2.GaussianBlur(b * b, (11, 11), 1.5) - mu_b2
    sigma_ab = cv2.GaussianBlur(a * b, (11, 11), 1.5) - mu_ab
    score_map = ((2 * mu_ab + c1) * (2 * sigma_ab + c2)) / (
        (mu_a2 + mu_b2 + c1) * (sigma_a2 + sigma_b2 + c2)
    )
    return float(np.mean(score_map))


def psnr(a: np.ndarray, b: np.ndarray):
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    if mse <= 1e-12:
        return float("inf")
    return float(20.0 * math.log10(255.0 / math.sqrt(mse)))


def frame_diff_strength(frames):
    if len(frames) < 2:
        return []
    vals = []
    for prev, cur in zip(frames[:-1], frames[1:]):
        vals.append(float(np.mean(np.abs(cur.astype(np.float32) - prev.astype(np.float32)))))
    return vals


def save_contact_sheet(large_frames, hybrid_frames, prompt: str, out_path: Path):
    indices = [0, len(large_frames) // 4, len(large_frames) // 2, 3 * len(large_frames) // 4, len(large_frames) - 1]
    rows = []
    font = cv2.FONT_HERSHEY_SIMPLEX
    for idx in indices:
        large = large_frames[idx].copy()
        hybrid = hybrid_frames[idx].copy()
        diff = np.clip(np.abs(large.astype(np.int16) - hybrid.astype(np.int16)) * 4, 0, 255).astype(np.uint8)
        for label, img in [("large", large), ("hybrid", hybrid), ("absdiff x4", diff)]:
            cv2.putText(img, f"{label} f={idx}", (18, 38), font, 1.0, (255, 40, 40), 2, cv2.LINE_AA)
        row = np.concatenate([large, hybrid, diff], axis=1)
        rows.append(row)
    sheet = np.concatenate(rows, axis=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR))


def summarize_pair(prompt: str, large_path: Path, hybrid_path: Path, frame_dir: Path | None, max_frames: int):
    large = read_video(large_path, max_frames=max_frames)
    hybrid = read_video(hybrid_path, max_frames=max_frames)
    n = min(len(large), len(hybrid))
    large = large[:n]
    hybrid = hybrid[:n]
    if large[0].shape != hybrid[0].shape:
        hybrid = [cv2.resize(f, (large[0].shape[1], large[0].shape[0]), interpolation=cv2.INTER_AREA) for f in hybrid]

    ssim_vals = []
    psnr_vals = []
    mae_vals = []
    rmse_vals = []
    for a, b in zip(large, hybrid):
        diff = a.astype(np.float32) - b.astype(np.float32)
        ssim_vals.append(ssim_gray(a, b))
        psnr_vals.append(psnr(a, b))
        mae_vals.append(float(np.mean(np.abs(diff))))
        rmse_vals.append(float(np.sqrt(np.mean(diff * diff))))

    worst_idx = int(np.argmin(ssim_vals))
    head = slice(0, min(5, n))
    tail = slice(min(5, n), n)
    large_motion = frame_diff_strength(large)
    hybrid_motion = frame_diff_strength(hybrid)
    motion_delta = [abs(a - b) for a, b in zip(large_motion, hybrid_motion)]

    if frame_dir is not None:
        safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in prompt)[:100]
        save_contact_sheet(large, hybrid, prompt, frame_dir / f"{safe}.jpg")

    return {
        "prompt": prompt,
        "frames": n,
        "large_path": str(large_path),
        "hybrid_path": str(hybrid_path),
        "ssim_mean": np.mean(ssim_vals),
        "ssim_min": np.min(ssim_vals),
        "ssim_min_frame": worst_idx,
        "ssim_first5": np.mean(ssim_vals[head]),
        "ssim_exclude_first5": np.mean(ssim_vals[tail]) if n > 5 else np.mean(ssim_vals),
        "psnr_mean": np.mean(psnr_vals),
        "psnr_first5": np.mean(psnr_vals[head]),
        "psnr_exclude_first5": np.mean(psnr_vals[tail]) if n > 5 else np.mean(psnr_vals),
        "mae_mean": np.mean(mae_vals),
        "mae_first5": np.mean(mae_vals[head]),
        "mae_exclude_first5": np.mean(mae_vals[tail]) if n > 5 else np.mean(mae_vals),
        "rmse_mean": np.mean(rmse_vals),
        "large_motion_mean": np.mean(large_motion) if large_motion else 0.0,
        "hybrid_motion_mean": np.mean(hybrid_motion) if hybrid_motion else 0.0,
        "motion_delta_mean": np.mean(motion_delta) if motion_delta else 0.0,
        "large_brightness": np.mean([np.mean(f) for f in large]),
        "hybrid_brightness": np.mean([np.mean(f) for f in hybrid]),
    }


def main():
    args = parse_args()
    large_root = Path(args.large_root)
    hybrid_root = Path(args.hybrid_root)
    frame_dir = Path(args.frame_dir) if args.frame_dir else None
    rows = []
    for prompt in args.prompts:
        name = f"{prompt}-{args.sample_idx}.mp4"
        rows.append(summarize_pair(
            prompt=prompt,
            large_path=large_root / name,
            hybrid_path=hybrid_root / name,
            frame_dir=frame_dir,
            max_frames=args.max_frames,
        ))
    out = Path(args.output_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(out)
    for row in rows:
        print(
            f"{row['prompt']}: SSIM={row['ssim_mean']:.4f} "
            f"PSNR={row['psnr_mean']:.2f} MAE={row['mae_mean']:.2f} "
            f"motion_delta={row['motion_delta_mean']:.2f}"
        )


if __name__ == "__main__":
    main()
