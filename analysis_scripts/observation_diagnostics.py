"""Lightweight diagnostics for ST-Hybrid observation scripts.

These helpers intentionally depend only on numpy/matplotlib/stdlib so they can
run in the same environment as the Wan observation scripts. The diagnostics are
for debugging motivation figures: they save extra CSV/JSON/plots that make it
possible to judge whether the resulting curves support the intended paper story.
"""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np


def parse_float_list(text: str) -> List[float]:
    vals = []
    for x in str(text).split(','):
        x = x.strip()
        if x:
            vals.append(float(x))
    return vals


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(path: str, obj) -> None:
    def convert(x):
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
        return x
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=convert)


def write_csv(path: str, rows: Sequence[Mapping], fieldnames: Optional[Sequence[str]] = None) -> None:
    if not rows:
        with open(path, 'w', encoding='utf-8') as f:
            f.write('')
        return
    if fieldnames is None:
        keys = []
        for r in rows:
            for k in r.keys():
                if k not in keys:
                    keys.append(k)
        fieldnames = keys
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, '') for k in fieldnames})


def array_stats(values: np.ndarray, thresholds: Sequence[float] = (0.05, 0.1, 0.2, 0.3), prefix: str = '') -> Dict[str, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    arr = arr[np.isfinite(arr)]
    out: Dict[str, float] = {f'{prefix}num_tokens': int(arr.size)}
    if arr.size == 0:
        for k in ['mean', 'std', 'min', 'max', 'p50', 'p75', 'p90', 'p95', 'p99']:
            out[f'{prefix}{k}'] = float('nan')
        return out
    out.update({
        f'{prefix}mean': float(np.mean(arr)),
        f'{prefix}std': float(np.std(arr)),
        f'{prefix}min': float(np.min(arr)),
        f'{prefix}max': float(np.max(arr)),
        f'{prefix}p50': float(np.percentile(arr, 50)),
        f'{prefix}p75': float(np.percentile(arr, 75)),
        f'{prefix}p90': float(np.percentile(arr, 90)),
        f'{prefix}p95': float(np.percentile(arr, 95)),
        f'{prefix}p99': float(np.percentile(arr, 99)),
    })
    for th in thresholds:
        tag = f'{th:.3f}'.rstrip('0').rstrip('.').replace('.', 'p')
        out[f'{prefix}frac_le_{tag}'] = float(np.mean(arr <= th) * 100.0)
        out[f'{prefix}frac_ge_{tag}'] = float(np.mean(arr >= th) * 100.0)
    return out


def weighted_hist_stats(bin_centers: np.ndarray, counts: np.ndarray, thresholds: Sequence[float]) -> Dict[str, float]:
    x = np.asarray(bin_centers, dtype=np.float64)
    w = np.asarray(counts, dtype=np.float64)
    total = float(np.sum(w))
    out = {'num_tokens': int(total)}
    if total <= 0:
        return out
    mean = float(np.sum(x * w) / total)
    var = float(np.sum(((x - mean) ** 2) * w) / total)
    out.update({'mean': mean, 'std': var ** 0.5})
    cdf = np.cumsum(w) / total
    for q in [50, 75, 90, 95, 99]:
        idx = int(np.searchsorted(cdf, q / 100.0, side='left'))
        idx = min(max(idx, 0), len(x) - 1)
        out[f'p{q}'] = float(x[idx])
    for th in thresholds:
        tag = f'{th:.3f}'.rstrip('0').rstrip('.').replace('.', 'p')
        out[f'frac_le_{tag}'] = float(np.sum(w[x <= th]) / total * 100.0)
        out[f'frac_ge_{tag}'] = float(np.sum(w[x >= th]) / total * 100.0)
    return out


def group_numeric_rows(rows: Sequence[Mapping], group_key: str, numeric_keys: Optional[Sequence[str]] = None) -> Dict[str, Dict[str, float]]:
    groups = defaultdict(list)
    for r in rows:
        groups[str(r[group_key])].append(r)
    out = {}
    if numeric_keys is None:
        numeric_keys = []
        for r in rows:
            for k, v in r.items():
                if k == group_key:
                    continue
                if isinstance(v, (int, float, np.integer, np.floating)) and k not in numeric_keys:
                    numeric_keys.append(k)
    for g, rs in groups.items():
        cur = {'count': len(rs)}
        for k in numeric_keys:
            vals = []
            for r in rs:
                try:
                    v = float(r.get(k, np.nan))
                except Exception:
                    v = np.nan
                if np.isfinite(v):
                    vals.append(v)
            if vals:
                arr = np.asarray(vals, dtype=np.float64)
                cur[f'{k}_mean'] = float(np.mean(arr))
                cur[f'{k}_std'] = float(np.std(arr))
                cur[f'{k}_min'] = float(np.min(arr))
                cur[f'{k}_max'] = float(np.max(arr))
        out[g] = cur
    return out


def curve_metrics(curve: np.ndarray, prefix: str = '') -> Dict[str, float]:
    y = np.asarray(curve, dtype=np.float64).reshape(-1)
    y = y[np.isfinite(y)]
    out: Dict[str, float] = {f'{prefix}num_segments': int(y.size)}
    if y.size == 0:
        return out
    mean = float(np.mean(y))
    std = float(np.std(y))
    peak_idx = int(np.argmax(y))
    out.update({
        f'{prefix}mean': mean,
        f'{prefix}std': std,
        f'{prefix}cv': float(std / (mean + 1e-8)),
        f'{prefix}min': float(np.min(y)),
        f'{prefix}max': float(np.max(y)),
        f'{prefix}p50': float(np.percentile(y, 50)),
        f'{prefix}p90': float(np.percentile(y, 90)),
        f'{prefix}peak_segment': peak_idx,
        f'{prefix}peak_to_mean': float(np.max(y) / (mean + 1e-8)),
        f'{prefix}dynamic_range': float(np.max(y) - np.min(y)),
    })
    return out


def plot_step_bar(path_pdf: str, path_png: str, step_to_value: Mapping[int, float], ylabel: str, title: str = '') -> None:
    steps = sorted(int(s) for s in step_to_value.keys())
    vals = [float(step_to_value[s]) for s in steps]
    plt.figure(figsize=(5.0, 3.4))
    plt.bar([str(s) for s in steps], vals)
    plt.xlabel('Denoising Step')
    plt.ylabel(ylabel)
    if title:
        plt.title(title, fontsize=10)
    plt.grid(True, axis='y', alpha=0.25)
    plt.tight_layout()
    plt.savefig(path_pdf, bbox_inches='tight')
    plt.savefig(path_png, dpi=300, bbox_inches='tight')
    plt.close()


def plot_two_metric_by_step(path_pdf: str, path_png: str, steps: Sequence[int], y1: Sequence[float], y2: Sequence[float], label1: str, label2: str, ylabel: str, title: str = '') -> None:
    x = np.arange(len(steps))
    width = 0.36
    plt.figure(figsize=(5.8, 3.6))
    plt.bar(x - width / 2, y1, width, label=label1)
    plt.bar(x + width / 2, y2, width, label=label2)
    plt.xticks(x, [str(s) for s in steps])
    plt.xlabel('Denoising Step')
    plt.ylabel(ylabel)
    if title:
        plt.title(title, fontsize=10)
    plt.grid(True, axis='y', alpha=0.25)
    plt.legend(frameon=True)
    plt.tight_layout()
    plt.savefig(path_pdf, bbox_inches='tight')
    plt.savefig(path_png, dpi=300, bbox_inches='tight')
    plt.close()


def plot_metric_errorbar_by_step(path_pdf: str, path_png: str, per_prompt_rows: Sequence[Mapping], value_key: str, ylabel: str, title: str = '') -> None:
    groups = defaultdict(list)
    for r in per_prompt_rows:
        if value_key in r:
            try:
                groups[int(r['step_id'])].append(float(r[value_key]))
            except Exception:
                pass
    steps = sorted(groups.keys())
    means = [float(np.mean(groups[s])) for s in steps]
    stds = [float(np.std(groups[s])) for s in steps]
    plt.figure(figsize=(5.2, 3.6))
    plt.errorbar(steps, means, yerr=stds, marker='o', capsize=4, linewidth=2)
    plt.xlabel('Denoising Step')
    plt.ylabel(ylabel)
    if title:
        plt.title(title, fontsize=10)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(path_pdf, bbox_inches='tight')
    plt.savefig(path_png, dpi=300, bbox_inches='tight')
    plt.close()


def plot_temporal_overlay(path_pdf: str, path_png: str, curves: Sequence[np.ndarray], ylabel: str, title: str = '') -> None:
    if not curves:
        return
    max_len = max(len(c) for c in curves)
    stack = []
    plt.figure(figsize=(6.2, 3.6))
    for c in curves:
        y = np.asarray(c, dtype=np.float64)
        x = np.arange(len(y))
        plt.plot(x, y, linewidth=0.8, alpha=0.25)
        if len(y) == max_len:
            stack.append(y)
    if stack:
        arr = np.stack(stack, axis=0)
        mean = np.mean(arr, axis=0)
        plt.plot(np.arange(max_len), mean, linewidth=2.5, label='Mean')
        plt.legend(frameon=True)
    plt.xlabel('Temporal Segment Index')
    plt.ylabel(ylabel)
    if title:
        plt.title(title, fontsize=10)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(path_pdf, bbox_inches='tight')
    plt.savefig(path_png, dpi=300, bbox_inches='tight')
    plt.close()


def plot_topk_mass_curve(path_pdf: str, path_png: str, heatmap: np.ndarray, title: str = '') -> Dict[str, float]:
    flat = np.asarray(heatmap, dtype=np.float64).reshape(-1)
    flat = flat[np.isfinite(flat)]
    flat = np.maximum(flat, 0)
    out: Dict[str, float] = {}
    if flat.size == 0 or flat.sum() <= 0:
        return out
    sorted_vals = np.sort(flat)[::-1]
    cumsum = np.cumsum(sorted_vals)
    pct_x = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals) * 100.0
    pct_y = cumsum / cumsum[-1] * 100.0
    plt.figure(figsize=(5.2, 3.6))
    plt.plot(pct_x, pct_y, linewidth=2)
    for p in [5, 10, 20, 30, 40, 50]:
        idx = min(len(sorted_vals) - 1, max(0, int(round(len(sorted_vals) * p / 100.0)) - 1))
        out[f'top_{p}_percent_mass'] = float(pct_y[idx])
        plt.scatter([p], [pct_y[idx]], s=20)
    plt.xlabel('Top spatial positions (%)')
    plt.ylabel('Cumulative update mass (%)')
    if title:
        plt.title(title, fontsize=10)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(path_pdf, bbox_inches='tight')
    plt.savefig(path_png, dpi=300, bbox_inches='tight')
    plt.close()
    return out


def write_markdown(path: str, title: str, lines: Sequence[str]) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f'# {title}\n\n')
        for line in lines:
            f.write(str(line).rstrip() + '\n')
