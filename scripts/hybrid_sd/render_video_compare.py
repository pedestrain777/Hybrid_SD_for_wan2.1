#!/usr/bin/env python3
import argparse
from pathlib import Path

import imageio.v3 as iio
import matplotlib.pyplot as plt


def _read_mid_frame(path: Path):
    frames = iio.imread(path)
    return frames[len(frames) // 2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True)
    parser.add_argument("--left-title", default="left")
    parser.add_argument("--right", required=True)
    parser.add_argument("--right-title", default="right")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    items = [
        (args.left_title, Path(args.left)),
        (args.right_title, Path(args.right)),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (title, path) in zip(axes, items):
        ax.imshow(_read_mid_frame(path))
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(args.out, dpi=160)
    print(args.out)


if __name__ == "__main__":
    main()
