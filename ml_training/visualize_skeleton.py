#!/usr/bin/env python3
"""
Load a clip CSV from train_data and render an animated GIF of the skeleton (stick figure)
to validate extraction and data reading. Uses root-space positions (px, py, pz) per bone.
"""
import argparse
import csv
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection

# Same bone order as extract_training_data.lua
R15_BONES = [
    "LowerTorso", "UpperTorso", "Head",
    "LeftUpperArm", "LeftLowerArm", "LeftHand",
    "RightUpperArm", "RightLowerArm", "RightHand",
    "LeftUpperLeg", "LeftLowerLeg", "LeftFoot",
    "RightUpperLeg", "RightLowerLeg", "RightFoot",
]

# (parent_index, child_index) for drawing segments; indices into R15_BONES
SKELETON_EDGES = [
    (0, 1), (1, 2),   # spine, head
    (1, 3), (3, 4), (4, 5),   # left arm
    (1, 6), (6, 7), (7, 8),   # right arm
    (0, 9), (9, 10), (10, 11),   # left leg
    (0, 12), (12, 13), (13, 14),   # right leg
]


def load_clip_csv(path: str) -> np.ndarray:
    """Load clip CSV -> (T, 15, 3) positions only (px, py, pz per bone)."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) < 1 + 15 * 6:
                continue
            pos = []
            for b in range(15):
                base = 1 + b * 6
                px = float(row[base])
                py = float(row[base + 1])
                pz = float(row[base + 2])
                pos.append([px, py, pz])
            rows.append(pos)
    return np.array(rows, dtype=np.float64)


def main():
    ap = argparse.ArgumentParser(description="Render animated GIF of skeleton from a clip CSV")
    ap.add_argument("csv_path", nargs="?", default=None, help="Path to clip CSV (e.g. ml_training/train_data/123-456.csv)")
    ap.add_argument("--output", "-o", default="ml_training/skeleton_preview.gif", help="Output GIF path")
    ap.add_argument("--fps", type=int, default=15, help="GIF frame rate (lower = smaller file)")
    ap.add_argument("--dpi", type=int, default=80, help="Figure DPI")
    ap.add_argument("--view", choices=["xy", "xz", "yz"], default="xy", help="2D view: xy=front, xz=top, yz=side")
    args = ap.parse_args()

    if args.csv_path is None:
        data_dir = "ml_training/train_data"
        if not os.path.isdir(data_dir):
            raise SystemExit("No csv_path given and ml_training/train_data not found.")
        files = [f for f in os.listdir(data_dir) if f.endswith(".csv") and f != "manifest.csv"]
        if not files:
            raise SystemExit("No clip CSVs in ml_training/train_data. Run extract_training_data.lua first.")
        path = os.path.join(data_dir, files[0])
        print(f"Using first clip: {path}")
    else:
        path = args.csv_path

    if not os.path.isfile(path):
        raise SystemExit(f"File not found: {path}")

    data = load_clip_csv(path)
    T, _, _ = data.shape
    if T == 0:
        raise SystemExit("No frames in CSV.")

    # 2D projection
    if args.view == "xy":
        x_idx, y_idx = 0, 1
    elif args.view == "xz":
        x_idx, y_idx = 0, 2
    else:
        x_idx, y_idx = 1, 2

    x_all = data[:, :, x_idx]
    y_all = data[:, :, y_idx]
    x_min, x_max = x_all.min(), x_all.max()
    y_min, y_max = y_all.min(), y_all.max()
    pad = 0.1
    x_margin = max((x_max - x_min) * pad, 0.1)
    y_margin = max((y_max - y_min) * pad, 0.1)
    x_lim = (x_min - x_margin, x_max + x_margin)
    y_lim = (y_min - y_margin, y_max + y_margin)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_aspect("equal")
    ax.set_xlabel("X" if x_idx == 0 else "Y" if x_idx == 1 else "Z")
    ax.set_ylabel("Y" if y_idx == 1 else "Z")
    ax.set_title(os.path.basename(path))

    scatter = ax.scatter([], [], s=40, c="blue", zorder=2)
    line_segs = LineCollection([], colors="black", linewidths=2, zorder=1)
    ax.add_collection(line_segs)
    frame_text = ax.text(0.02, 0.98, "", transform=ax.transAxes, verticalalignment="top", fontsize=10)

    def init():
        scatter.set_offsets(np.empty((0, 2)))
        line_segs.set_segments([])
        frame_text.set_text("")
        return scatter, line_segs, frame_text

    def update(frame):
        pos = data[frame]
        xy = pos[:, [x_idx, y_idx]]
        scatter.set_offsets(xy)
        segments = []
        for i, j in SKELETON_EDGES:
            segments.append([(pos[i, x_idx], pos[i, y_idx]), (pos[j, x_idx], pos[j, y_idx])])
        line_segs.set_segments(segments)
        frame_text.set_text(f"frame {frame + 1}/{T}")
        return scatter, line_segs, frame_text

    ani = animation.FuncAnimation(
        fig, update, init_func=init, frames=T, interval=1000 / args.fps, blit=False
    )
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    ani.save(args.output, writer="pillow", fps=args.fps, dpi=args.dpi)
    plt.close()
    print(f"Saved {args.output} ({T} frames at {args.fps} fps)")


if __name__ == "__main__":
    main()
