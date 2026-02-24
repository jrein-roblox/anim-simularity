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


def _safe_float(s: str) -> float:
    try:
        x = float(s)
        return 0.0 if not np.isfinite(x) else x
    except (ValueError, TypeError):
        return 0.0


def load_clip_csv(path: str) -> np.ndarray:
    """Load clip CSV -> (T, 15, 3) positions only (px, py, pz per bone). NaN/inf replaced with 0."""
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
                px = _safe_float(row[base])
                py = _safe_float(row[base + 1])
                pz = _safe_float(row[base + 2])
                pos.append([px, py, pz])
            rows.append(pos)
    arr = np.array(rows, dtype=np.float64) if rows else np.zeros((0, 15, 3), dtype=np.float64)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0, copy=False)


def render_one_gif(path: str, output_path: str, fps: int = 15, dpi: int = 80, view: str = "xy") -> bool:
    """Render a single clip CSV to an animated GIF. Returns True on success."""
    data = load_clip_csv(path)
    T, _, _ = data.shape
    if T == 0:
        return False
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

    if view == "xy":
        x_idx, y_idx = 0, 1
    elif view == "xz":
        x_idx, y_idx = 0, 2
    else:
        x_idx, y_idx = 1, 2

    x_all = data[:, :, x_idx]
    y_all = data[:, :, y_idx]
    x_min, x_max = np.nanmin(x_all), np.nanmax(x_all)
    y_min, y_max = np.nanmin(y_all), np.nanmax(y_all)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)
    half_size = max(x_span, y_span, 6.0) / 2
    x_lim = (x_center - half_size, x_center + half_size)
    y_lim = (y_center - half_size, y_center + half_size)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
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
            segments.append([(float(pos[i, x_idx]), float(pos[i, y_idx])), (float(pos[j, x_idx]), float(pos[j, y_idx]))])
        line_segs.set_segments(segments)
        frame_text.set_text(f"frame {frame + 1}/{T}")
        return scatter, line_segs, frame_text

    ani = animation.FuncAnimation(
        fig, update, init_func=init, frames=T, interval=1000 / fps, blit=False
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    ani.save(output_path, writer="pillow", fps=fps, dpi=dpi)
    plt.close()
    return True


def main():
    ap = argparse.ArgumentParser(description="Render animated GIF of skeleton from a clip CSV")
    ap.add_argument("csv_path", nargs="?", default=None, help="Path to clip CSV (e.g. ml_training/train_data/123-456.csv)")
    ap.add_argument("--output", "-o", default="ml_training/skeleton_preview.gif", help="Output GIF path (single mode)")
    ap.add_argument("--out_dir", default="ml_training/skeleton_gifs", help="Output directory in recursive mode")
    ap.add_argument("--recursive", "-r", action="store_true", help="Build GIFs for all CSV files in training directory")
    ap.add_argument("--data_dir", default="ml_training/train_data", help="Training data directory (for recursive mode)")
    ap.add_argument("--fps", type=int, default=15, help="GIF frame rate (lower = smaller file)")
    ap.add_argument("--dpi", type=int, default=80, help="Figure DPI")
    ap.add_argument("--view", choices=["xy", "xz", "yz"], default="xy", help="2D view: xy=front, xz=top, yz=side")
    args = ap.parse_args()

    if args.recursive:
        data_dir = args.data_dir
        if not os.path.isdir(data_dir):
            raise SystemExit(f"Data directory not found: {data_dir}")
        files = sorted([f for f in os.listdir(data_dir) if f.endswith(".csv") and f != "manifest.csv"])
        if not files:
            raise SystemExit(f"No clip CSVs in {data_dir}. Run extract_training_data.lua first.")
        os.makedirs(args.out_dir, exist_ok=True)
        ok = 0
        for i, f in enumerate(files):
            path = os.path.join(data_dir, f)
            base = os.path.splitext(f)[0]
            out_path = os.path.join(args.out_dir, base + ".gif")
            try:
                if render_one_gif(path, out_path, fps=args.fps, dpi=args.dpi, view=args.view):
                    ok += 1
                    print(f"[{i + 1}/{len(files)}] {f} -> {out_path}")
            except Exception as e:
                print(f"[{i + 1}/{len(files)}] {f} skipped: {e}")
        print(f"Done. Rendered {ok}/{len(files)} GIFs to {args.out_dir}")
        return

    if args.csv_path is None:
        data_dir = args.data_dir
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
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

    if render_one_gif(path, args.output, fps=args.fps, dpi=args.dpi, view=args.view):
        print(f"Saved {args.output} ({T} frames at {args.fps} fps)")


if __name__ == "__main__":
    main()
