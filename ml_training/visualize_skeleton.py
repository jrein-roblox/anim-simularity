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
from mpl_toolkits.mplot3d import Axes3D

# Same bone order as extract_training_data.lua / train_autoencoder2.py
R15_BONES = [
    "LowerTorso", "UpperTorso", "Head",
    "LeftUpperArm", "LeftLowerArm", "LeftHand",
    "RightUpperArm", "RightLowerArm", "RightHand",
    "LeftUpperLeg", "LeftLowerLeg", "LeftFoot",
    "RightUpperLeg", "RightLowerLeg", "RightFoot",
]

# R15 mirroring: same defaults as train_autoencoder2.py (mirror across X = left-right)
R15_MIRROR_AXIS = "x"
R15_MIRROR_PAIRS = [(3, 6), (4, 7), (5, 8), (9, 12), (10, 13), (11, 14)]  # L/R arm, L/R leg (0-based)

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


def build_swap_index(num_bones: int, pairs: list) -> np.ndarray:
    """Build index array: after swap, bone i comes from position swap_idx[i]."""
    idx = np.arange(num_bones, dtype=np.int64)
    for a, b in pairs:
        if 0 <= a < num_bones and 0 <= b < num_bones:
            idx[a], idx[b] = idx[b], idx[a]
    return idx


def mirror_positions(data: np.ndarray, axis: str = "x", swap_idx: np.ndarray | None = None) -> np.ndarray:
    """
    Mirror (T, 15, 3) root-space positions for R15: swap L/R bones, then negate axis.
    Matches train_autoencoder2.py mirror logic (position part only).
    """
    out = data.copy()
    if swap_idx is not None:
        out = out[:, swap_idx, :]  # (T, 15, 3)
    ax = axis.lower()
    if ax == "x":
        out[..., 0] = -out[..., 0]
    elif ax == "y":
        out[..., 1] = -out[..., 1]
    else:
        out[..., 2] = -out[..., 2]
    return out


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


def _limits_2d(data: np.ndarray, x_idx: int, y_idx: int, min_size: float = 6.0):
    """Compute equal-aspect limits for a 2D projection of (T, 15, 3) data."""
    x_all = data[:, :, x_idx]
    y_all = data[:, :, y_idx]
    x_min, x_max = np.nanmin(x_all), np.nanmax(x_all)
    y_min, y_max = np.nanmin(y_all), np.nanmax(y_all)
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)
    half = max(x_span, y_span, min_size) / 2
    return (x_center - half, x_center + half), (y_center - half, y_center + half)


def _segments_for_frame(pos: np.ndarray, x_idx: int, y_idx: int) -> list:
    """Line segments for skeleton edges in 2D (x_idx, y_idx) projection."""
    segments = []
    for i, j in SKELETON_EDGES:
        segments.append([
            (float(pos[i, x_idx]), float(pos[i, y_idx])),
            (float(pos[j, x_idx]), float(pos[j, y_idx])),
        ])
    return segments


def _limits_3d(data: np.ndarray, min_size: float = 6.0):
    """Equal-scale 3D box limits from (T, N, 3) data."""
    x = data[:, :, 0]
    y = data[:, :, 1]
    z = data[:, :, 2]
    cx = (np.nanmin(x) + np.nanmax(x)) / 2
    cy = (np.nanmin(y) + np.nanmax(y)) / 2
    cz = (np.nanmin(z) + np.nanmax(z)) / 2
    half = max(
        np.nanmax(x) - np.nanmin(x),
        np.nanmax(y) - np.nanmin(y),
        np.nanmax(z) - np.nanmin(z),
        min_size,
    ) / 2
    return (cx - half, cx + half), (cy - half, cy + half), (cz - half, cz + half)


def render_one_gif(
    path: str,
    output_path: str,
    fps: int = 15,
    dpi: int = 80,
    view: str = "xy",
    data: np.ndarray | None = None,
    title_override: str | None = None,
    side_views: bool = True,
) -> bool:
    """Render a single clip CSV (or preloaded data) to an animated GIF. Returns True on success.
    If side_views is True, main view is 3D at ~45° and Front/Side/Top orthographic views are on the right."""
    if data is None:
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

    # 2D limits for when main view is 2D (no side_views)
    x_lim, y_lim = _limits_2d(data, x_idx, y_idx)
    # Orthographic panel limits: Front (xy), Side (-Z,Y), Top (xz)
    x_lim_xy, y_lim_xy = _limits_2d(data, 0, 1, min_size=4.0)
    side_x = -data[:, :, 2]
    side_y = data[:, :, 1]
    x_lim_side = (np.nanmin(side_x) - 1e-6, np.nanmax(side_x) + 1e-6)
    y_lim_side = (np.nanmin(side_y) - 1e-6, np.nanmax(side_y) + 1e-6)
    side_half = max(np.nanmax(side_x) - np.nanmin(side_x), np.nanmax(side_y) - np.nanmin(side_y), 4.0) / 2
    x_lim_side = ((x_lim_side[0] + x_lim_side[1]) / 2 - side_half, (x_lim_side[0] + x_lim_side[1]) / 2 + side_half)
    y_lim_side = ((y_lim_side[0] + y_lim_side[1]) / 2 - side_half, (y_lim_side[0] + y_lim_side[1]) / 2 + side_half)
    x_lim_xz, y_lim_xz = _limits_2d(data, 0, 2, min_size=4.0)

    if side_views:
        fig = plt.figure(figsize=(9, 6))
        # Main: 3D view on the left at ~45° (elev 30°, azim 45°)
        ax_main = fig.add_axes([0.02, 0.05, 0.62, 0.9], projection="3d")
        # 3D plot uses (X, Z, Y) so Y is up (matplotlib’s z-axis = vertical)
        x_lim_3d, y_lim_3d, z_lim_3d = _limits_3d(data, min_size=6.0)
        ax_main.set_xlim(x_lim_3d)
        ax_main.set_ylim(z_lim_3d)
        ax_main.set_zlim(y_lim_3d)
        ax_main.view_init(elev=30, azim=-45)
        ax_main.set_xlabel("X")
        ax_main.set_ylabel("Z")
        ax_main.set_zlabel("Y (up)")
        ax_main.set_title(title_override if title_override is not None else os.path.basename(path))
        scatter_main = ax_main.scatter([], [], [], s=25, c="blue")
        lines_3d = []
        for (i, j) in SKELETON_EDGES:
            (line,) = ax_main.plot([], [], [], "k-", linewidth=2)
            lines_3d.append((i, j, line))
        frame_text = ax_main.text2D(0.02, 0.98, "", transform=ax_main.transAxes, verticalalignment="top", fontsize=10)

        # Right column: Front (xy), Side (-Z,Y), Top (xz) — top to bottom
        ax_front = fig.add_axes([0.66, 0.68, 0.32, 0.28])
        ax_side = fig.add_axes([0.66, 0.36, 0.32, 0.28])
        ax_top = fig.add_axes([0.66, 0.04, 0.32, 0.28])
        for ax, xl, yl, xlabel, ylabel, title in [
            (ax_front, x_lim_xy, y_lim_xy, "X", "Y", "Front"),
            (ax_side, x_lim_side, y_lim_side, "-Z", "Y", "Side"),
            (ax_top, x_lim_xz, y_lim_xz, "X", "Z", "Top"),
        ]:
            ax.set_aspect("equal")
            ax.set_xlim(xl)
            ax.set_ylim(yl)
            ax.set_xlabel(xlabel, fontsize=8)
            ax.set_ylabel(ylabel, fontsize=8)
            ax.set_title(title, fontsize=9)
        scatter_front = ax_front.scatter([], [], s=18, c="blue", zorder=2)
        line_front = LineCollection([], colors="black", linewidths=1.5, zorder=1)
        ax_front.add_collection(line_front)
        scatter_side = ax_side.scatter([], [], s=18, c="blue", zorder=2)
        line_side = LineCollection([], colors="black", linewidths=1.5, zorder=1)
        ax_side.add_collection(line_side)
        scatter_top = ax_top.scatter([], [], s=18, c="blue", zorder=2)
        line_top = LineCollection([], colors="black", linewidths=1.5, zorder=1)
        ax_top.add_collection(line_top)
    else:
        fig, ax_main = plt.subplots(figsize=(6, 6))
        ax_main.set_aspect("equal")
        ax_main.set_xlim(x_lim)
        ax_main.set_ylim(y_lim)
        ax_main.set_xlabel("X" if x_idx == 0 else "Y" if x_idx == 1 else "Z")
        ax_main.set_ylabel("Y" if y_idx == 1 else "Z")
        ax_main.set_title(title_override if title_override is not None else os.path.basename(path))
        scatter_main = ax_main.scatter([], [], s=40, c="blue", zorder=2)
        line_main = LineCollection([], colors="black", linewidths=2, zorder=1)
        ax_main.add_collection(line_main)
        frame_text = ax_main.text(0.02, 0.98, "", transform=ax_main.transAxes, verticalalignment="top", fontsize=10)
        lines_3d = None

    def init():
        if side_views:
            scatter_main._offsets3d = ([], [], [])
            for _, __, line in lines_3d:
                line.set_data_3d([], [], [])
            frame_text.set_text("")
            scatter_front.set_offsets(np.empty((0, 2)))
            line_front.set_segments([])
            scatter_side.set_offsets(np.empty((0, 2)))
            line_side.set_segments([])
            scatter_top.set_offsets(np.empty((0, 2)))
            line_top.set_segments([])
            return [scatter_main] + [l for _, __, l in lines_3d] + [frame_text, scatter_front, line_front, scatter_side, line_side, scatter_top, line_top]
        else:
            scatter_main.set_offsets(np.empty((0, 2)))
            line_main.set_segments([])
            frame_text.set_text("")
            return [scatter_main, line_main, frame_text]

    def update(frame):
        pos = data[frame]
        frame_text.set_text(f"frame {frame + 1}/{T}")
        if side_views:
            # 3D: (X, Z, Y) so Y is up
            scatter_main._offsets3d = (pos[:, 0], pos[:, 2], pos[:, 1])
            for i, j, line in lines_3d:
                line.set_data_3d([pos[i, 0], pos[j, 0]], [pos[i, 2], pos[j, 2]], [pos[i, 1], pos[j, 1]])
            scatter_front.set_offsets(pos[:, [0, 1]])
            line_front.set_segments(_segments_for_frame(pos, 0, 1))
            side_xy = np.column_stack([-pos[:, 2], pos[:, 1]])
            scatter_side.set_offsets(side_xy)
            side_pos = np.column_stack([-pos[:, 2], pos[:, 1], np.zeros(pos.shape[0])])
            line_side.set_segments(_segments_for_frame(side_pos, 0, 1))
            scatter_top.set_offsets(pos[:, [0, 2]])
            line_top.set_segments(_segments_for_frame(pos, 0, 2))
            return [scatter_main] + [l for _, __, l in lines_3d] + [frame_text, scatter_front, line_front, scatter_side, line_side, scatter_top, line_top]
        else:
            xy = pos[:, [x_idx, y_idx]]
            scatter_main.set_offsets(xy)
            line_main.set_segments(_segments_for_frame(pos, x_idx, y_idx))
            return [scatter_main, line_main, frame_text]

    ani = animation.FuncAnimation(
        fig, update, init_func=init, frames=T, interval=1000 / fps, blit=False
    )
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    ani.save(output_path, writer="pillow", fps=fps, dpi=dpi)
    plt.close()
    return True


def render_mirrored_gif(
    path: str,
    output_path: str,
    fps: int = 15,
    dpi: int = 80,
    view: str = "xy",
    mirror_axis: str = R15_MIRROR_AXIS,
    mirror_pairs: list = R15_MIRROR_PAIRS,
    side_views: bool = True,
) -> bool:
    """Load clip, mirror it (R15 L/R swap + axis flip), and render to GIF. Returns True on success."""
    data = load_clip_csv(path)
    T, _, _ = data.shape
    if T == 0:
        return False
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    swap_idx = build_swap_index(15, mirror_pairs)
    data_mirrored = mirror_positions(data, axis=mirror_axis, swap_idx=swap_idx)
    base = os.path.basename(path)
    return render_one_gif(
        path,
        output_path,
        fps=fps,
        dpi=dpi,
        view=view,
        data=data_mirrored,
        title_override=f"{base} (mirrored)",
        side_views=side_views,
    )


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
    ap.add_argument("--side-views", action="store_true", default=True, help="Main view = 3D at 45°; Front/Side/Top orthographic views on the right.")
    ap.add_argument("--no-side-views", action="store_false", dest="side_views", help="Single main view only.")
    ap.add_argument("--mirrored", action="store_true", default=True, help="Also render a mirrored GIF for debugging (R15 L/R swap + axis flip).")
    ap.add_argument("--no-mirrored", action="store_false", dest="mirrored", help="Disable mirrored GIF output.")
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
                if render_one_gif(path, out_path, fps=args.fps, dpi=args.dpi, view=args.view, side_views=args.side_views):
                    ok += 1
                    print(f"[{i + 1}/{len(files)}] {f} -> {out_path}")
                    if args.mirrored:
                        out_mirrored = os.path.join(args.out_dir, base + "_mirrored.gif")
                        if render_mirrored_gif(path, out_mirrored, fps=args.fps, dpi=args.dpi, view=args.view, side_views=args.side_views):
                            print(f"         mirrored -> {out_mirrored}")
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

    if render_one_gif(path, args.output, fps=args.fps, dpi=args.dpi, view=args.view, side_views=args.side_views):
        print(f"Saved {args.output} ({T} frames at {args.fps} fps)")
    if args.mirrored:
        out_mirrored = args.output.replace(".gif", "_mirrored.gif")
        if render_mirrored_gif(path, out_mirrored, fps=args.fps, dpi=args.dpi, view=args.view, side_views=args.side_views):
            print(f"Saved mirrored {out_mirrored}")


if __name__ == "__main__":
    main()
