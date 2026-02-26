#!/usr/bin/env python3
"""
Read all clip CSVs from data_dir and write a single .npz file for fast loading.

Output npz contains:
  frames: (N, T_max, 90) float32
  energy: (N, 15) float32
  anim_id: (N,) string array
  clip_id: (N,) string array
  duration: (N,) float32
  T_max: int scalar
  spread_frames: bool (1/0)

Run after extraction; then use --packed <path> with train_autoencoder.py and embed_animations.py.
"""
import argparse
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from train_autoencoder import (
    FEAT_PER_FRAME,
    load_clip_csv,
    load_manifest,
    compute_bone_energy,
)

# Match trainer default
T_MAX = 90


def _pad_or_sample(arr: np.ndarray, t_max: int, spread_frames: bool) -> np.ndarray:
    T = arr.shape[0]
    if T == 0:
        return np.zeros((t_max, FEAT_PER_FRAME), dtype=np.float32)
    if T < t_max:
        pad = np.repeat(arr[-1:], t_max - T, axis=0)
        return np.concatenate([arr, pad], axis=0).astype(np.float32)
    if T > t_max:
        if spread_frames:
            indices = np.linspace(0, T - 1, t_max, dtype=np.int64)
            return arr[indices].astype(np.float32)
        return arr[:t_max].astype(np.float32)
    return arr.astype(np.float32)


def _load_one(args_tuple):
    data_dir, item, t_max, spread_frames = args_tuple
    anim_id, clip_id = item[0], item[1]
    duration = item[2] if len(item) >= 3 else 0.0
    path = os.path.join(data_dir, f"{anim_id}-{clip_id}.csv")
    if not os.path.isfile(path):
        return None
    arr = load_clip_csv(path)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    energy = compute_bone_energy(arr)
    arr = _pad_or_sample(arr, t_max, spread_frames)
    if arr.shape[0] != t_max or arr.shape[1] != FEAT_PER_FRAME:
        out = np.zeros((t_max, FEAT_PER_FRAME), dtype=np.float32)
        r, c = min(arr.shape[0], t_max), min(arr.shape[1], FEAT_PER_FRAME)
        out[:r, :c] = arr[:r, :c]
        if r < t_max and r > 0:
            out[r:] = out[r - 1 : r]
        arr = out
    return (anim_id, clip_id, duration, arr, energy.astype(np.float32))


def main():
    ap = argparse.ArgumentParser(description="Pack clip CSVs into one .npz for fast training/embedding")
    ap.add_argument("--data_dir", default="ml_training/train_data", help="Directory with clip CSVs and manifest.csv")
    ap.add_argument("--output", default="ml_training/train_data.npz", help="Output .npz path")
    ap.add_argument("--T_max", type=int, default=T_MAX, help="Frames per clip (must match training)")
    ap.add_argument("--spread_frames", action="store_true", default=True, help="Sample T_max frames across clip when longer")
    ap.add_argument("--workers", type=int, default=8, help="Parallel workers (0 = single-threaded)")
    args = ap.parse_args()

    manifest = load_manifest(args.data_dir)
    if not manifest:
        manifest = []
        for f in sorted(os.listdir(args.data_dir)):
            if f.endswith(".csv") and f != "manifest.csv":
                parts = f[:-4].split("-")
                if len(parts) == 2:
                    manifest.append((parts[0], parts[1], 0.0))

    if not manifest:
        raise SystemExit("No clips found. Run extract_training_data.lua first.")

    work = [(args.data_dir, item, args.T_max, args.spread_frames) for item in manifest]
    results = []
    if args.workers and args.workers > 0:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            for r in ex.map(_load_one, work):
                if r is not None:
                    results.append(r)
    else:
        for w in work:
            r = _load_one(w)
            if r is not None:
                results.append(r)

    if not results:
        raise SystemExit("No clips loaded.")

    frames = np.stack([r[3] for r in results], axis=0)
    energy = np.stack([r[4] for r in results], axis=0)
    anim_id = np.array([r[0] for r in results], dtype=object)
    clip_id = np.array([r[1] for r in results], dtype=object)
    duration = np.array([r[2] for r in results], dtype=np.float32)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez(
        args.output,
        frames=frames,
        energy=energy,
        anim_id=anim_id,
        clip_id=clip_id,
        duration=duration,
        T_max=np.int64(args.T_max),
        spread_frames=np.int64(1 if args.spread_frames else 0),
    )
    print(f"Packed {len(results)} clips to {args.output} (frames {frames.shape}, energy {energy.shape})")


if __name__ == "__main__":
    main()
