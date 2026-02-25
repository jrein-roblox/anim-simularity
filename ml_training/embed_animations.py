#!/usr/bin/env python3
"""
Load trained encoder and clip CSVs; output embeddings CSV (animId, clipId, duration, emb1..embK).
"""
import csv
import os
import sys

import numpy as np
import torch

# Import same model as train_autoencoder
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_autoencoder import Encoder, load_clip_csv, load_manifest, FEAT_PER_FRAME


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="ml_training/train_data")
    ap.add_argument("--checkpoint_dir", default="ml_training/checkpoints")
    ap.add_argument("--output", default="ml_training/embeddings.csv")
    ap.add_argument("--spread_frames", action="store_true", help="Sample T_max frames across clip (use if model was trained with --spread_frames)")
    args = ap.parse_args()

    ckpt_dir = args.checkpoint_dir
    norm_path = os.path.join(ckpt_dir, "norm.npz")
    enc_path = os.path.join(ckpt_dir, "encoder.pt")
    if not os.path.isfile(norm_path) or not os.path.isfile(enc_path):
        raise SystemExit("Run train_autoencoder.py first to create norm.npz and encoder.pt")

    norm = np.load(norm_path, allow_pickle=True)
    T_max = int(norm["T_max"])
    input_dim = int(norm["input_dim"])
    latent_dim = int(norm["latent_dim"])
    h = norm.get("hidden_dims")
    if h is not None and getattr(h, "size", 0) > 0:
        hidden_dims = h.tolist() if hasattr(h, "tolist") else list(h)
    else:
        hidden_dims = [512, 256]
    mean = norm["mean"]
    std = norm["std"]

    encoder = Encoder(input_dim, hidden_dims, latent_dim)
    state = torch.load(enc_path, map_location="cpu", weights_only=True)
    encoder.load_state_dict(state)
    encoder.eval()

    manifest = load_manifest(args.data_dir)
    if not manifest:
        manifest = []
        for f in os.listdir(args.data_dir):
            if f.endswith(".csv") and f != "manifest.csv":
                parts = f[:-4].split("-")
                if len(parts) == 2:
                    path = os.path.join(args.data_dir, f)
                    arr = load_clip_csv(path)
                    manifest.append((parts[0], parts[1], 0.0, arr.shape[0]))

    rows = []
    with torch.no_grad():
        for item in manifest:
            if len(item) == 4:
                anim_id, clip_id, duration, num_frames = item
            else:
                anim_id, clip_id = item[0], item[1]
                duration = 0.0
            path = os.path.join(args.data_dir, f"{anim_id}-{clip_id}.csv")
            if not os.path.isfile(path):
                continue
            arr = load_clip_csv(path)
            T = arr.shape[0]
            if T == 0:
                arr = np.zeros((T_max, FEAT_PER_FRAME), dtype=np.float32)
            elif T < T_max:
                pad = np.repeat(arr[-1:], T_max - T, axis=0)
                arr = np.concatenate([arr, pad], axis=0)
            else:
                if args.spread_frames:
                    indices = np.linspace(0, T - 1, T_max).astype(np.int64)
                    arr = arr[indices]
                else:
                    arr = arr[:T_max]
            x = arr.reshape(1, -1).astype(np.float32)
            x = (x - mean) / std
            z = encoder(torch.from_numpy(x)).numpy()[0]
            row = [anim_id, clip_id, f"{duration:.6f}"] + [f"{z[i]:.8f}" for i in range(len(z))]
            rows.append(row)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["animId", "clipId", "duration"] + [f"emb{i+1}" for i in range(latent_dim)])
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
