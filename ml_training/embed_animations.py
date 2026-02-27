#!/usr/bin/env python3
"""
Load trained encoder and packed .npz; output embeddings CSV (animId, clipId, duration, emb1..embK).
"""
import csv
import os
import sys

import numpy as np
import torch

# Import same model as train_autoencoder
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_autoencoder import Encoder


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--packed", default="ml_training/train_data.npz", help="Path to .npz from pack_training_data.py")
    ap.add_argument("--checkpoint_dir", default="ml_training/checkpoints")
    ap.add_argument("--output", default="ml_training/embeddings.csv")
    args = ap.parse_args()

    ckpt_dir = args.checkpoint_dir
    norm_path = os.path.join(ckpt_dir, "norm.npz")
    enc_path = os.path.join(ckpt_dir, "encoder.pt")
    if not os.path.isfile(norm_path) or not os.path.isfile(enc_path):
        raise SystemExit("Run train_autoencoder.py first to create norm.npz and encoder.pt")

    norm = np.load(norm_path, allow_pickle=True)
    T_max = int(norm["T_max"])
    input_dim = int(norm["input_dim"])  # frames only (T_max*90)
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

    if not os.path.isfile(args.packed):
        raise SystemExit(f"Packed file not found: {args.packed}. Run pack_training_data.py first.")
    data = np.load(args.packed, allow_pickle=True)
    frames = data["frames"]
    anim_ids = data["anim_id"]
    clip_ids = data["clip_id"]
    durations = data["duration"]
    N = frames.shape[0]
    rows = []
    with torch.no_grad():
        for i in range(N):
            x = np.asarray(frames[i], dtype=np.float32).reshape(1, -1)
            x = (x - mean) / std
            z = encoder(torch.from_numpy(x)).numpy()[0]
            row = [str(anim_ids[i]), str(clip_ids[i]), f"{float(durations[i]):.6f}"] + [f"{z[j]:.8f}" for j in range(len(z))]
            rows.append(row)
            if (i + 1) % 1000 == 0:
                print(f"Processed {i + 1} clips")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["animId", "clipId", "duration"] + [f"emb{i+1}" for i in range(latent_dim)])
        w.writerows(rows)
    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
