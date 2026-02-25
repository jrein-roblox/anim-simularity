#!/usr/bin/env python3
"""
Train an MLP auto-encoder on pose sequences (T_max x 90 per clip).
Export PyTorch checkpoint and encoder weights for Lua (encoder_weights.lua).
"""
import argparse
import csv
import os
import sys

import numpy as np
import torch
import torch.nn as nn

# Defaults (must match Lua inference)
FEAT_PER_FRAME = 90
T_MAX = 90
HIDDEN_DIMS = [512, 256]
LATENT_DIM = 128


def _safe_float(s: str) -> float:
    if not isinstance(s, str):
        s = str(s)
    s_lower = s.strip().lower()
    if s_lower in ("nan", "-nan", "nan(ind)", "-nan(ind)", "inf", "-inf", "+inf", ""):
        return 0.0
    try:
        x = float(s)
        return 0.0 if not np.isfinite(x) else x
    except (ValueError, TypeError):
        return 0.0


def load_manifest(data_dir: str):
    manifest_path = os.path.join(data_dir, "manifest.csv")
    if not os.path.isfile(manifest_path):
        return None
    rows = []
    with open(manifest_path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if len(row) >= 4:
                nf = max(0, int(_safe_float(row[3])))
                rows.append((row[0], row[1], _safe_float(row[2]), nf))
    return rows


def load_clip_csv(path: str) -> np.ndarray:
    """Load one clip CSV -> (T, 90). Replaces NaN/inf with 0."""
    data = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) < 1 + FEAT_PER_FRAME:
                continue
            row_floats = [_safe_float(row[i]) for i in range(1, 1 + FEAT_PER_FRAME)]
            data.append(row_floats)
    arr = np.array(data, dtype=np.float32) if data else np.zeros((0, FEAT_PER_FRAME), dtype=np.float32)
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0, copy=False)


def load_all_data(data_dir: str, manifest: list, t_max: int, spread_frames: bool = True):
    """Load all clips; pad/truncate or sample to (N, t_max, 90); flatten to (N, t_max*90).
    If spread_frames is True and clip has T > t_max, sample t_max frames uniformly across the clip duration."""
    if not manifest:
        # Fallback: glob *-*.csv (skip manifest.csv)
        manifest = []
        for f in os.listdir(data_dir):
            if f.endswith(".csv") and f != "manifest.csv":
                m = f[:-4].split("-")
                if len(m) == 2:
                    path = os.path.join(data_dir, f)
                    arr = load_clip_csv(path)
                    manifest.append((m[0], m[1], 0.0, arr.shape[0]))
    X_list = []
    for item in manifest:
        if len(item) == 4:
            anim_id, clip_id, duration, num_frames = item
        else:
            anim_id, clip_id = item[0], item[1]
        path = os.path.join(data_dir, f"{anim_id}-{clip_id}.csv")
        if not os.path.isfile(path):
            continue
        arr = load_clip_csv(path)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        T = arr.shape[0]
        if T == 0:
            arr = np.zeros((t_max, FEAT_PER_FRAME), dtype=np.float32)
        elif T < t_max:
            pad = np.repeat(arr[-1:], t_max - T, axis=0)
            arr = np.concatenate([arr, pad], axis=0)
        elif T > t_max:
            if spread_frames:
                indices = np.linspace(0, T - 1, t_max, dtype=np.int64)
                arr = arr[indices]
            else:
                arr = arr[:t_max]
        else:
            pass
        X_list.append(arr)
    if not X_list:
        raise SystemExit("No clips loaded. Run extract_training_data.lua first.")
    X = np.stack(X_list, axis=0)
    N, _, _ = X.shape
    X_flat = X.reshape(N, -1)
    return X_flat


class Encoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, latent_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.latent_dim = latent_dim
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, latent_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dims: list, output_dim: int):
        super().__init__()
        layers = []
        prev = latent_dim
        for h in reversed(hidden_dims):
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class AutoEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list, latent_dim: int):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dims, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dims, input_dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def export_encoder_weights_lua(encoder: Encoder, out_path: str, t_max: int):
    """Write encoder weights to a Lua file that returns a table. Each layer: weight[out][in], bias[out]."""
    linear_layers = [m for m in encoder.net if isinstance(m, nn.Linear)]
    lines = [
        "-- Encoder weights (auto-generated); input dim = T_max * 90, ReLU hidden, linear out.",
        "return {",
        f"  T_max = {t_max},",
        f"  inputDim = {encoder.input_dim},",
        f"  latentDim = {encoder.latent_dim},",
        "  layers = {",
    ]
    for idx, mod in enumerate(linear_layers):
        w = mod.weight.detach().cpu().numpy()
        b = mod.bias.detach().cpu().numpy()
        lines.append(f"    {{  -- layer {idx + 1}: out={w.shape[0]}, in={w.shape[1]}")
        lines.append("      weight = {")
        for i in range(w.shape[0]):
            row = ", ".join(f"{w[i, j]:.8f}" for j in range(w.shape[1]))
            lines.append(f"        {{ {row} }},")
        lines.append("      },")
        lines.append("      bias = { " + ", ".join(f"{b[j]:.8f}" for j in range(b.shape[0])) + " },")
        lines.append("    },")
    lines.append("  },")
    lines.append("}")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="ml_training/train_data", help="Directory with clip CSVs and manifest.csv")
    ap.add_argument("--checkpoint_dir", default="ml_training/checkpoints", help="Where to save .pt and encoder_weights.lua")
    ap.add_argument("--T_max", type=int, default=T_MAX, help="Max frames per clip (input length)")
    ap.add_argument("--hidden", type=int, nargs="+", default=HIDDEN_DIMS, help="Hidden layer sizes")
    ap.add_argument("--latent", type=int, default=LATENT_DIM, help="Latent dimension")
    ap.add_argument("--epochs", type=int, default=30, help="Training epochs")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--spread_frames", action="store_true", default=True, help="When clip is longer than T_max, sample T_max frames uniformly across duration instead of taking the first T_max")
    args = ap.parse_args()

    data_dir = args.data_dir
    checkpoint_dir = args.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    manifest = load_manifest(data_dir)
    X = load_all_data(data_dir, manifest, t_max=args.T_max, spread_frames=args.spread_frames)
    N, input_dim = X.shape
    print(f"Loaded {N} clips, input_dim={input_dim}")

    # Normalize (optional): use mean/std of training set
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-6
    X = (X - mean) / std

    # Train/val split
    perm = np.random.default_rng(42).permutation(N)
    n_val = int(N * args.val_ratio)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X_train = torch.from_numpy(X[train_idx]).float().to(device)
    X_val = torch.from_numpy(X[val_idx]).float().to(device)

    model = AutoEncoder(input_dim, args.hidden, args.latent).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()

    for ep in range(args.epochs):
        model.train()
        perm_train = torch.randperm(X_train.size(0), device=device)
        total_loss = 0.0
        n_batches = 0
        for i in range(0, X_train.size(0), args.batch_size):
            idx = perm_train[i : i + args.batch_size]
            batch = X_train[idx]
            recon, _ = model(batch)
            loss = mse(recon, batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        train_loss = total_loss / max(n_batches, 1)
        model.eval()
        with torch.no_grad():
            recon_val, _ = model(X_val)
            val_loss = mse(recon_val, X_val).item()
        if (ep + 1) % 10 == 0 or ep == 0:
            print(f"Epoch {ep + 1}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}")

    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "autoencoder.pt"))
    torch.save(model.encoder.state_dict(), os.path.join(checkpoint_dir, "encoder.pt"))
    model = model.cpu()
    # Export normalization for embed script
    np.savez(
        os.path.join(checkpoint_dir, "norm.npz"),
        mean=mean.astype(np.float32),
        std=std.astype(np.float32),
        T_max=args.T_max,
        input_dim=input_dim,
        latent_dim=args.latent,
        hidden_dims=np.array(args.hidden),
    )
    export_encoder_weights_lua(model.encoder, os.path.join(checkpoint_dir, "encoder_weights.lua"), args.T_max)
    print(f"Saved checkpoints and encoder_weights.lua to {checkpoint_dir}")


if __name__ == "__main__":
    main()
