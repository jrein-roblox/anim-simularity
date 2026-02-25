#!/usr/bin/env python3
"""
Train an MLP auto-encoder on pose sequences (T_max x 90 per clip).
Export PyTorch checkpoint and encoder weights for Lua (encoder_weights.lua).
"""
import argparse
import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import torch
import torch.nn as nn

# Defaults (must match Lua inference)
FEAT_PER_FRAME = 90  # 15 bones × 6 (pos xyz + quatlog xyz)
NUM_BONES = 15
ENERGY_DIM = 15  # accumulated delta pos + quatlog per bone
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
    """Load one clip CSV -> (T, 90). Replaces NaN/inf with 0. Uses numpy for fast parsing."""
    try:
        # usecols 1..90 (skip frame index col 0); invalid/NaN -> 0 via filling_values
        arr = np.genfromtxt(
            path,
            delimiter=",",
            skip_header=1,
            usecols=range(1, 1 + FEAT_PER_FRAME),
            dtype=np.float32,
            filling_values=0.0,
            invalid_raise=False,
            encoding="utf-8",
        )
    except Exception:
        arr = np.zeros((0, FEAT_PER_FRAME), dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.size == 0:
        arr = np.zeros((0, FEAT_PER_FRAME), dtype=np.float32)
    # Ensure exactly FEAT_PER_FRAME columns (pad or truncate) so all clips stack
    if arr.shape[1] < FEAT_PER_FRAME:
        pad = np.zeros((arr.shape[0], FEAT_PER_FRAME - arr.shape[1]), dtype=np.float32)
        arr = np.hstack([arr, pad])
    elif arr.shape[1] > FEAT_PER_FRAME:
        arr = arr[:, :FEAT_PER_FRAME]
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0, copy=False)


def compute_bone_energy(arr: np.ndarray) -> np.ndarray:
    """Per-bone energy: sum of frame-to-frame L2 delta (pos) + L2 delta (quatlog). arr (T, 90) -> (15,)."""
    T = arr.shape[0]
    out = np.zeros(ENERGY_DIM, dtype=np.float32)
    if T < 2:
        return out
    for b in range(NUM_BONES):
        pos = arr[:, b * 6 : b * 6 + 3]
        quat = arr[:, b * 6 + 3 : b * 6 + 6]
        dpos = np.diff(pos, axis=0)
        dquat = np.diff(quat, axis=0)
        out[b] = np.sqrt((dpos ** 2).sum(axis=1)).sum() + np.sqrt((dquat ** 2).sum(axis=1)).sum()
    return out


def _load_one_clip(args_tuple):
    """Worker: (data_dir, item, t_max, spread_frames) -> (frames_arr, energy) or None."""
    data_dir, item, t_max, spread_frames = args_tuple
    if len(item) == 4:
        anim_id, clip_id = item[0], item[1]
    else:
        anim_id, clip_id = item[0], item[1]
    path = os.path.join(data_dir, f"{anim_id}-{clip_id}.csv")
    if not os.path.isfile(path):
        return None
    arr = load_clip_csv(path)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    energy = compute_bone_energy(arr)
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
    # Guarantee shape (t_max, FEAT_PER_FRAME) so np.stack never fails
    arr = np.asarray(arr, dtype=np.float32)
    if arr.shape[0] != t_max or arr.shape[1] != FEAT_PER_FRAME:
        out = np.zeros((t_max, FEAT_PER_FRAME), dtype=np.float32)
        r = min(arr.shape[0], t_max)
        c = min(arr.shape[1], FEAT_PER_FRAME)
        out[:r, :c] = arr[:r, :c]
        if r < t_max and r > 0:
            out[r:] = np.repeat(out[r - 1 : r], t_max - r, axis=0)
        arr = out
    return (arr, energy)


def load_all_data(
    data_dir: str,
    manifest: list,
    t_max: int,
    spread_frames: bool = True,
    num_workers: int = 0,
):
    """Load all clips; pad/truncate or sample to (N, t_max, 90); compute per-bone energy from raw clip;
    flatten frames to (N, t_max*90) and concat energy -> (N, t_max*90+15).
    If num_workers > 0, load CSVs in parallel."""
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
    work = [(data_dir, item, t_max, spread_frames) for item in manifest]
    if num_workers and num_workers > 0:
        frame_list = []
        energy_list = []
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            for result in ex.map(_load_one_clip, work):
                if result is not None:
                    frame_list.append(result[0])
                    energy_list.append(result[1])
    else:
        frame_list = []
        energy_list = []
        for w in work:
            result = _load_one_clip(w)
            if result is not None:
                frame_list.append(result[0])
                energy_list.append(result[1])
    if not frame_list:
        raise SystemExit("No clips loaded. Run extract_training_data.lua first.")
    X_frames = np.stack(frame_list, axis=0)
    X_energy = np.stack(energy_list, axis=0)
    N = X_frames.shape[0]
    frame_dim = t_max * FEAT_PER_FRAME
    X = np.concatenate([X_frames.reshape(N, -1), X_energy], axis=1)
    return X, frame_dim


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
    def __init__(self, input_dim: int, frame_dim: int, hidden_dims: list, latent_dim: int):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dims, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dims, frame_dim)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def infonce_loss(z: torch.Tensor, z_aug: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    """
    InfoNCE contrastive loss. z and z_aug are (B, L) latents for two views of the same batch.
    For anchor z_i the positive is z_aug_i; negatives are all other 2B-2 samples.
    Expects z and z_aug to be L2-normalized (done inside for safety).
    """
    z = z / (z.norm(dim=1, keepdim=True) + 1e-8)
    z_aug = z_aug / (z_aug.norm(dim=1, keepdim=True) + 1e-8)
    B = z.size(0)
    # Stack into (2B, L); rows 0..B-1 are z, rows B..2B-1 are z_aug
    features = torch.cat([z, z_aug], dim=0)
    logits = (features @ features.T) / tau
    # Mask self-similarity to avoid trivial solution
    mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
    logits = logits.masked_fill(mask, -1e9)
    # For anchor i (0..B-1) positive is B+i; for anchor B+i positive is i
    labels = torch.cat([torch.arange(B, 2 * B, device=z.device), torch.arange(B, device=z.device)])
    return nn.functional.cross_entropy(logits, labels)


def export_encoder_weights_lua(encoder: Encoder, out_path: str, t_max: int):
    """Write encoder weights to a Lua file that returns a table. Each layer: weight[out][in], bias[out]."""
    linear_layers = [m for m in encoder.net if isinstance(m, nn.Linear)]
    lines = [
        "-- Encoder weights (auto-generated); input = T_max*90 frame features + 15 per-bone energy (if used).",
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
    ap.add_argument("--epochs", type=int, default=80, help="Training epochs")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--spread_frames", action="store_true", default=True, help="When clip is longer than T_max, sample T_max frames uniformly across duration instead of taking the first T_max")
    ap.add_argument("--workers", type=int, default=8, help="Number of parallel workers for loading CSVs (0 = single-threaded)")
    ap.add_argument("--scheduler", choices=["none", "step", "multistep"], default="step", help="LR scheduler: step (every N epochs), multistep (at milestones), none")
    ap.add_argument("--lr_step", type=int, default=25, help="For step scheduler: reduce LR every this many epochs")
    ap.add_argument("--lr_gamma", type=float, default=0.5, help="Multiply LR by this factor when scheduler steps")
    ap.add_argument("--lr_milestones", type=int, nargs="+", default=[25, 50, 75], help="For multistep scheduler: epochs at which to reduce LR")
    ap.add_argument("--infonce_weight", type=float, default=0.1, help="Weight for InfoNCE contrastive loss (0 to disable)")
    ap.add_argument("--infonce_tau", type=float, default=0.07, help="Temperature for InfoNCE")
    ap.add_argument("--infonce_noise", type=float, default=0.02, help="Std of Gaussian noise for contrastive augmentation")
    args = ap.parse_args()

    data_dir = args.data_dir
    checkpoint_dir = args.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    print("Loading manifest with {} workers".format(args.workers))
    manifest = load_manifest(data_dir)
    X, frame_dim = load_all_data(
        data_dir,
        manifest,
        t_max=args.T_max,
        spread_frames=args.spread_frames,
        num_workers=args.workers,
    )
    N, input_dim = X.shape
    print(f"Loaded {N} clips, input_dim={input_dim} (frame_dim={frame_dim}, energy={ENERGY_DIM})")

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

    model = AutoEncoder(input_dim, frame_dim, args.hidden, args.latent).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    if args.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=args.lr_step, gamma=args.lr_gamma)
    elif args.scheduler == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=args.lr_milestones, gamma=args.lr_gamma)
    else:
        scheduler = None
    mse = nn.MSELoss()
    if args.infonce_weight > 0:
        print(f"InfoNCE: weight={args.infonce_weight}, tau={args.infonce_tau}, noise_std={args.infonce_noise}")

    for ep in range(args.epochs):
        model.train()
        perm_train = torch.randperm(X_train.size(0), device=device)
        total_loss = 0.0
        n_batches = 0
        for i in range(0, X_train.size(0), args.batch_size):
            idx = perm_train[i : i + args.batch_size]
            batch = X_train[idx]
            recon, z = model(batch)
            loss = mse(recon, batch[:, :frame_dim])
            if args.infonce_weight > 0 and batch.size(0) > 1:
                batch_aug = batch + args.infonce_noise * torch.randn_like(batch, device=batch.device)
                z_aug = model.encoder(batch_aug)
                loss = loss + args.infonce_weight * infonce_loss(z, z_aug, tau=args.infonce_tau)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        if scheduler is not None:
            scheduler.step()
        train_loss = total_loss / max(n_batches, 1)
        model.eval()
        with torch.no_grad():
            recon_val, _ = model(X_val)
            val_loss = mse(recon_val, X_val[:, :frame_dim]).item()
        lr_str = f"  lr={opt.param_groups[0]['lr']:.2e}" if scheduler else ""
        if (ep + 1) % 1 == 0 or ep == 0:
            print(f"Epoch {ep + 1}/{args.epochs}  train_loss={train_loss:.6f}  val_loss={val_loss:.6f}{lr_str}")

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
        frame_dim=frame_dim,
        energy_dim=ENERGY_DIM,
        latent_dim=args.latent,
        hidden_dims=np.array(args.hidden),
    )
    export_encoder_weights_lua(model.encoder, os.path.join(checkpoint_dir, "encoder_weights.lua"), args.T_max)
    print(f"Saved checkpoints and encoder_weights.lua to {checkpoint_dir}")


if __name__ == "__main__":
    main()
