#!/usr/bin/env python3
"""
Train an MLP auto-encoder on pose sequences (T_max x 90 per clip).
Export PyTorch checkpoint and encoder weights for Lua (encoder_weights.lua).
"""
import argparse
import os

import numpy as np
import torch
import torch.nn as nn

# Defaults (must match Lua inference)
FEAT_PER_FRAME = 90  # 15 bones × 6 (pos xyz + quatlog xyz)
FEAT_PER_BONE = 6
NUM_BONES = 15
ENERGY_DIM = 15  # accumulated delta pos + quatlog per bone
T_MAX = 90
HIDDEN_DIMS = [512, 256]
LATENT_DIM = 128


def load_packed_data(packed_path: str):
    """Load from a .npz written by pack_training_data.py. Returns (X, frame_dim)."""
    data = np.load(packed_path, allow_pickle=True)
    frames = data["frames"]
    energy = data["energy"]
    N = frames.shape[0]
    t_max = int(data["T_max"])
    frame_dim = t_max * FEAT_PER_FRAME
    X = np.concatenate([frames.reshape(N, -1), energy], axis=1)
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
    ap.add_argument("--packed", default="ml_training/train_data.npz", help="Path to .npz from pack_training_data.py")
    ap.add_argument("--checkpoint_dir", default="ml_training/checkpoints", help="Where to save .pt and encoder_weights.lua")
    ap.add_argument("--hidden", type=int, nargs="+", default=HIDDEN_DIMS, help="Hidden layer sizes")
    ap.add_argument("--latent", type=int, default=LATENT_DIM, help="Latent dimension")
    ap.add_argument("--epochs", type=int, default=80, help="Training epochs")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--scheduler", choices=["none", "step", "multistep"], default="step", help="LR scheduler: step (every N epochs), multistep (at milestones), none")
    ap.add_argument("--lr_step", type=int, default=25, help="For step scheduler: reduce LR every this many epochs")
    ap.add_argument("--lr_gamma", type=float, default=0.5, help="Multiply LR by this factor when scheduler steps")
    ap.add_argument("--lr_milestones", type=int, nargs="+", default=[25, 50, 75], help="For multistep scheduler: epochs at which to reduce LR")
    ap.add_argument("--infonce_weight", type=float, default=0.1, help="Weight for InfoNCE contrastive loss (0 to disable)")
    ap.add_argument("--infonce_tau", type=float, default=0.07, help="Temperature for InfoNCE")
    ap.add_argument("--infonce_noise", type=float, default=0.02, help="Std of Gaussian noise for contrastive augmentation")
    args = ap.parse_args()

    checkpoint_dir = args.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)

    if not os.path.isfile(args.packed):
        raise SystemExit(f"Packed file not found: {args.packed}. Run pack_training_data.py first.")
    print(f"Loading packed data from {args.packed}")
    X, frame_dim = load_packed_data(args.packed)
    data = np.load(args.packed, allow_pickle=True)
    args.T_max = int(data["T_max"])
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

        if (ep + 1) % 10 == 0 or ep + 1 == args.epochs:
            with torch.no_grad():
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

                model = model.to(device)


if __name__ == "__main__":
    main()
