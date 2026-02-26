#!/usr/bin/env python3
"""
Train an MLP auto-encoder on animation pose sequences.
- Train encoder on FRAME FEATURES ONLY (no energy concatenated into encoder input).
- Use DERIVATIVE (velocity) input by default: Δpos + Δquatlog (same 90 dims per frame).
- Contrastive loss uses SEMANTIC augmentations: time-shift and optional mirroring (no "noise on whole vector").
- Contrastive uses a PROJECTION HEAD (SimCLR style): InfoNCE is applied to proj(z), not z.
- Mean/std normalization computed on TRAIN ONLY (no val leakage).
- Energy is still computed and exported (for retrieval-time concatenation / auxiliary use), but not fed to encoder.

Notes:
- Mirroring depends on your coordinate system and bone left/right mapping.
  This script provides:
    * axis mirror convention for pos/quatlog (configurable via --mirror_axis)
    * a user-specified left/right bone swap map via --mirror_pairs.
  You MUST set mirror_pairs correctly for your skeleton if you enable mirroring.
"""

import argparse
import csv
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Defaults (R15: 15 bones, same order as extract_training_data.lua / anim_fingerprint.lua)
FEAT_PER_FRAME = 90  # 15 bones × 6 (pos xyz + quatlog xyz)
NUM_BONES = 15
FEAT_PER_BONE = 6
ENERGY_DIM = 15

# R15 mirroring: mirror across X (left-right flip in Roblox coords). Bone pairs to swap (0-based).
# Order: LowerTorso(0), UpperTorso(1), Head(2), L_arm(3,4,5), R_arm(6,7,8), L_leg(9,10,11), R_leg(12,13,14)
R15_MIRROR_AXIS = "x"
R15_MIRROR_PAIRS = "3:6,4:7,5:8,9:12,10:13,11:14"  # LeftArm<->RightArm, LeftLeg<->RightLeg
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
        next(reader, None)
        for row in reader:
            if len(row) >= 4:
                nf = max(0, int(_safe_float(row[3])))
                rows.append((row[0], row[1], _safe_float(row[2]), nf))
    return rows


def load_clip_csv(path: str) -> np.ndarray:
    """Load one clip CSV -> (T, 90). Replaces NaN/inf with 0. Uses numpy for fast parsing."""
    try:
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

    # Ensure exactly FEAT_PER_FRAME columns
    if arr.shape[1] < FEAT_PER_FRAME:
        pad = np.zeros((arr.shape[0], FEAT_PER_FRAME - arr.shape[1]), dtype=np.float32)
        arr = np.hstack([arr, pad])
    elif arr.shape[1] > FEAT_PER_FRAME:
        arr = arr[:, :FEAT_PER_FRAME]

    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0, copy=False)


def compute_bone_energy(arr_abs: np.ndarray) -> np.ndarray:
    """
    Per-bone energy from ABSOLUTE frames (pre-derivative):
    sum_t ||Δpos|| + ||Δquatlog|| per bone.
    arr_abs: (T, 90) -> (15,)
    """
    T = arr_abs.shape[0]
    out = np.zeros(ENERGY_DIM, dtype=np.float32)
    if T < 2:
        return out
    for b in range(NUM_BONES):
        pos = arr_abs[:, b * 6 : b * 6 + 3]
        quatlog = arr_abs[:, b * 6 + 3 : b * 6 + 6]
        dpos = np.diff(pos, axis=0)
        dquat = np.diff(quatlog, axis=0)
        out[b] = np.sqrt((dpos ** 2).sum(axis=1)).sum() + np.sqrt((dquat ** 2).sum(axis=1)).sum()
    return out


def to_derivative(arr_abs: np.ndarray) -> np.ndarray:
    """
    Convert absolute frames (T,90) -> derivative (velocity-like) frames (T,90).
    Padding: first frame repeats the first delta (stable).
    """
    T = arr_abs.shape[0]
    if T == 0:
        return arr_abs
    out = np.zeros_like(arr_abs)
    if T == 1:
        return out  # all zeros
    out[1:] = arr_abs[1:] - arr_abs[:-1]
    out[0] = out[1]
    return out


def pad_or_sample_to_tmax(arr: np.ndarray, t_max: int, spread_frames: bool) -> np.ndarray:
    """Pad/trim/sample to (t_max, 90) using last-frame pad or uniform sampling."""
    T = arr.shape[0]
    if T == 0:
        return np.zeros((t_max, FEAT_PER_FRAME), dtype=np.float32)
    if T < t_max:
        pad = np.repeat(arr[-1:], t_max - T, axis=0)
        return np.concatenate([arr, pad], axis=0).astype(np.float32, copy=False)
    if T > t_max:
        if spread_frames:
            idx = np.linspace(0, T - 1, t_max, dtype=np.int64)
            return arr[idx].astype(np.float32, copy=False)
        return arr[:t_max].astype(np.float32, copy=False)
    return arr.astype(np.float32, copy=False)


def _load_one_clip(args_tuple):
    """
    Worker:
      (data_dir, item, t_max, spread_frames, use_derivative)
    Returns:
      frames_in: (t_max,90)  [abs or derivative depending on flag]
      energy:    (15,)       [computed from abs, pre-derivative]
    """
    data_dir, item, t_max, spread_frames, use_derivative = args_tuple
    anim_id, clip_id = item[0], item[1]
    path = os.path.join(data_dir, f"{anim_id}-{clip_id}.csv")
    if not os.path.isfile(path):
        return None

    arr_abs = load_clip_csv(path)
    arr_abs = np.nan_to_num(arr_abs, nan=0.0, posinf=0.0, neginf=0.0, copy=False)

    energy = compute_bone_energy(arr_abs)

    # Pad/sample in ABS space first (so energy corresponds to original clip, but you can also
    # choose to compute energy after resampling if desired).
    arr_abs = pad_or_sample_to_tmax(arr_abs, t_max, spread_frames)

    if use_derivative:
        arr_in = to_derivative(arr_abs)
    else:
        arr_in = arr_abs

    # Guarantee shape
    if arr_in.shape != (t_max, FEAT_PER_FRAME):
        out = np.zeros((t_max, FEAT_PER_FRAME), dtype=np.float32)
        r = min(arr_in.shape[0], t_max)
        c = min(arr_in.shape[1], FEAT_PER_FRAME)
        out[:r, :c] = arr_in[:r, :c]
        if r < t_max and r > 0:
            out[r:] = np.repeat(out[r - 1 : r], t_max - r, axis=0)
        arr_in = out

    return (arr_in.astype(np.float32, copy=False), energy.astype(np.float32, copy=False))


def load_all_data(
    data_dir: str,
    manifest: list,
    t_max: int,
    spread_frames: bool = True,
    num_workers: int = 0,
    use_derivative: bool = True,
):
    """
    Load all clips:
      frames_in: (N, t_max, 90)  [abs or derivative]
      energy:    (N, 15)         [from abs]
    """
    if not manifest:
        manifest = []
        for f in os.listdir(data_dir):
            if f.endswith(".csv") and f != "manifest.csv":
                m = f[:-4].split("-")
                if len(m) == 2:
                    manifest.append((m[0], m[1], 0.0, 0))

    work = [(data_dir, item, t_max, spread_frames, use_derivative) for item in manifest]

    frame_list = []
    energy_list = []

    if num_workers and num_workers > 0:
        with ThreadPoolExecutor(max_workers=num_workers) as ex:
            for result in ex.map(_load_one_clip, work):
                if result is not None:
                    frame_list.append(result[0])
                    energy_list.append(result[1])
    else:
        for w in work:
            result = _load_one_clip(w)
            if result is not None:
                frame_list.append(result[0])
                energy_list.append(result[1])

    if not frame_list:
        raise SystemExit("No clips loaded. Run your extraction first / check data_dir.")

    X_frames = np.stack(frame_list, axis=0)  # (N,T,90)
    X_energy = np.stack(energy_list, axis=0)  # (N,15)
    return X_frames, X_energy


# -----------------------------
# Mirroring utilities
# -----------------------------
def parse_mirror_pairs(s: str) -> List[Tuple[int, int]]:
    """
    Parse --mirror_pairs like: "1:2,3:4,5:6"
    Indices are 0-based (recommended).
    """
    pairs: List[Tuple[int, int]] = []
    s = s.strip()
    if not s:
        return pairs
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        a, b = part.split(":")
        pairs.append((int(a), int(b)))
    return pairs


def build_swap_index(num_bones: int, pairs: Sequence[Tuple[int, int]]) -> np.ndarray:
    """
    Build an index array idx where bone i maps to idx[i] after swapping.
    """
    idx = np.arange(num_bones, dtype=np.int64)
    for a, b in pairs:
        if a < 0 or a >= num_bones or b < 0 or b >= num_bones:
            raise ValueError(f"mirror pair out of range: {a}:{b} for num_bones={num_bones}")
        idx[a], idx[b] = idx[b], idx[a]
    return idx


def mirror_frames_torch(
    x: torch.Tensor,
    swap_idx: Optional[torch.Tensor],
    mirror_axis: str = "x",
) -> torch.Tensor:
    """
    Mirror frames:
      x: (B,T,90) layout [bone0(pos3,rotlog3), bone1(...), ...]
    Steps:
      1) swap left/right bones (optional)
      2) mirror position across axis: pos[axis] *= -1
      3) mirror quatlog rotation vector as an axial vector under reflection:
         w' = det(M) * M * w

    For mirror across X:
      M = diag(-1,1,1), det(M)=-1 => w' = -M w = diag(1,-1,-1) w
      i.e. rotlog.x same, rotlog.y/z negated.

    You MUST ensure your axis choice matches how your data is expressed.
    """
    B, T, Fdim = x.shape
    assert Fdim == FEAT_PER_FRAME

    x4 = x.view(B, T, NUM_BONES, FEAT_PER_BONE)

    # 1) bone swap
    if swap_idx is not None:
        x4 = x4[:, :, swap_idx, :]

    pos = x4[..., 0:3]
    rot = x4[..., 3:6]

    axis = mirror_axis.lower()
    if axis not in ("x", "y", "z"):
        raise ValueError("--mirror_axis must be one of x,y,z")

    # 2) mirror position vector (polar vector)
    if axis == "x":
        pos = pos.clone()
        pos[..., 0] = -pos[..., 0]
        # 3) mirror rotation vector (axial vector): w' = det(M) * M w
        # for M=diag(-1,1,1) => diag(1,-1,-1)
        rot = rot.clone()
        rot[..., 1] = -rot[..., 1]
        rot[..., 2] = -rot[..., 2]
    elif axis == "y":
        pos = pos.clone()
        pos[..., 1] = -pos[..., 1]
        # M=diag(1,-1,1), det=-1 => w' = -M w = diag(-1,1,-1)
        rot = rot.clone()
        rot[..., 0] = -rot[..., 0]
        rot[..., 2] = -rot[..., 2]
    else:  # "z"
        pos = pos.clone()
        pos[..., 2] = -pos[..., 2]
        # M=diag(1,1,-1), det=-1 => w' = -M w = diag(-1,-1,1)
        rot = rot.clone()
        rot[..., 0] = -rot[..., 0]
        rot[..., 1] = -rot[..., 1]

    x4 = torch.cat([pos, rot], dim=-1)
    return x4.view(B, T, Fdim)


def time_shift_torch(x: torch.Tensor, max_shift: int, cyclic: bool) -> torch.Tensor:
    """
    Random time shift per sample.
    x: (B,T,90)
    If cyclic: torch.roll
    else: shift with edge padding (replicate) to avoid wrap-around.
    """
    if max_shift <= 0:
        return x
    B, T, F = x.shape
    shifts = torch.randint(-max_shift, max_shift + 1, (B,), device=x.device)

    if cyclic:
        out = torch.empty_like(x)
        for b in range(B):
            out[b] = torch.roll(x[b], int(shifts[b].item()), dims=0)
        return out

    # non-cyclic: shift with replicate padding
    out = torch.empty_like(x)
    for b in range(B):
        s = int(shifts[b].item())
        if s == 0:
            out[b] = x[b]
        elif s > 0:
            # shift forward: pad first frame
            out[b, :s] = x[b, :1].expand(s, F)
            out[b, s:] = x[b, : T - s]
        else:
            s = -s
            # shift backward: pad last frame
            out[b, T - s :] = x[b, -1:].expand(s, F)
            out[b, : T - s] = x[b, s:]
    return out


# -----------------------------
# Model: AE + projector
# -----------------------------
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

        # projection head for contrastive
        proj_dim = latent_dim  # you can change to smaller like 64
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, proj_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon, z

    def proj(self, z):
        return self.projector(z)


def infonce_loss_simclr(p1: torch.Tensor, p2: torch.Tensor, tau: float = 0.07) -> torch.Tensor:
    """
    SimCLR / NT-Xent over 2B samples.
    p1,p2: (B,D)
    """
    p1 = F.normalize(p1, dim=1, eps=1e-8)
    p2 = F.normalize(p2, dim=1, eps=1e-8)

    B = p1.size(0)
    feats = torch.cat([p1, p2], dim=0)  # (2B,D)
    logits = (feats @ feats.t()) / tau  # (2B,2B)

    mask = torch.eye(2 * B, device=logits.device, dtype=torch.bool)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    logits = logits - logits.max(dim=1, keepdim=True).values

    labels = torch.cat(
        [torch.arange(B, 2 * B, device=logits.device), torch.arange(0, B, device=logits.device)],
        dim=0,
    )
    return F.cross_entropy(logits, labels)


def export_encoder_weights_lua(encoder: Encoder, out_path: str, t_max: int):
    """Write encoder weights to a Lua file that returns a table. Each layer: weight[out][in], bias[out]."""
    linear_layers = [m for m in encoder.net if isinstance(m, nn.Linear)]
    lines = [
        "-- Encoder weights (auto-generated).",
        "-- Input = flattened frames: T_max*90 (derivative or absolute depending on training/inference choice).",
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


@dataclass
class TrainConfig:
    batch_size: int
    lr: float
    epochs: int
    val_ratio: float
    spread_frames: bool
    workers: int
    scheduler: str
    lr_step: int
    lr_gamma: float
    lr_milestones: List[int]

    infonce_weight: float
    infonce_tau: float

    # augment
    use_mirror: bool
    mirror_axis: str
    mirror_pairs: List[Tuple[int, int]]
    mirror_prob: float

    time_shift: int
    time_shift_cyclic: bool


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_dir", default="ml_training/train_data")
    ap.add_argument("--checkpoint_dir", default="ml_training/checkpoints")
    ap.add_argument("--T_max", type=int, default=T_MAX)
    ap.add_argument("--hidden", type=int, nargs="+", default=HIDDEN_DIMS)
    ap.add_argument("--latent", type=int, default=LATENT_DIM)

    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--spread_frames", action="store_true", default=True)
    ap.add_argument("--workers", type=int, default=8)

    ap.add_argument("--scheduler", choices=["none", "step", "multistep"], default="step")
    ap.add_argument("--lr_step", type=int, default=25)
    ap.add_argument("--lr_gamma", type=float, default=0.5)
    ap.add_argument("--lr_milestones", type=int, nargs="+", default=[25, 50, 75])

    ap.add_argument("--infonce_weight", type=float, default=0.2)
    ap.add_argument("--infonce_tau", type=float, default=0.07)

    # Input choice
    ap.add_argument("--use_derivative", action="store_true", default=True, help="Train on Δpos/Δquatlog instead of absolute.")

    # Augmentations
    ap.add_argument("--time_shift", type=int, default=8, help="Max random frame shift (±). 0 disables.")
    ap.add_argument("--time_shift_cyclic", action="store_true", default=True, help="Use cyclic roll for time shift (loop-like).")

    ap.add_argument("--use_mirror", action="store_true", default=True, help="Enable mirroring augmentation.")
    ap.add_argument("--mirror_axis", choices=["x", "y", "z"], default=R15_MIRROR_AXIS, help="Axis to reflect (R15: x = left-right).")
    ap.add_argument("--mirror_pairs", type=str, default=R15_MIRROR_PAIRS, help='Bone swap pairs "a:b,c:d" (0-based). R15 default: L/R arm and leg.')
    ap.add_argument("--mirror_prob", type=float, default=0.5, help="Probability to apply mirror to a view.")

    args = ap.parse_args()

    cfg = TrainConfig(
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        val_ratio=args.val_ratio,
        spread_frames=args.spread_frames,
        workers=args.workers,
        scheduler=args.scheduler,
        lr_step=args.lr_step,
        lr_gamma=args.lr_gamma,
        lr_milestones=list(args.lr_milestones),

        infonce_weight=args.infonce_weight,
        infonce_tau=args.infonce_tau,

        use_mirror=args.use_mirror,
        mirror_axis=args.mirror_axis,
        mirror_pairs=parse_mirror_pairs(args.mirror_pairs),
        mirror_prob=args.mirror_prob,

        time_shift=args.time_shift,
        time_shift_cyclic=args.time_shift_cyclic,
    )

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    print(f"Loading manifest with {cfg.workers} workers...")
    manifest = load_manifest(args.data_dir)

    X_frames, X_energy = load_all_data(
        args.data_dir,
        manifest,
        t_max=args.T_max,
        spread_frames=cfg.spread_frames,
        num_workers=cfg.workers,
        use_derivative=args.use_derivative,
    )
    N = X_frames.shape[0]
    frame_dim = args.T_max * FEAT_PER_FRAME
    print(f"Loaded {N} clips")
    print(f"Frames: {X_frames.shape}  Energy: {X_energy.shape}  frame_dim={frame_dim}")
    print(f"Training on {'DERIVATIVE' if args.use_derivative else 'ABSOLUTE'} frames")

    # Train/val split
    perm = np.random.default_rng(42).permutation(N)
    n_val = int(N * cfg.val_ratio)
    val_idx = perm[:n_val]
    train_idx = perm[n_val:]

    Xf_train = X_frames[train_idx].reshape(-1, frame_dim).astype(np.float32, copy=False)
    Xf_val = X_frames[val_idx].reshape(-1, frame_dim).astype(np.float32, copy=False)

    # Normalize using TRAIN only (no leakage)
    mean = Xf_train.mean(axis=0, keepdims=True)
    std = Xf_train.std(axis=0, keepdims=True) + 1e-6

    Xf_train = (Xf_train - mean) / std
    Xf_val = (Xf_val - mean) / std

    # Keep energy around for export / retrieval usage; normalize separately for convenience
    e_mean = X_energy[train_idx].mean(axis=0, keepdims=True).astype(np.float32)
    e_std = X_energy[train_idx].std(axis=0, keepdims=True).astype(np.float32) + 1e-6

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X_train = torch.from_numpy(Xf_train).to(device)
    X_val = torch.from_numpy(Xf_val).to(device)

    model = AutoEncoder(input_dim=frame_dim, hidden_dims=args.hidden, latent_dim=args.latent).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)

    if cfg.scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=cfg.lr_step, gamma=cfg.lr_gamma)
    elif cfg.scheduler == "multistep":
        scheduler = torch.optim.lr_scheduler.MultiStepLR(opt, milestones=cfg.lr_milestones, gamma=cfg.lr_gamma)
    else:
        scheduler = None

    print(f"InfoNCE: weight={cfg.infonce_weight}, tau={cfg.infonce_tau}")
    print(f"Augment: time_shift={cfg.time_shift} (cyclic={cfg.time_shift_cyclic})")
    print(f"Augment: mirror={cfg.use_mirror} axis={cfg.mirror_axis} prob={cfg.mirror_prob} pairs={cfg.mirror_pairs}")

    swap_idx_t = None
    if cfg.use_mirror:
        if len(cfg.mirror_pairs) == 0:
            print("WARNING: --use_mirror is set but --mirror_pairs is empty. "
                  "No left/right bone swapping will occur; mirroring may be semantically incorrect.")
        swap_idx_np = build_swap_index(NUM_BONES, cfg.mirror_pairs)
        swap_idx_t = torch.from_numpy(swap_idx_np).to(device)

    mse = nn.MSELoss()

    def make_two_views(batch_flat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        batch_flat: (B, frame_dim) normalized
        returns v1_flat, v2_flat: (B, frame_dim) normalized
        """
        B = batch_flat.size(0)
        x = batch_flat.view(B, args.T_max, FEAT_PER_FRAME)

        v1 = x
        v2 = x

        # time shift
        if cfg.time_shift > 0:
            v1 = time_shift_torch(v1, cfg.time_shift, cfg.time_shift_cyclic)
            v2 = time_shift_torch(v2, cfg.time_shift, cfg.time_shift_cyclic)

        # mirror (independently per view)
        if cfg.use_mirror:
            if cfg.mirror_prob > 0:
                m1 = (torch.rand((B,), device=device) < cfg.mirror_prob)
                m2 = (torch.rand((B,), device=device) < cfg.mirror_prob)
                if m1.any():
                    v1m = mirror_frames_torch(v1[m1], swap_idx_t, cfg.mirror_axis)
                    v1 = v1.clone()
                    v1[m1] = v1m
                if m2.any():
                    v2m = mirror_frames_torch(v2[m2], swap_idx_t, cfg.mirror_axis)
                    v2 = v2.clone()
                    v2[m2] = v2m

        return v1.view(B, -1), v2.view(B, -1)

    for ep in range(cfg.epochs):
        model.train()
        perm_train = torch.randperm(X_train.size(0), device=device)

        total_loss = 0.0
        total_recon = 0.0
        total_nce = 0.0
        n_batches = 0

        for i in range(0, X_train.size(0), cfg.batch_size):
            idx = perm_train[i : i + cfg.batch_size]
            batch = X_train[idx]
            if batch.size(0) < 2:
                continue

            # Reconstruction on original batch
            recon, z = model(batch)
            loss_recon = mse(recon, batch)

            loss = loss_recon

            # Contrastive on augmented views
            if cfg.infonce_weight > 0:
                v1, v2 = make_two_views(batch)
                z1 = model.encoder(v1)
                z2 = model.encoder(v2)
                p1 = model.proj(z1)
                p2 = model.proj(z2)
                loss_nce = infonce_loss_simclr(p1, p2, tau=cfg.infonce_tau)
                loss = loss + cfg.infonce_weight * loss_nce
            else:
                loss_nce = torch.zeros((), device=device)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total_loss += float(loss.detach())
            total_recon += float(loss_recon.detach())
            total_nce += float(loss_nce.detach())
            n_batches += 1

        if scheduler is not None:
            scheduler.step()

        train_loss = total_loss / max(n_batches, 1)
        train_recon = total_recon / max(n_batches, 1)
        train_nce = total_nce / max(n_batches, 1)

        model.eval()
        with torch.no_grad():
            recon_val, _ = model(X_val)
            val_loss = float(mse(recon_val, X_val).detach())

        lr_str = f"  lr={opt.param_groups[0]['lr']:.2e}" if scheduler else ""
        print(
            f"Epoch {ep + 1}/{cfg.epochs}  "
            f"train={train_loss:.6f} (recon={train_recon:.6f}, nce={train_nce:.6f})  "
            f"val_recon={val_loss:.6f}{lr_str}"
        )

    # Save checkpoints
    ckpt_dir = args.checkpoint_dir
    torch.save(model.state_dict(), os.path.join(ckpt_dir, "ae_contrastive.pt"))
    torch.save(model.encoder.state_dict(), os.path.join(ckpt_dir, "encoder.pt"))

    # Export normalization + metadata
    # NOTE: These mean/std apply to the FRAME INPUT (flattened). Energy is separate.
    np.savez(
        os.path.join(ckpt_dir, "norm.npz"),
        frame_mean=mean.astype(np.float32),
        frame_std=std.astype(np.float32),
        energy_mean=e_mean,
        energy_std=e_std,
        T_max=args.T_max,
        frame_dim=frame_dim,
        feat_per_frame=FEAT_PER_FRAME,
        num_bones=NUM_BONES,
        latent_dim=args.latent,
        hidden_dims=np.array(args.hidden, dtype=np.int32),
        use_derivative=np.array([1 if args.use_derivative else 0], dtype=np.int32),
        mirror_axis=np.array([ord(cfg.mirror_axis)], dtype=np.int32),
    )

    # Export encoder weights to Lua
    model_cpu = model.cpu()
    export_encoder_weights_lua(model_cpu.encoder, os.path.join(ckpt_dir, "encoder_weights.lua"), args.T_max)

    # Also save energy for your corpus (optional but useful for retrieval concatenation)
    # Align with original ordering in X_frames/X_energy.
    np.save(os.path.join(ckpt_dir, "energy.npy"), X_energy.astype(np.float32))

    print(f"Saved: {ckpt_dir}/ae_contrastive.pt, encoder.pt, norm.npz, encoder_weights.lua, energy.npy")


if __name__ == "__main__":
    main()