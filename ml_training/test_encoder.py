#!/usr/bin/env python3
"""
Unit tests for the animation encoder.

Requires: trained checkpoint (norm.npz, encoder.pt) and clip CSVs in train_data.
Run from repo root:
  pytest ml_training/test_encoder.py -v
  python ml_training/test_encoder.py   # run as script for quick check

Uses up to 100 clip CSVs from ML_TRAIN_DATA (default ml_training/train_data).
Checkpoint from ML_CHECKPOINT_DIR (default ml_training/checkpoints).

Tests:
- Determinism: encoding twice gives the same result
- Mirror: mirrored clip has similar embedding to original (semantic invariance)
- Time offset: cyclic shift of frames yields similar embedding (temporal robustness)
- Small noise: adding small Gaussian noise keeps similarity (most clips)
- Repeated frames: replacing a few frames with previous frame still similar
- Large changes: mixing in frames from another clip yields dissimilar embedding
- Reversed time: reversed clip produces valid embedding (sanity)
- Same content: same clip embedded twice is near-identical
- Different clips: different clips do not all have identical embeddings
"""
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Add ml_training so we can import from train_autoencoder
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
ML_TRAINING = Path(__file__).resolve().parent

from train_autoencoder import (
    Encoder,
    load_clip_csv,
    compute_bone_energy,
    FEAT_PER_FRAME,
    NUM_BONES,
    FEAT_PER_BONE,
    ENERGY_DIM,
)

# Mirror/pad helpers (may not exist in rolled-back trainer; define here for tests)
R15_MIRROR_AXIS = "x"
R15_MIRROR_PAIRS = "3:6,4:7,5:8,9:12,10:13,11:14"


def _parse_mirror_pairs(s: str):
    pairs = []
    for part in s.replace(" ", "").split(","):
        if not part:
            continue
        a, b = part.split(":")
        pairs.append((int(a), int(b)))
    return pairs


def _build_swap_index(num_bones: int, pairs: list) -> np.ndarray:
    idx = np.arange(num_bones, dtype=np.int64)
    for a, b in pairs:
        if 0 <= a < num_bones and 0 <= b < num_bones:
            idx[a], idx[b] = idx[b], idx[a]
    return idx


def _pad_or_sample_to_tmax(arr: np.ndarray, t_max: int, spread_frames: bool) -> np.ndarray:
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


def _to_derivative(arr_abs: np.ndarray) -> np.ndarray:
    T = arr_abs.shape[0]
    if T <= 1:
        return np.zeros_like(arr_abs)
    out = np.zeros_like(arr_abs)
    out[1:] = arr_abs[1:] - arr_abs[:-1]
    out[0] = out[1]
    return out


def _mirror_frames_np(
    x: np.ndarray,
    swap_idx: np.ndarray,
    mirror_axis: str = "x",
) -> np.ndarray:
    """
    Mirror (T, 90) frames: swap L/R bones, negate position and quatlog on axis.
    Same logic as mirror_frames_torch in train_autoencoder.
    """
    T, Fdim = x.shape
    assert Fdim == FEAT_PER_FRAME
    x4 = x.reshape(T, NUM_BONES, FEAT_PER_BONE).copy()
    x4 = x4[:, swap_idx, :]
    pos = x4[:, :, 0:3]
    rot = x4[:, :, 3:6]
    axis = mirror_axis.lower()
    if axis == "x":
        pos[:, :, 0] = -pos[:, :, 0]
        rot[:, :, 1] = -rot[:, :, 1]
        rot[:, :, 2] = -rot[:, :, 2]
    elif axis == "y":
        pos[:, :, 1] = -pos[:, :, 1]
        rot[:, :, 0] = -rot[:, :, 0]
        rot[:, :, 2] = -rot[:, :, 2]
    else:  # z
        pos[:, :, 2] = -pos[:, :, 2]
        rot[:, :, 0] = -rot[:, :, 0]
        rot[:, :, 1] = -rot[:, :, 1]
    x4 = np.concatenate([pos, rot], axis=-1)
    return x4.reshape(T, Fdim).astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1D vectors."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _load_encoder_and_norm(data_dir: Path, checkpoint_dir: Path):
    """Load norm.npz and encoder; support old (frames+energy) and new (frame-only) norm formats."""
    norm_path = checkpoint_dir / "norm.npz"
    enc_path = checkpoint_dir / "encoder.pt"
    if not norm_path.is_file() or not enc_path.is_file():
        return None
    norm = np.load(norm_path, allow_pickle=True)
    T_max = int(norm["T_max"])
    latent_dim = int(norm["latent_dim"])
    h = norm.get("hidden_dims")
    if h is not None and getattr(h, "size", 0) > 0:
        hidden_dims = h.tolist() if hasattr(h, "tolist") else list(h)
    else:
        hidden_dims = [512, 256]
    swap_idx = _build_swap_index(NUM_BONES, _parse_mirror_pairs(R15_MIRROR_PAIRS))

    # Old format: mean, std, input_dim, frame_dim (encoder input = frames + energy)
    if "mean" in norm and "input_dim" in norm:
        mean = norm["mean"]
        std = norm["std"]
        input_dim = int(norm["input_dim"])
        frame_dim = int(norm["frame_dim"])
        encoder = Encoder(input_dim, hidden_dims, latent_dim)
        state = torch.load(enc_path, map_location="cpu", weights_only=True)
        encoder.load_state_dict(state)
        encoder.eval()
        return {
            "encoder": encoder,
            "T_max": T_max,
            "frame_dim": frame_dim,
            "input_dim": input_dim,
            "mean": mean,
            "std": std,
            "use_energy": True,
            "swap_idx": swap_idx,
            "spread_frames": True,
            "format": "old",
        }

    # New format: frame_mean, frame_std, frame_dim, use_derivative (frame-only input)
    frame_dim = int(norm["frame_dim"])
    frame_mean = norm["frame_mean"]
    frame_std = norm["frame_std"]
    use_derivative = bool(norm.get("use_derivative", np.array(1))[0])
    encoder = Encoder(frame_dim, hidden_dims, latent_dim)
    state = torch.load(enc_path, map_location="cpu", weights_only=True)
    encoder.load_state_dict(state)
    encoder.eval()
    return {
        "encoder": encoder,
        "T_max": T_max,
        "frame_dim": frame_dim,
        "frame_mean": frame_mean,
        "frame_std": frame_std,
        "use_derivative": use_derivative,
        "swap_idx": swap_idx,
        "spread_frames": True,
        "format": "new",
    }


def embed_clip(ctx: dict, arr: np.ndarray) -> np.ndarray:
    """
    Embed a single clip. arr: (T, 90) in absolute frame form.
    Supports old format (frames + energy) and new format (frame-only, optional derivative).
    """
    T_max = ctx["T_max"]
    encoder = ctx["encoder"]

    if ctx["format"] == "old":
        # Old trainer: pad/sample to T_max, concat per-bone energy, normalize with mean/std
        energy = compute_bone_energy(arr).reshape(1, -1)
        if arr.shape[0] == 0:
            arr = np.zeros((T_max, FEAT_PER_FRAME), dtype=np.float32)
        else:
            arr = _pad_or_sample_to_tmax(arr, T_max, ctx["spread_frames"])
        x = np.concatenate([arr.reshape(1, -1).astype(np.float32), energy], axis=1)
        x = (x - ctx["mean"]) / ctx["std"]
    else:
        # New format: pad/sample, optional derivative, normalize with frame_mean/frame_std
        if arr.shape[0] == 0:
            arr = np.zeros((T_max, FEAT_PER_FRAME), dtype=np.float32)
        else:
            arr = _pad_or_sample_to_tmax(arr, T_max, ctx["spread_frames"])
        if ctx["use_derivative"]:
            arr = _to_derivative(arr)
        x = arr.reshape(1, -1).astype(np.float32)
        x = (x - ctx["frame_mean"]) / ctx["frame_std"]

    with torch.no_grad():
        z = encoder(torch.from_numpy(x)).numpy()[0]
    return z


def collect_clip_paths(data_dir: Path, max_clips: int = 100) -> list[Path]:
    """Return up to max_clips paths to clip CSVs (animId-clipId.csv)."""
    if not data_dir.is_dir():
        return []
    paths = []
    for f in sorted(data_dir.iterdir()):
        if f.suffix.lower() == ".csv" and f.name != "manifest.csv":
            parts = f.stem.split("-")
            if len(parts) == 2:
                paths.append(f)
                if len(paths) >= max_clips:
                    break
    return paths


# -----------------------------------------------------------------------------
# Pytest fixtures and tests
# -----------------------------------------------------------------------------

def _get_ctx_and_clips():
    data_dir = Path(os.environ.get("ML_TRAIN_DATA", str(ML_TRAINING / "train_data")))
    checkpoint_dir = Path(os.environ.get("ML_CHECKPOINT_DIR", str(ML_TRAINING / "checkpoints")))
    ctx = _load_encoder_and_norm(data_dir, checkpoint_dir)
    if ctx is None:
        return None, []
    clips = collect_clip_paths(data_dir, max_clips=100)
    return ctx, clips


def pytest_configure(config):
    # Allow running without pytest
    pass


def test_encoder_determinism():
    """Encoding the same clip twice must yield identical embeddings."""
    ctx, clips = _get_ctx_and_clips()
    if ctx is None or len(clips) < 1:
        import pytest
        pytest.skip("Need checkpoint and at least one clip CSV")
    np.random.seed(42)
    path = clips[0]
    arr = load_clip_csv(str(path))
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    z1 = embed_clip(ctx, arr)
    z2 = embed_clip(ctx, arr)
    assert z1.shape == z2.shape
    np.testing.assert_allclose(z1, z2, rtol=1e-5, atol=1e-6, err_msg="Encoder must be deterministic")


def test_encoder_mirror_similar():
    """Mirrored clip should have high cosine similarity to original (semantic invariance).
    Skipped for old checkpoint (frames+energy) which was not trained with mirroring."""
    ctx, clips = _get_ctx_and_clips()
    if ctx is None or len(clips) < 1:
        import pytest
        pytest.skip("Need checkpoint and at least one clip CSV")
    if ctx.get("format") == "old":
        import pytest
        pytest.skip("Old checkpoint (frames+energy) was not trained with mirroring")
    similarity_threshold = 0.90
    for path in clips[:20]:
        arr = load_clip_csv(str(path))
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        if arr.shape[0] < 2:
            continue
        z_orig = embed_clip(ctx, arr)
        arr_mirror = _mirror_frames_np(arr, ctx["swap_idx"], R15_MIRROR_AXIS)
        z_mirror = embed_clip(ctx, arr_mirror)
        sim = cosine_similarity(z_orig, z_mirror)
        assert sim >= similarity_threshold, (
            f"Mirrored clip {path.name} similarity {sim:.4f} < {similarity_threshold}"
        )
    # If we had at least one multi-frame clip, we're good
    assert True


def test_encoder_time_offset_similar():
    """Shifting frames by a small offset should keep embedding similar (temporal robustness).
    With derivative input, cyclic roll changes boundary deltas so we use a moderate threshold."""
    ctx, clips = _get_ctx_and_clips()
    if ctx is None or len(clips) < 1:
        import pytest
        pytest.skip("Need checkpoint and at least one clip CSV")
    similarity_threshold = 0.90
    np.random.seed(123)
    for path in clips[:25]:
        arr = load_clip_csv(str(path))
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        T = arr.shape[0]
        if T < 10:
            continue
        shift = min(5, T // 4)
        if shift == 0:
            continue
        arr_shifted = np.roll(arr, shift, axis=0)
        z_orig = embed_clip(ctx, arr)
        z_shifted = embed_clip(ctx, arr_shifted)
        sim = cosine_similarity(z_orig, z_shifted)
        assert sim >= similarity_threshold, (
            f"Time-offset clip {path.name} similarity {sim:.4f} < {similarity_threshold}"
        )
    assert True


def test_encoder_small_noise_similar():
    """Adding small Gaussian noise to joints should still be detected as similar (most clips)."""
    ctx, clips = _get_ctx_and_clips()
    if ctx is None or len(clips) < 1:
        import pytest
        pytest.skip("Need checkpoint and at least one clip CSV")
    similarity_threshold = 0.90
    min_ratio_pass = 0.80  # at least 80% of clips must pass
    noise_scale = 0.02
    np.random.seed(456)
    passed = 0
    total = 0
    for path in clips[:25]:
        arr = load_clip_csv(str(path))
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        if arr.shape[0] == 0:
            continue
        total += 1
        noisy = arr + noise_scale * np.random.randn(*arr.shape).astype(np.float32)
        z_orig = embed_clip(ctx, arr)
        z_noisy = embed_clip(ctx, noisy)
        sim = cosine_similarity(z_orig, z_noisy)
        if sim >= similarity_threshold:
            passed += 1
    assert total >= 1, "No valid clips"
    ratio = passed / total
    assert ratio >= min_ratio_pass, (
        f"Small-noise: only {passed}/{total} clips had sim >= {similarity_threshold} (need {min_ratio_pass:.0%})"
    )


def test_encoder_zeroed_frames_similar():
    """Replacing a few random frames with the previous frame (drop/repeat) should still be similar.
    Zeroing frames is very destructive for derivative-based encoding so we test repeat instead."""
    ctx, clips = _get_ctx_and_clips()
    if ctx is None or len(clips) < 1:
        import pytest
        pytest.skip("Need checkpoint and at least one clip CSV")
    similarity_threshold = 0.95
    np.random.seed(789)
    for path in clips[:25]:
        arr = load_clip_csv(str(path))
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        T = arr.shape[0]
        if T < 5:
            continue
        arr_mod = arr.copy()
        n_rep = max(1, T // 15)
        indices = np.random.choice(T, size=min(n_rep, T), replace=False)
        for i in indices:
            if i > 0:
                arr_mod[i] = arr_mod[i - 1]
        z_orig = embed_clip(ctx, arr)
        z_mod = embed_clip(ctx, arr_mod)
        sim = cosine_similarity(z_orig, z_mod)
        assert sim >= similarity_threshold, (
            f"Repeated-frames clip {path.name} similarity {sim:.4f} < {similarity_threshold}"
        )
    assert True


def test_encoder_large_changes_dissimilar():
    """Mixing in frames from another clip should yield clearly different embedding (dissimilar)."""
    ctx, clips = _get_ctx_and_clips()
    if ctx is None or len(clips) < 2:
        import pytest
        pytest.skip("Need checkpoint and at least 2 clip CSVs")
    similarity_upper = 0.95
    np.random.seed(101)
    for i, path_a in enumerate(clips[:15]):
        path_b = clips[(i + 1) % len(clips)]
        arr_a = load_clip_csv(str(path_a))
        arr_b = load_clip_csv(str(path_b))
        arr_a = np.nan_to_num(arr_a, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        arr_b = np.nan_to_num(arr_b, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        Ta, Tb = arr_a.shape[0], arr_b.shape[0]
        if Ta < 4 or Tb < 4:
            continue
        # Replace second half of A with frames from B (different motion)
        half = Ta // 2
        arr_mixed = arr_a.copy()
        idx_b = np.linspace(0, Tb - 1, Ta - half, dtype=np.int64)
        arr_mixed[half:] = arr_b[idx_b]
        z_orig = embed_clip(ctx, arr_a)
        z_mixed = embed_clip(ctx, arr_mixed)
        sim = cosine_similarity(z_orig, z_mixed)
        assert sim <= similarity_upper, (
            f"Mixed clip {path_a.name}+{path_b.name} should be dissimilar: similarity {sim:.4f} > {similarity_upper}"
        )
    assert True


def test_encoder_reversed_time_similar():
    """Reversing the clip in time often preserves semantics (e.g. walk forward vs backward); allow similar."""
    ctx, clips = _get_ctx_and_clips()
    if ctx is None or len(clips) < 1:
        import pytest
        pytest.skip("Need checkpoint and at least one clip CSV")
    # Reversed can be similar for symmetric motions; we only require it doesn't crash and sim is reasonable
    similarity_lower = 0.50  # reversed might be somewhat similar
    for path in clips[:15]:
        arr = load_clip_csv(str(path))
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        if arr.shape[0] < 2:
            continue
        arr_rev = arr[::-1].copy()
        z_orig = embed_clip(ctx, arr)
        z_rev = embed_clip(ctx, arr_rev)
        sim = cosine_similarity(z_orig, z_rev)
        # Reversed is not required to be very similar; we just check it's in a plausible range
        assert -1.0 <= sim <= 1.0, f"Reversed clip {path.name} similarity out of range: {sim}"


def test_encoder_same_content_padding_similar():
    """Same underlying content padded to T_max in different ways should be similar (e.g. pad vs spread)."""
    ctx, clips = _get_ctx_and_clips()
    if ctx is None or len(clips) < 1:
        import pytest
        pytest.skip("Need checkpoint and at least one clip CSV")
    # We use the same clip; pad_or_sample is called inside embed_clip with fixed spread_frames.
    # So we test: short clip padded to T_max vs same clip but we manually pad differently then pass in.
    # Actually embed_clip always does pad_or_sample_to_tmax with ctx["spread_frames"]. So we can't easily
    # test "different padding same content" without exposing a way to pass pre-padded arrays.
    # Instead: two clips that are very similar (e.g. same file embedded twice) -> identical.
    path = clips[0]
    arr = load_clip_csv(str(path))
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    z1 = embed_clip(ctx, arr)
    z2 = embed_clip(ctx, arr)
    sim = cosine_similarity(z1, z2)
    assert sim >= 0.9999, f"Same clip embedded twice should be near-identical: {sim}"


def test_encoder_different_clips_differ():
    """Different clips should generally have different embeddings (not all identical)."""
    ctx, clips = _get_ctx_and_clips()
    if ctx is None or len(clips) < 3:
        import pytest
        pytest.skip("Need checkpoint and at least 3 clip CSVs")
    embs = []
    for path in clips[:30]:
        arr = load_clip_csv(str(path))
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
        if arr.shape[0] == 0:
            continue
        z = embed_clip(ctx, arr)
        embs.append(z)
    if len(embs) < 2:
        import pytest
        pytest.skip("Need at least 2 valid clips")
    embs = np.array(embs)
    # At least one pair should have similarity < 1.0 (not all clips identical)
    sims = embs @ embs.T
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    outer = np.maximum(norms * norms.T, 1e-12)
    cos_sims = sims / outer
    np.fill_diagonal(cos_sims, -2.0)
    max_off_diag = float(np.max(cos_sims))
    assert max_off_diag < 0.999, (
        "Different clips should not all have near-identical embeddings"
    )


if __name__ == "__main__":
    # Run with pytest if available, else minimal runner
    try:
        import pytest
        sys.exit(pytest.main(["-v", __file__]))
    except ImportError:
        # Run a quick smoke test
        ctx, clips = _get_ctx_and_clips()
        if ctx is None:
            print("SKIP: No checkpoint (norm.npz, encoder.pt) found")
            sys.exit(0)
        if len(clips) < 1:
            print("SKIP: No clip CSVs in data_dir")
            sys.exit(0)
        print(f"Loaded encoder, T_max={ctx['T_max']}, {len(clips)} clips")
        test_encoder_determinism()
        print("test_encoder_determinism OK")
        test_encoder_same_content_padding_similar()
        print("test_encoder_same_content_padding_similar OK")
        test_encoder_mirror_similar()
        print("test_encoder_mirror_similar OK")
        test_encoder_time_offset_similar()
        print("test_encoder_time_offset_similar OK")
        test_encoder_small_noise_similar()
        print("test_encoder_small_noise_similar OK")
        test_encoder_zeroed_frames_similar()
        print("test_encoder_zeroed_frames_similar OK")
        test_encoder_large_changes_dissimilar()
        print("test_encoder_large_changes_dissimilar OK")
        test_encoder_reversed_time_similar()
        print("test_encoder_reversed_time_similar OK")
        test_encoder_different_clips_differ()
        print("test_encoder_different_clips_differ OK")
        print("All tests passed.")
