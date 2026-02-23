#!/usr/bin/env python3
"""
Diagnose why animations in a group (e.g. group 0) are grouped together.

Loads groups.csv and fingerprints.csv, then for the specified group:
- Lists anim IDs and duration spread (min/max/mean).
- Computes pairwise cosine similarity for a sample of pairs.
- Reports min/max/mean cosine within the group and how many pairs are above threshold.
- If many pairs are BELOW threshold, the group is likely due to transitivity
  (A~B, B~C, C~D => A,B,C,D in one component but A and D not similar).

Usage:
  python diagnose_group.py [group_id]
  Default group_id is 0.
"""

import csv
import random
import sys
from collections import defaultdict

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

GROUPS_CSV = "groups.csv"
FINGERPRINTS_CSV = "fingerprints.csv"
THRESHOLD = 0.99
SAMPLE_PAIRS = 1000  # number of random pairs to sample for large groups
CHAIN_LENGTH = 20    # check similarity along chain: 0-1, 1-2, ... (consecutive in sorted order)


def load_groups(path: str):
    """anim_id -> group_id"""
    anim_to_group = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            if len(row) >= 2:
                anim_to_group[row[0]] = int(row[1])
    return anim_to_group


def load_fingerprints(path: str):
    """anim_id -> embedding (list of 128 floats), anim_id -> duration, anim_id -> clip_id"""
    embeddings = {}
    durations = {}
    clip_ids = {}
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            if len(row) < 4 + 128:
                continue
            aid = row[0]
            clip_ids[aid] = row[1]
            try:
                durations[aid] = float(row[2])
            except ValueError:
                durations[aid] = 0.0
            emb = [float(x) for x in row[4 : 4 + 128]]
            embeddings[aid] = emb
    return embeddings, durations, clip_ids


def cosine_sim(a, b) -> float:
    n = min(len(a), len(b))
    sa = sum(a[i] * b[i] for i in range(n))
    na = sum(a[i] * a[i] for i in range(n)) ** 0.5
    nb = sum(b[i] * b[i] for i in range(n)) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return float(sa / (na * nb))


def main():
    group_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    print(f"Loading {GROUPS_CSV} and {FINGERPRINTS_CSV} ...")
    anim_to_group = load_groups(GROUPS_CSV)
    embeddings, durations, clip_ids = load_fingerprints(FINGERPRINTS_CSV)

    group_anim_ids = [aid for aid, gid in anim_to_group.items() if gid == group_id]
    group_anim_ids.sort(key=lambda x: (len(x), x))
    n = len(group_anim_ids)
    print(f"Group {group_id} has {n} animations.")

    # Filter to those we have embeddings for
    in_group = [aid for aid in group_anim_ids if aid in embeddings]
    missing = n - len(in_group)
    if missing:
        print(f"  ({missing} missing from fingerprints.csv)")
    if not in_group:
        print("No embeddings for this group. Exiting.")
        return
    n = len(in_group)

    # Duration spread
    durs = [durations[aid] for aid in in_group]
    print(f"Duration: min={min(durs):.2f}s max={max(durs):.2f}s mean={sum(durs)/len(durs):.2f}s")
    if max(durs) - min(durs) > 5:
        print("  -> Large duration spread suggests many unrelated animations in one group (transitivity or weak fingerprint).")

    # Pairwise cosine: sample pairs
    rng = random.Random(42)
    if n * (n - 1) // 2 <= SAMPLE_PAIRS:
        pairs = [(in_group[i], in_group[j]) for i in range(n) for j in range(i + 1, n)]
    else:
        indices = list(range(n))
        pairs = []
        seen = set()
        for _ in range(SAMPLE_PAIRS):
            i, j = rng.sample(indices, 2)
            if i > j:
                i, j = j, i
            if i != j and (i, j) not in seen:
                seen.add((i, j))
                pairs.append((in_group[i], in_group[j]))
        if len(pairs) < SAMPLE_PAIRS:
            # fallback: add all pairs if sampling gave duplicates
            pairs = [(in_group[i], in_group[j]) for i in range(n) for j in range(i + 1, min(i + 1 + 50, n))][:SAMPLE_PAIRS]

    sims = []
    for a, b in pairs:
        sim = cosine_sim(embeddings[a], embeddings[b])
        sims.append(sim)
    above = sum(1 for s in sims if s >= THRESHOLD)
    sorted_sims = sorted(sims)
    mean_sim = sum(sims) / len(sims)
    mid = len(sims) // 2
    median_sim = (sorted_sims[mid] + sorted_sims[mid - 1]) / 2.0 if len(sims) % 2 == 0 else sorted_sims[mid]
    print(f"\nPairwise cosine (sample of {len(sims)} pairs):")
    print(f"  min={min(sims):.6f} max={max(sims):.6f} mean={mean_sim:.6f} median={median_sim:.6f}")
    print(f"  Pairs with cosine >= {THRESHOLD}: {above} / {len(sims)} ({100.0 * above / len(sims):.1f}%)")
    if above < len(sims) * 0.5:
        print("  -> Most pairs are BELOW threshold: group is likely from transitivity (long chain of pairwise-similar anims).")

    # Consecutive chain (first 20 by anim id order)
    sorted_ids = sorted(in_group, key=lambda x: (len(x), x))[: CHAIN_LENGTH + 1]
    chain_sims = []
    for i in range(len(sorted_ids) - 1):
        s = cosine_sim(embeddings[sorted_ids[i]], embeddings[sorted_ids[i + 1]])
        chain_sims.append(s)
    if chain_sims:
        print(f"\nChain similarity (consecutive anim IDs, first {len(chain_sims)} links):")
        print(f"  min={min(chain_sims):.6f} max={max(chain_sims):.6f} mean={sum(chain_sims)/len(chain_sims):.6f}")
        if any(s < THRESHOLD for s in chain_sims):
            print("  -> At least one consecutive pair is below threshold; group is from transitivity.")

    # Show a few lowest-similarity pairs in the group (evidence they shouldn't be together)
    order = sorted(range(len(sims)), key=lambda i: sims[i])
    print("\nFive lowest-similarity pairs in sampled set (should they be in same group?):")
    for idx in order[:5]:
        a, b = pairs[idx]
        s = sims[idx]
        da, db = durations[a], durations[b]
        print(f"  cosine={s:.4f}  dur={da:.2f}s vs {db:.2f}s  https://www.roblox.com/catalog/{a}  https://www.roblox.com/catalog/{b}")

    print("\nDone.")


if __name__ == "__main__":
    main()
