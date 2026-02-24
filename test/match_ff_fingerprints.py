#!/usr/bin/env python3
"""
Match animation clips by FFT (Shazam-style) fingerprint overlap.

Loads ff_fingerprints.csv (animId, clipId, duration, time_offset, hash).
Builds: hash -> list of (anim_id, clip_id, t1).
For each pair of clips, count how many hashes they share (optionally with
consistent time offset). Output top pairs by overlap count.

Usage:
  python match_ff_fingerprints.py [ff_fingerprints.csv]
  python match_ff_fingerprints.py --top 50
"""

import csv
import os
import sys
from collections import defaultdict

CSV_PATH = "ff_fingerprints.csv"
DEFAULT_TOP = 20


def load_fingerprints(path: str):
    """Returns: list of (anim_id, clip_id, duration, time_offset, hash)."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            if len(row) < 5:
                continue
            anim_id, clip_id, duration, time_offset, h = row[0], row[1], row[2], row[3], row[4]
            try:
                rows.append((anim_id, clip_id, float(duration), int(time_offset), int(h)))
            except ValueError:
                continue
    return rows


def main():
    path = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else CSV_PATH
    top = DEFAULT_TOP
    for i, arg in enumerate(sys.argv):
        if arg == "--top" and i + 1 < len(sys.argv):
            top = int(sys.argv[i + 1])
            break

    if not os.path.isfile(path):
        # Try from repo root
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(base, "ff_fingerprints.csv")
    if not os.path.isfile(path):
        print("ff_fingerprints.csv not found. Run anim_fingerprint_fft.lua first.")
        sys.exit(1)

    rows = load_fingerprints(path)
    print(f"Loaded {len(rows)} hash rows from {path}")

    # Clip key: (anim_id, clip_id) -> set of hashes (matching by hash overlap)
    clip_hashes: dict[tuple[str, str], set[int]] = defaultdict(set)
    for anim_id, clip_id, _dur, _t1, h in rows:
        clip_hashes[(anim_id, clip_id)].add(h)

    clips = list(clip_hashes.keys())
    print(f"Clips: {len(clips)}")

    # Pairwise overlap (by hash only; time-offset consistency could be added)
    overlaps: list[tuple[int, tuple[str, str], tuple[str, str]]] = []
    for i in range(len(clips)):
        a = clips[i]
        set_a = clip_hashes[a]
        for j in range(i + 1, len(clips)):
            b = clips[j]
            set_b = clip_hashes[b]
            common = len(set_a & set_b)  # same (h, t1) pairs
            if common > 0:
                overlaps.append((common, a, b))

    overlaps.sort(key=lambda x: -x[0])
    print(f"\nTop {top} clip pairs by shared hash count:")
    print("-" * 60)
    for i, (count, (a_id, a_clip), (b_id, b_clip)) in enumerate(overlaps[: top]):
        size_a = len(clip_hashes[(a_id, a_clip)])
        size_b = len(clip_hashes[(b_id, b_clip)])
        print(f"  {count:5}  ({a_id}, {a_clip}) [#{size_a}]  <->  ({b_id}, {b_clip}) [#{size_b}]")

    if not overlaps:
        print("  No overlapping hashes between any pair.")


if __name__ == "__main__":
    main()
