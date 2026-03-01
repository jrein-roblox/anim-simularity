#!/usr/bin/env python3
"""
Find all clips above a similarity threshold, then merge into connected groups.
Each group contains every clip that is transitively similar (A~B and B~C => A,B,C in one group).
Outputs: groups CSV (animId, clipId, duration, groupId) and groups.txt (one line per group).
"""
import argparse
import csv
import os

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


EMB_START_COL = 3


def load_embeddings(path: str):
    """Load (anim_id, clip_id, duration, embedding) per row."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) < EMB_START_COL + 1:
                continue
            anim_id, clip_id, duration = row[0], row[1], row[2]
            emb = np.array([float(x) for x in row[EMB_START_COL:]], dtype=np.float64)
            rows.append((anim_id, clip_id, duration, emb))
    return rows


def main():
    ap = argparse.ArgumentParser(description="Similarity threshold + connected groups")
    ap.add_argument("--input", default="ml_training/embeddings.csv", help="Embeddings CSV")
    ap.add_argument("--output_prefix", default="ml_training/similarity_groups", help="Output prefix for .csv and .txt")
    ap.add_argument("--threshold", type=float, default=0.95, help="Cosine similarity threshold")
    ap.add_argument("--batch", type=int, default=40000, help="Batch size for similarity matrix computation")
    args = ap.parse_args()

    if not os.path.isfile(args.input):
        raise SystemExit(f"Missing {args.input}. Run embed_animations.py first.")

    rows = load_embeddings(args.input)
    N = len(rows)
    if N == 0:
        raise SystemExit("No rows in embeddings CSV.")

    embeddings = np.stack([r[3] for r in rows])
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    normed = embeddings / np.maximum(norms, 1e-10)

    print(f"Loaded {N} clips, threshold={args.threshold}")

    # Find all pairs (i, j) with i < j and sim[i,j] >= threshold
    edges = []
    for start in range(0, N, args.batch):
        end = min(start + args.batch, N)
        A = normed[start:end]
        sim_block = A @ normed.T
        for bi in range(end - start):
            i = start + bi
            for j in range(i + 1, N):
                if sim_block[bi, j] >= args.threshold:
                    edges.append((i, j))
        print(f"  Similarity batch {end}/{N}")

    # Symmetrize edges for undirected connected components
    if not edges:
        n_components = N
        labels = np.arange(N)
        print("No pairs above threshold; each clip is its own group.")
    else:
        ii, jj = zip(*edges)
        ii = np.array(ii, dtype=np.int64)
        jj = np.array(jj, dtype=np.int64)
        ii_sym = np.concatenate([ii, jj])
        jj_sym = np.concatenate([jj, ii])
        adj = csr_matrix((np.ones(len(ii_sym)), (ii_sym, jj_sym)), shape=(N, N))
        n_components, labels = connected_components(adj, directed=False)
        print(f"  {len(edges)} edges -> {n_components} connected groups")

    # groups.csv: animId, clipId, duration, groupId
    csv_path = args.output_prefix + ".csv"
    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["animId", "clipId", "duration", "groupId"])
        for i, r in enumerate(rows):
            w.writerow([r[0], r[1], r[2], int(labels[i])])

    # groups.txt: one line per group, space-separated "animId-clipId" (sorted smallest to largest)
    from collections import defaultdict

    def sort_clip_ids(clips):
        """Sort list of 'animId-clipId' by (anim_id, clip_id) numerically then lexically."""
        def key(c):
            a, b = c.split("-", 1)
            try:
                return (int(a), int(b))
            except ValueError:
                return (a, b)
        return sorted(clips, key=key)

    by_group = defaultdict(list)
    for i, r in enumerate(rows):
        by_group[int(labels[i])].append(f"{r[0]}-{r[1]}")
    for gid in by_group:
        by_group[gid] = sort_clip_ids(by_group[gid])
    txt_path = args.output_prefix + ".txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        for gid in sorted(by_group.keys()):
            clips = by_group[gid]
            if len(clips) <= 1:
                continue
            for clip in clips:
                anim_id, _ = clip.split("-", 1)
                f.write(f"{anim_id},")
            f.write("\n")
            #f.write(" ".join(by_group[gid]) + "\n")

    # Catalog URLs per group: only groups with >1 clip, numbered sequentially, ids sorted smallest to largest
    catalog_path = args.output_prefix + "_catalog.txt"
    with open(catalog_path, "w", encoding="utf-8") as f:
        seq = 0
        for gid in sorted(by_group.keys()):
            clips = by_group[gid]
            if len(clips) <= 1:
                continue
            seq += 1
            f.write(f"# group {seq} ({len(clips)} clips)\n")
            for clip in clips:
                anim_id, _ = clip.split("-", 1)
                f.write(f"  https://www.roblox.com/catalog/{anim_id}\n")
            f.write("\n")

    print(f"Wrote {csv_path}, {txt_path}, {catalog_path}")


if __name__ == "__main__":
    main()
