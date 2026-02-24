#!/usr/bin/env python3
"""
Load embeddings CSV; L2-normalize; threshold by cosine similarity; output groups (connected components).
"""
import argparse
import csv
import os

import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix


def load_embeddings(path: str):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        emb_start = 3
        for row in reader:
            if len(row) < emb_start + 1:
                continue
            anim_id, clip_id, duration = row[0], row[1], row[2]
            emb = np.array([float(x) for x in row[emb_start:]], dtype=np.float64)
            rows.append((anim_id, clip_id, duration, emb))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="ml_training/embeddings.csv", help="Embeddings CSV")
    ap.add_argument("--output_prefix", default="ml_training/groups", help="Output groups.csv and groups.txt")
    ap.add_argument("--threshold", type=float, default=0.95, help="Cosine similarity threshold")
    args = ap.parse_args()

    if not os.path.isfile(args.input):
        raise SystemExit(f"Missing {args.input}. Run embed_animations.py first.")

    rows = load_embeddings(args.input)
    N = len(rows)
    embeddings = np.stack([r[3] for r in rows])
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-10)
    embeddings = embeddings / norms

    # Pairwise cosine = dot product when normalized
    sim = embeddings @ embeddings.T
    edges = []
    for i in range(N):
        for j in range(i + 1, N):
            if sim[i, j] >= args.threshold:
                edges.append((i, j))

    # Build adjacency and connected components
    if not edges:
        n_components = N
        labels = np.arange(N)
    else:
        ii, jj = zip(*edges)
        ii = list(ii) + list(jj)
        jj = list(jj) + list(ii)
        adj = csr_matrix((np.ones(len(ii)), (ii, jj)), shape=(N, N))
        n_components, labels = connected_components(adj, directed=False)

    # groups.csv: animId, clipId, duration, groupId
    csv_path = args.output_prefix + ".csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["animId", "clipId", "duration", "groupId"])
        for i, r in enumerate(rows):
            w.writerow([r[0], r[1], r[2], int(labels[i])])

    # groups.txt: one line per group, space-separated animIds
    txt_path = args.output_prefix + ".txt"
    from collections import defaultdict
    by_group = defaultdict(list)
    for i, r in enumerate(rows):
        by_group[int(labels[i])].append(r[0])
    with open(txt_path, "w", encoding="utf-8") as f:
        for gid in sorted(by_group.keys()):
            f.write(" ".join(by_group[gid]) + "\n")

    print(f"Threshold {args.threshold}: {len(edges)} edges, {n_components} groups")
    print(f"Wrote {csv_path} and {txt_path}")


if __name__ == "__main__":
    main()
