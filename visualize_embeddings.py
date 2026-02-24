#!/usr/bin/env python3
"""
Load normalization.csv (or fingerprints.csv) and visualize embeddings:
- t-SNE 2D with K-means clustering (color by cluster)
- PCA 2D with variance explained
- UMAP 2D if umap-learn is installed
- Pairwise cosine similarity distribution
- Elbow/silhouette for cluster count
- Optional: sample similarity heatmap
"""

import csv
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples

CSV_PATH = "ml_training/embeddings.csv"
EMB_DIM = 128
EMB_START_COL = 3
OUT_DIR = "embedding_plots"
MAX_POINTS_TSNE = 5000  # subsample for t-SNE (slow on 30k+)
RANDOM_STATE = 42
N_CLUSTERS = 12  # for K-means; also try 8–20


def load_embeddings(path: str):
    """Load CSV: returns anim_ids, durations, embeddings (N x 128)."""
    anim_ids = []
    durations = []
    emb_list = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) < EMB_START_COL + EMB_DIM:
                continue
            anim_ids.append(row[0])
            try:
                durations.append(float(row[2]))
            except ValueError:
                durations.append(0.0)
            def safe_float(s):
                try:
                    v = float(s)
                    return v if np.isfinite(v) else 0.0
                except (ValueError, TypeError):
                    return 0.0
            emb = [safe_float(x) for x in row[EMB_START_COL : EMB_START_COL + EMB_DIM]]
            emb_list.append(emb)
    embeddings = np.array(emb_list, dtype=np.float64)
    return anim_ids, durations, embeddings


def l2_normalize(X: np.ndarray):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(norms, 1e-10)


def main():
    path = CSV_PATH
    if not os.path.isfile(path):
        print(f" {CSV_PATH} not found. Exiting.")
        sys.exit(1)
    print(f"Loading {path}...")
    anim_ids, durations, embeddings = load_embeddings(path)
    N = len(anim_ids)
    print(f"Loaded {N} rows, embedding dim {embeddings.shape[1]}")

    embeddings = l2_normalize(embeddings)
    durations = np.array(durations)

    os.makedirs(OUT_DIR, exist_ok=True)

    # Subsample for t-SNE / UMAP if large
    if N > MAX_POINTS_TSNE:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(N, size=MAX_POINTS_TSNE, replace=False)
        idx.sort()
        X_plot = embeddings[idx]
        anim_plot = [anim_ids[i] for i in idx]
        dur_plot = durations[idx]
        n_plot = MAX_POINTS_TSNE
        print(f"Subsampled to {n_plot} points for t-SNE/UMAP.")
    else:
        X_plot = embeddings
        anim_plot = anim_ids
        dur_plot = durations
        n_plot = N

    # --- 1. PCA 2D ---
    print("Running PCA...")
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_plot)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(X_pca[:, 0], X_pca[:, 1], s=8, alpha=0.6, c="steelblue", edgecolors="none")
    ax.set_xlabel(f"PC1 ({100*pca.explained_variance_ratio_[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({100*pca.explained_variance_ratio_[1]:.1f}%)")
    ax.set_title("PCA of embeddings (L2-normalized)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "pca_2d.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved pca_2d.png")

    # --- 2. t-SNE 2D (no cluster yet) ---
    print("Running t-SNE (this may take a minute)...")
    tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=min(30, n_plot - 1), max_iter=1000)
    X_tsne = tsne.fit_transform(X_plot)

    # --- 3. K-means on full embedding for coloring ---
    print("Clustering with K-means...")
    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    labels_plot = kmeans.fit_predict(X_plot)

    # --- 4. t-SNE colored by cluster ---
    fig, ax = plt.subplots(figsize=(12, 10))
    scatter = ax.scatter(
        X_tsne[:, 0], X_tsne[:, 1],
        c=labels_plot, s=12, alpha=0.7, cmap="tab20", edgecolors="none"
    )
    plt.colorbar(scatter, ax=ax, label="Cluster")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title(f"t-SNE of embeddings (K-means k={N_CLUSTERS})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "tsne_clusters.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved tsne_clusters.png")

    # --- 5. t-SNE colored by duration ---
    fig, ax = plt.subplots(figsize=(12, 10))
    sc = ax.scatter(
        X_tsne[:, 0], X_tsne[:, 1],
        c=dur_plot, s=12, alpha=0.7, cmap="viridis", edgecolors="none"
    )
    plt.colorbar(sc, ax=ax, label="Duration (s)")
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.set_title("t-SNE colored by duration")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "tsne_duration.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved tsne_duration.png")

    # --- 6. PCA colored by cluster ---
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        X_pca[:, 0], X_pca[:, 1],
        c=labels_plot, s=12, alpha=0.7, cmap="tab20", edgecolors="none"
    )
    plt.colorbar(scatter, ax=ax, label="Cluster")
    ax.set_xlabel(f"PC1 ({100*pca.explained_variance_ratio_[0]:.1f}%)")
    ax.set_ylabel(f"PC2 ({100*pca.explained_variance_ratio_[1]:.1f}%)")
    ax.set_title(f"PCA colored by K-means (k={N_CLUSTERS})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "pca_clusters.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved pca_clusters.png")

    # --- 7. Pairwise cosine similarity distribution (sample pairs) ---
    print("Computing pairwise similarity sample...")
    n_sim_sample = min(100_000, (N * (N - 1)) // 2)
    rng = np.random.default_rng(RANDOM_STATE)
    sims = []
    for _ in range(n_sim_sample):
        i, j = rng.integers(0, N, size=2)
        if i != j:
            sims.append(float(np.dot(embeddings[i], embeddings[j])))
    sims = np.array(sims)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(sims, bins=80, color="steelblue", alpha=0.8, edgecolor="white")
    ax.axvline(0.99, color="red", linestyle="--", label="Threshold 0.99")
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Count")
    ax.set_title("Pairwise cosine similarity (random sample)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "similarity_histogram.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved similarity_histogram.png")

    # --- 8. Elbow + silhouette for cluster count ---
    k_range = range(4, min(21, n_plot // 20))
    inertias = []
    sil_scores = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=5)
        km.fit(X_plot)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_plot, km.labels_, sample_size=min(5000, n_plot)))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(list(k_range), inertias, "o-", color="steelblue")
    ax1.set_xlabel("Number of clusters k")
    ax1.set_ylabel("Inertia")
    ax1.set_title("Elbow method")
    ax2.plot(list(k_range), sil_scores, "o-", color="coral")
    ax2.set_xlabel("Number of clusters k")
    ax2.set_ylabel("Silhouette score")
    ax2.set_title("Silhouette vs k")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "cluster_choice.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved cluster_choice.png")

    # --- 9. UMAP if available ---
    try:
        import umap
        print("Running UMAP...")
        reducer = umap.UMAP(n_components=2, random_state=RANDOM_STATE, n_neighbors=15, min_dist=0.1)
        X_umap = reducer.fit_transform(X_plot)
        fig, ax = plt.subplots(figsize=(12, 10))
        scatter = ax.scatter(
            X_umap[:, 0], X_umap[:, 1],
            c=labels_plot, s=12, alpha=0.7, cmap="tab20", edgecolors="none"
        )
        plt.colorbar(scatter, ax=ax, label="Cluster")
        ax.set_xlabel("UMAP 1")
        ax.set_ylabel("UMAP 2")
        ax.set_title(f"UMAP of embeddings (K-means k={N_CLUSTERS})")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "umap_clusters.png"), dpi=150, bbox_inches="tight")
        plt.close()
        print("Saved umap_clusters.png")
    except ImportError:
        print("UMAP not installed (pip install umap-learn). Skipping UMAP plot.")

    # --- 10. PCA variance explained (first 20 components) ---
    pca_full = PCA(n_components=min(20, X_plot.shape[1], X_plot.shape[0] - 1), random_state=RANDOM_STATE)
    pca_full.fit(X_plot)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(range(1, len(pca_full.explained_variance_ratio_) + 1), pca_full.explained_variance_ratio_, color="steelblue", alpha=0.8)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Variance explained ratio")
    ax.set_title("PCA variance explained (first components)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "pca_variance.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved pca_variance.png")

    print(f"\nAll plots saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
