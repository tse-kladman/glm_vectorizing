#!/usr/bin/env python
"""
kmer_sweep.py -- sweep the k-mer word length and evaluate clustering/retrieval quality.

For each k in K_VALUES:
  1. Compute k-mer frequency vectors for all 50 reference samples
  2. Calculate P@5, ARI, Silhouette for each sample
  3. Aggregate mean +/- std across samples
  4. Save everything to one Excel file
  5. Plot metrics vs k with error bars (std)

This is the source of the numbers behind Fig. 3.  It needs the reference FASTA
samples in ``data/random_samples/``, which are too large to ship with the repo, so
its Excel output is kept under ``data/metrics/kmer_sweep/`` and the aggregated
figure data is committed as ``results/figure_data/fig3_kmer_sweep.csv``.  Unless you
have the raw samples, run only Stage 2 of ``main_figures.ipynb`` -- nothing here has
to be re-run to restyle Fig. 3.

Usage:
    python visualization/kmer_sweep.py
"""

import warnings
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tqdm
from Bio import SeqIO
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

try:
    from sklearn_extra.cluster import KMedoids
except ImportError:
    KMedoids = None
    warnings.warn("sklearn_extra not installed — ARI will use KMeans instead of KMedoids")

# ─────────────────────────── CONFIGURATION ───────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent   # repository root

K_VALUES     = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # k-mer sizes to evaluate
N_SAMPLES    = 50                               # number of reference samples
PK_K         = 5                                # k for precision@K
SAMPLE_DIR   = ROOT / "data" / "random_samples"                 # input  (not shipped)
OUTPUT_DIR   = ROOT / "data" / "metrics" / "kmer_sweep"         # metric table
OUTPUT_FILE  = OUTPUT_DIR / "kmer_sweep_results.xlsx"
PLOT_FILE    = ROOT / "results" / "figures" / "exploratory" / "kmer_sweep_plot.png"

# ─────────────────────────── K-MER FUNCTIONS ─────────────────────────────────

def all_kmers(k: int):
    """Lexicographically ordered list of all 4^k k-mers."""
    return [''.join(p) for p in product("ACGT", repeat=k)]


def kmer_counts(seq: str, k: int):
    """Count k-mers in a sequence (skip non-ACGT)."""
    seq = seq.upper()
    counts = Counter()
    valid = {"A", "C", "G", "T"}
    for i in range(len(seq) - k + 1):
        kmer = seq[i:i + k]
        if set(kmer) <= valid:
            counts[kmer] += 1
    return counts


def sequence_to_kmer_vector(seq: str, k: int = 4, normalize: bool = True):
    """Return a vector of length 4^k for a sequence."""
    kmers = all_kmers(k)
    counts = kmer_counts(seq, k)

    vector = np.zeros(len(kmers), dtype=float)
    for i, kmer in enumerate(kmers):
        vector[i] = counts.get(kmer, 0)

    if normalize and vector.sum() > 0:
        vector /= vector.sum()

    return vector

# ─────────────────────────── METRIC FUNCTIONS ────────────────────────────────

def precision_K(embeddings, true_labels, k=5):
    nn = NearestNeighbors(n_neighbors=k + 1)
    nn.fit(embeddings)
    _, indices = nn.kneighbors(embeddings)
    indices = indices[:, 1:]  # exclude self

    true_labels = np.array(true_labels)
    individual_precisions = []

    for i, neighbor_indices in enumerate(indices):
        neighbor_labels = true_labels[neighbor_indices]
        same_class = np.sum(neighbor_labels == true_labels[i])
        individual_precisions.append(same_class / k)

    return np.mean(individual_precisions)


def ARI(embeddings, true_labels, n_clusters):
    if KMedoids is not None:
        km = KMedoids(n_clusters=n_clusters, metric="cosine", random_state=42)
    else:
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    preds = km.fit_predict(embeddings)
    return adjusted_rand_score(true_labels, preds)


def silhouette_cosine(embeddings, n_clusters):
    labels = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto").fit_predict(embeddings)
    return silhouette_score(embeddings, labels, metric="cosine")


def calc_metrics(data_dict, pk_k=5):
    """Calculate P@K, ARI, Silhouette for a data_dict {label: [vectors]}."""
    all_embeddings = []
    all_labels = []
    valid_labels = []

    for label, embeddings in data_dict.items():
        if len(embeddings) > 1:
            all_embeddings.append(embeddings)
            all_labels.extend([label] * len(embeddings))
            valid_labels.append(label)

    if not all_embeddings:
        return None

    all_embeddings = np.vstack(all_embeddings)
    scaled = StandardScaler().fit_transform(all_embeddings)
    n_clusters = len(valid_labels)

    pk = precision_K(scaled, all_labels, k=pk_k)
    ari = ARI(scaled, all_labels, n_clusters=n_clusters)
    sil = silhouette_cosine(scaled, n_clusters=n_clusters)

    return pk, ari, sil

# ─────────────────────────── MAIN ────────────────────────────────────────────

def load_sample_fasta(sample_idx):
    """Parse FASTA file, return list of dicts with sequence info."""
    fasta_path = SAMPLE_DIR / f"sample_{sample_idx}.fa"
    sequences_info = []
    for record in SeqIO.parse(str(fasta_path), "fasta"):
        if '_' in record.id:
            original_label = record.id.split('_')[0]
        else:
            original_label = record.id
        sequences_info.append({
            'id': record.id,
            'sequence': str(record.seq),
            'original_label': original_label,
        })
    return sequences_info


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Results storage: {k_mer_size: {metric_name: [values_per_sample]}}
    results = {k: {"P@5": [], "ARI": [], "Silhouette": []} for k in K_VALUES}

    for kmer_k in K_VALUES:
        print(f"\n{'='*60}")
        print(f"  Processing k-mer size k={kmer_k}  (vector dim = {4**kmer_k})")
        print(f"{'='*60}")

        for sample_idx in tqdm.tqdm(range(N_SAMPLES), desc=f"k={kmer_k}"):
            # Load FASTA
            sequences_info = load_sample_fasta(sample_idx)

            # Build k-mer dict
            data_dict = defaultdict(list)
            for s in sequences_info:
                vec = sequence_to_kmer_vector(s['sequence'], k=kmer_k)
                data_dict[str(s['original_label'])].append(vec)

            # Calculate metrics
            metr = calc_metrics(dict(data_dict), pk_k=PK_K)
            if metr is not None:
                results[kmer_k]["P@5"].append(metr[0])
                results[kmer_k]["ARI"].append(metr[1])
                results[kmer_k]["Silhouette"].append(metr[2])
            else:
                print(f"  WARNING: sample_{sample_idx} returned None for k={kmer_k}")

    # ── Build summary table ──────────────────────────────────────────────
    rows = []
    for kmer_k in K_VALUES:
        row = {"k": kmer_k}
        for metric_name in ["P@5", "ARI", "Silhouette"]:
            vals = results[kmer_k][metric_name]
            if vals:
                row[f"{metric_name}_mean"] = np.mean(vals)
                row[f"{metric_name}_std"]  = np.std(vals)
            else:
                row[f"{metric_name}_mean"] = np.nan
                row[f"{metric_name}_std"]  = np.nan
        rows.append(row)

    df_summary = pd.DataFrame(rows)

    # Also save raw per-sample values on a second sheet
    raw_rows = []
    for kmer_k in K_VALUES:
        for i, (pk, ari, sil) in enumerate(zip(
            results[kmer_k]["P@5"],
            results[kmer_k]["ARI"],
            results[kmer_k]["Silhouette"]
        )):
            raw_rows.append({"k": kmer_k, "sample": i, "P@5": pk, "ARI": ari, "Silhouette": sil})

    df_raw = pd.DataFrame(raw_rows)

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        df_summary.to_excel(writer, sheet_name="summary", index=False)
        df_raw.to_excel(writer, sheet_name="raw", index=False)

    print(f"\nResults saved → {OUTPUT_FILE}")
    print(df_summary.to_string(index=False))

    # ── Plot ─────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = ["P@5", "ARI", "Silhouette"]
    colors = ["#2196F3", "#4CAF50", "#FF9800"]

    for ax, metric_name, color in zip(axes, metrics, colors):
        means = df_summary[f"{metric_name}_mean"].values
        stds  = df_summary[f"{metric_name}_std"].values
        ks    = df_summary["k"].values

        ax.errorbar(ks, means, yerr=stds, fmt='o-', color=color,
                    capsize=5, capthick=1.5, linewidth=2, markersize=8,
                    ecolor=color, alpha=0.85, label=metric_name)
        ax.fill_between(ks, means - stds, means + stds, alpha=0.15, color=color)

        ax.set_xlabel("k (k-mer size)", fontsize=13)
        ax.set_ylabel(metric_name, fontsize=13)
        ax.set_title(metric_name, fontsize=14, fontweight='bold')
        ax.set_xticks(ks)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=11)

    fig.suptitle("Clustering & Retrieval Metrics vs k-mer Size", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(PLOT_FILE, dpi=200, bbox_inches='tight')
    print(f"Plot saved → {PLOT_FILE}")
    plt.show()


if __name__ == "__main__":
    main()
