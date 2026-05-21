#!/usr/bin/env python3
"""
Codec Comparison: E8 vs Random vs KMeans for quantizing LLM operator vectors
=============================================================================
Compares five codebooks for quantizing 8D compressed reasoning vectors from
the memory store (~992 geometries).

Codebooks:
  E8-240:     240 E8 root vectors (geometric, optimal sphere packing)
  Random-256: 256 random unit vectors on S⁷ (byte-aligned, unstructured)
  KMeans-256: 256 centroids from k-means on actual data (data-optimized)
  KMeans-240: 240 centroids from k-means (same count as E8)
  Random-240: 240 random unit vectors (same count as E8)

Key question: does E8-240 beat Random-256 on angle preservation despite
having fewer entries?
"""

import json
import math
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

import sys
sys.path.insert(0, str(ROOT_DIR / "src"))
from tower import embed_verbs_r8, load_e8_roots

N_RANDOM_TRIALS = 10


def load_operator_vectors():
    """Load all operator vectors from memory store."""
    mem_dir = ROOT_DIR / "memory"
    vectors = []
    metadata = []
    for f in sorted(mem_dir.glob("*.json")):
        with open(f) as fh:
            entry = json.load(fh)
        verbs = entry.get("unique_verbs", [])
        if not verbs:
            continue
        v = embed_verbs_r8(verbs)
        if np.linalg.norm(v) > 0:
            vectors.append(v)
            metadata.append({
                "model": entry.get("model"),
                "task_id": entry.get("task_id"),
                "generation": entry.get("generation"),
                "budget": entry.get("budget_at_creation"),
                "operator_count": entry.get("operator_count"),
            })
    return np.array(vectors), metadata


def build_e8_codebook():
    roots = load_e8_roots()
    return roots  # already unit-normalized


def build_random_codebook(n, dim=8, seed=0):
    rng = np.random.RandomState(seed)
    v = rng.randn(n, dim)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def build_kmeans_codebook(vectors, n_clusters, seed=42):
    # Normalize inputs
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normed = vectors / norms

    km = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10, max_iter=300)
    km.fit(normed)
    centroids = km.cluster_centers_
    # Normalize centroids to unit sphere
    cnorms = np.linalg.norm(centroids, axis=1, keepdims=True)
    cnorms[cnorms == 0] = 1
    return centroids / cnorms


def pairwise_cosines_flat(vecs):
    """Upper-triangle pairwise cosines."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normed = vecs / norms
    G = normed @ normed.T
    idx = np.triu_indices(len(vecs), k=1)
    return G[idx]


def quantize(vectors, codebook):
    """Quantize each vector to nearest codebook entry. Returns indices and reconstructed vectors."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normed = vectors / norms
    sim = normed @ codebook.T  # (N, K)
    indices = sim.argmax(axis=1)
    reconstructed = codebook[indices]
    return indices, reconstructed


def evaluate_codebook(vectors, codebook, label):
    """Evaluate a codebook on all metrics."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normed = vectors / norms

    indices, reconstructed = quantize(vectors, codebook)

    # Mean reconstruction error (cosine distance = 1 - cosine_similarity)
    cos_to_nearest = np.sum(normed * reconstructed, axis=1)
    mean_cos = float(cos_to_nearest.mean())
    mean_error = float((1 - cos_to_nearest).mean())

    # Angle preservation
    orig_angles = pairwise_cosines_flat(normed)
    quant_angles = pairwise_cosines_flat(reconstructed)

    if np.std(orig_angles) > 0 and np.std(quant_angles) > 0:
        angle_corr = float(np.corrcoef(orig_angles, quant_angles)[0, 1])
    else:
        angle_corr = 0.0

    # Round-trip fidelity: quantize → reconstruct → measure drift from original
    # (same as reconstruction error for nearest-neighbor VQ)
    drift = float(np.mean(np.linalg.norm(normed - reconstructed, axis=1)))

    # Bits per vector
    bits = math.log2(codebook.shape[0])

    # Codebook utilization
    unique_used = len(set(indices.tolist()))
    utilization = unique_used / codebook.shape[0]

    # Efficiency: angle preservation per bit
    efficiency = angle_corr / bits if bits > 0 else 0

    return {
        "label": label,
        "codebook_size": codebook.shape[0],
        "bits_per_vector": round(bits, 2),
        "mean_cosine": round(mean_cos, 6),
        "mean_error": round(mean_error, 6),
        "angle_preservation": round(angle_corr, 6),
        "round_trip_drift": round(drift, 6),
        "efficiency_per_bit": round(efficiency, 6),
        "codebook_utilization": round(utilization, 4),
        "unique_entries_used": unique_used,
    }


def main():
    print("=" * 70)
    print("CODEC COMPARISON: E8 vs Random vs KMeans")
    print("=" * 70)

    # Load data
    vectors, meta = load_operator_vectors()
    print(f"\nOperator vectors: {len(vectors)}")
    sparsities = [np.sum(v > 0) for v in vectors]
    print(f"Mean sparsity: {np.mean(sparsities):.1f} / 8 nonzero dims")

    # Build codebooks
    print("\nBuilding codebooks...")
    e8 = build_e8_codebook()
    print(f"  E8-240:     {e8.shape}")

    kmeans_256 = build_kmeans_codebook(vectors, 256)
    print(f"  KMeans-256: {kmeans_256.shape}")

    kmeans_240 = build_kmeans_codebook(vectors, 240, seed=43)
    print(f"  KMeans-240: {kmeans_240.shape}")

    # Evaluate deterministic codebooks
    print("\nEvaluating codebooks...")
    results = {}

    for label, cb in [
        ("E8-240", e8),
        ("KMeans-256", kmeans_256),
        ("KMeans-240", kmeans_240),
    ]:
        r = evaluate_codebook(vectors, cb, label)
        results[label] = r
        print(f"  {label:<15} cos={r['mean_cosine']:.4f}  angle_pres={r['angle_preservation']:.4f}  "
              f"bits={r['bits_per_vector']:.1f}  eff/bit={r['efficiency_per_bit']:.4f}  "
              f"util={r['codebook_utilization']:.2f}")

    # Evaluate random codebooks (10 trials each)
    for n_vecs, label_base in [(256, "Random-256"), (240, "Random-240")]:
        trial_results = []
        for seed in range(N_RANDOM_TRIALS):
            cb = build_random_codebook(n_vecs, 8, seed=seed)
            r = evaluate_codebook(vectors, cb, f"{label_base}-s{seed}")
            trial_results.append(r)

        # Aggregate
        agg = {
            "label": label_base,
            "codebook_size": n_vecs,
            "bits_per_vector": trial_results[0]["bits_per_vector"],
            "mean_cosine": round(np.mean([r["mean_cosine"] for r in trial_results]), 6),
            "mean_cosine_std": round(np.std([r["mean_cosine"] for r in trial_results]), 6),
            "mean_error": round(np.mean([r["mean_error"] for r in trial_results]), 6),
            "angle_preservation": round(np.mean([r["angle_preservation"] for r in trial_results]), 6),
            "angle_preservation_std": round(np.std([r["angle_preservation"] for r in trial_results]), 6),
            "round_trip_drift": round(np.mean([r["round_trip_drift"] for r in trial_results]), 6),
            "efficiency_per_bit": round(np.mean([r["efficiency_per_bit"] for r in trial_results]), 6),
            "codebook_utilization": round(np.mean([r["codebook_utilization"] for r in trial_results]), 4),
            "unique_entries_used": round(np.mean([r["unique_entries_used"] for r in trial_results]), 1),
            "n_trials": N_RANDOM_TRIALS,
            "all_trials": trial_results,
        }
        results[label_base] = agg
        ap = agg["angle_preservation"]
        ap_std = agg["angle_preservation_std"]
        print(f"  {label_base:<15} cos={agg['mean_cosine']:.4f}  angle_pres={ap:.4f}±{ap_std:.4f}  "
              f"bits={agg['bits_per_vector']:.1f}  eff/bit={agg['efficiency_per_bit']:.4f}  "
              f"util={agg['codebook_utilization']:.2f}")

    # ── Summary table ───────────────────────────────────────────────────────
    ordered = ["E8-240", "Random-240", "KMeans-240", "Random-256", "KMeans-256"]
    print(f"\n{'=' * 85}")
    print("CODEC COMPARISON RESULTS")
    print(f"{'─' * 85}")
    print(f"  {'Codebook':<15} | {'Size':>5} | {'Bits':>5} | {'Mean Cos':>9} | "
          f"{'Angle Pres':>11} | {'Eff/Bit':>8} | {'Util':>6}")
    print(f"  {'─' * 80}")

    for label in ordered:
        r = results[label]
        ap = r["angle_preservation"]
        ap_str = f"{ap:.4f}"
        if "angle_preservation_std" in r:
            ap_str += f"±{r['angle_preservation_std']:.3f}"
        mc = r["mean_cosine"]
        mc_str = f"{mc:.4f}"
        if "mean_cosine_std" in r:
            mc_str += f"±{r['mean_cosine_std']:.3f}"
        print(f"  {label:<15} | {r['codebook_size']:>5} | {r['bits_per_vector']:>5.1f} | "
              f"{mc_str:>9} | {ap_str:>11} | {r['efficiency_per_bit']:>8.4f} | "
              f"{r['codebook_utilization']:>6.2f}")

    print(f"  {'─' * 80}")

    # Key comparisons
    e8_ap = results["E8-240"]["angle_preservation"]
    r256_ap = results["Random-256"]["angle_preservation"]
    r240_ap = results["Random-240"]["angle_preservation"]
    km240_ap = results["KMeans-240"]["angle_preservation"]
    km256_ap = results["KMeans-256"]["angle_preservation"]

    print(f"\n  KEY COMPARISONS:")
    print(f"  E8-240 vs Random-256:  {e8_ap - r256_ap:+.4f} angle preservation")
    print(f"  E8-240 vs Random-240:  {e8_ap - r240_ap:+.4f} angle preservation (same size)")
    print(f"  E8-240 vs KMeans-240:  {e8_ap - km240_ap:+.4f} angle preservation (same size)")
    print(f"  KMeans-256 vs E8-240:  {km256_ap - e8_ap:+.4f} angle preservation")

    if e8_ap > r256_ap + 0.01:
        print(f"\n  VERDICT: E8 WINS — geometric structure adds value as codec (+{e8_ap - r256_ap:.3f} over Random-256)")
    elif e8_ap > r240_ap + 0.01:
        print(f"\n  VERDICT: E8 WINS AT SIZE — beats random at same codebook size (+{e8_ap - r240_ap:.3f})")
    elif abs(e8_ap - r256_ap) < 0.01:
        print(f"\n  VERDICT: E8 ≈ RANDOM — no meaningful advantage ({e8_ap - r256_ap:+.3f})")
    else:
        print(f"\n  VERDICT: E8 LOSES — random codebook outperforms ({e8_ap - r256_ap:+.3f})")

    if km240_ap > e8_ap + 0.01:
        print(f"  NOTE: KMeans-240 beats E8-240 by {km240_ap - e8_ap:+.3f} — data-adapted > geometric")
    print(f"{'=' * 85}")

    # ── Bar chart ───────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    colors = {
        "E8-240": "#e63946",
        "Random-240": "#a8dadc",
        "KMeans-240": "#457b9d",
        "Random-256": "#bde0fe",
        "KMeans-256": "#1d3557",
    }

    # Angle preservation
    ax = axes[0]
    vals = [results[l]["angle_preservation"] for l in ordered]
    errs = [results[l].get("angle_preservation_std", 0) for l in ordered]
    bars = ax.bar(range(len(ordered)), vals, yerr=errs, capsize=4,
                  color=[colors[l] for l in ordered], edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(ordered)))
    ax.set_xticklabels(ordered, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Angle Preservation (r)")
    ax.set_title("Angle Preservation")
    ax.grid(True, alpha=0.3, axis="y")

    # Mean cosine
    ax = axes[1]
    vals = [results[l]["mean_cosine"] for l in ordered]
    errs = [results[l].get("mean_cosine_std", 0) for l in ordered]
    ax.bar(range(len(ordered)), vals, yerr=errs, capsize=4,
           color=[colors[l] for l in ordered], edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(ordered)))
    ax.set_xticklabels(ordered, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Mean Cosine to Nearest")
    ax.set_title("Reconstruction Quality")
    ax.grid(True, alpha=0.3, axis="y")

    # Efficiency per bit
    ax = axes[2]
    vals = [results[l]["efficiency_per_bit"] for l in ordered]
    ax.bar(range(len(ordered)), vals,
           color=[colors[l] for l in ordered], edgecolor="black", linewidth=0.5)
    ax.set_xticks(range(len(ordered)))
    ax.set_xticklabels(ordered, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Angle Preservation / Bit")
    ax.set_title("Efficiency per Bit")
    ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle(f"Codec Comparison: Quantizing {len(vectors)} LLM Operator Vectors (8D)", fontsize=13)
    plt.tight_layout()
    fig.savefig(str(PLOTS_DIR / "codec_comparison.png"), dpi=150, bbox_inches="tight")
    print(f"\n  Saved: {PLOTS_DIR / 'codec_comparison.png'}")

    # ── Save JSON ───────────────────────────────────────────────────────────
    # Strip trial-level detail from random codebooks for main output
    json_out = {}
    for label in ordered:
        r = dict(results[label])
        r.pop("all_trials", None)
        json_out[label] = r

    json_out["metadata"] = {
        "n_vectors": len(vectors),
        "mean_sparsity": round(float(np.mean(sparsities)), 2),
        "n_random_trials": N_RANDOM_TRIALS,
    }

    json_path = RESULTS_DIR / "codec_comparison.json"
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"  Saved: {json_path}")


if __name__ == "__main__":
    main()
