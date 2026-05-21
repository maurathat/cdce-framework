#!/usr/bin/env python3
"""
E8 Specificity Test
===================
Is E8 special, or can any 240-vector codebook achieve the same alignment?

Train identical projection heads (Linear 1280→8, 20 epochs, Adam lr=0.001)
against E8 roots vs a random codebook of 240 unit vectors in 8D.
If both hit ~0.997, E8 isn't special. If E8 wins, that's specificity.
"""

import json
import ssl
import os
from pathlib import Path

import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Reuse infrastructure from the main test
from jepa_e8_test import (
    DEVICE, RESULTS_DIR, NUM_IMAGES, TRAIN_EPOCHS, TRAIN_LR,
    build_e8_roots, load_ijepa_encoder, load_cifar10_subset,
    extract_embeddings, train_projection, analyze_alignment,
)


def build_random_codebook(n_vectors=240, dim=8, seed=42):
    """Generate n random unit vectors in dim-D as a control codebook."""
    rng = np.random.RandomState(seed)
    vecs = rng.randn(n_vectors, dim)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def main():
    print("=" * 60)
    print("E8 SPECIFICITY TEST")
    print("Is E8 special, or will any 240-vector codebook do?")
    print("=" * 60)

    # Load E8 roots
    e8_roots = build_e8_roots()

    # Generate multiple random codebooks for robustness
    n_codebooks = 5
    random_codebooks = [build_random_codebook(240, 8, seed=i) for i in range(n_codebooks)]

    # Load encoder & data
    print("\n[1] Loading I-JEPA encoder...")
    model, embed_dim = load_ijepa_encoder()

    print("\n[2] Loading CIFAR-10...")
    loader = load_cifar10_subset(NUM_IMAGES)

    print("\n[3] Extracting embeddings...")
    embeddings = extract_embeddings(model, loader)
    print(f"  Shape: {embeddings.shape}")

    # Train against E8
    print(f"\n{'─' * 60}")
    print("[4] Training projection → E8 roots (240 vectors)")
    print(f"{'─' * 60}")
    e8_proj = train_projection(embeddings, e8_roots, 8)
    with torch.no_grad():
        e8_projected = e8_proj(embeddings.to(DEVICE)).cpu().numpy()
    e8_res = analyze_alignment(e8_projected, e8_roots, "E8-trained")

    # Train against each random codebook
    random_results = []
    for i, codebook in enumerate(random_codebooks):
        print(f"\n{'─' * 60}")
        print(f"[5.{i+1}] Training projection → Random codebook #{i+1} (240 vectors, seed={i})")
        print(f"{'─' * 60}")
        rand_proj = train_projection(embeddings, codebook, 8)
        with torch.no_grad():
            rand_projected = rand_proj(embeddings.to(DEVICE)).cpu().numpy()
        rand_res = analyze_alignment(rand_projected, codebook, f"Random-{i+1}-trained")
        random_results.append(rand_res)

    # Also: train against E8, then measure against random codebook (cross-test)
    print(f"\n{'─' * 60}")
    print("[6] Cross-test: E8-trained projection measured against random codebook")
    print(f"{'─' * 60}")
    cross_res = analyze_alignment(e8_projected, random_codebooks[0], "E8-proj→random-cb")

    # Summary
    rand_mean_cos = np.mean([r["mean_cos"] for r in random_results])
    rand_std_cos = np.std([r["mean_cos"] for r in random_results])

    print(f"\n{'=' * 70}")
    print("E8 SPECIFICITY RESULTS")
    print(f"{'─' * 70}")
    print(f"  {'Condition':<35} | {'Mean Cos':>10} | {'Std':>8} | {'Max Cos':>10}")
    print(f"  {'─' * 67}")
    print(f"  {'E8 roots (trained)':<35} | {e8_res['mean_cos']:>10.4f} | {e8_res['std_cos']:>8.4f} | {e8_res['max_cos']:>10.4f}")
    for i, r in enumerate(random_results):
        print(f"  {f'Random codebook #{i+1} (trained)':<35} | {r['mean_cos']:>10.4f} | {r['std_cos']:>8.4f} | {r['max_cos']:>10.4f}")
    print(f"  {'─' * 67}")
    print(f"  {'Random codebook avg':<35} | {rand_mean_cos:>10.4f} | {rand_std_cos:>8.4f} |")
    print(f"  {'E8-proj → random codebook':<35} | {cross_res['mean_cos']:>10.4f} | {cross_res['std_cos']:>8.4f} | {cross_res['max_cos']:>10.4f}")
    print(f"  {'─' * 67}")

    delta = e8_res["mean_cos"] - rand_mean_cos
    print(f"\n  E8 advantage: {delta:+.4f} (E8 {e8_res['mean_cos']:.4f} vs Random avg {rand_mean_cos:.4f})")
    if delta > 0.01:
        print(f"  VERDICT: E8 IS SPECIFIC — {delta:.4f} advantage over random codebooks")
    elif delta > 0.001:
        print(f"  VERDICT: MARGINAL — E8 has small {delta:.4f} edge, needs more investigation")
    else:
        print(f"  VERDICT: NOT SPECIFIC — random codebooks achieve comparable alignment")

    print(f"{'=' * 70}")

    # Histogram
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.hist(e8_res["cosine_distribution"], bins=50, alpha=0.6,
            color="#e63946", label=f"E8 roots (mean={e8_res['mean_cos']:.4f})", density=True)
    ax.hist(random_results[0]["cosine_distribution"], bins=50, alpha=0.6,
            color="#457b9d", label=f"Random codebook (mean={random_results[0]['mean_cos']:.4f})", density=True)
    ax.hist(cross_res["cosine_distribution"], bins=50, alpha=0.6,
            color="#2a9d8f", label=f"E8-proj → random cb (mean={cross_res['mean_cos']:.4f})", density=True)
    ax.set_xlabel("Cosine similarity to nearest codebook vector", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("E8 Specificity: E8 vs Random 240-Vector Codebook", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    hist_path = RESULTS_DIR / "e8_specificity_histogram.png"
    fig.savefig(str(hist_path), dpi=150)
    print(f"  Saved: {hist_path}")

    # Save JSON
    json_out = {
        "e8_trained": {k: v for k, v in e8_res.items() if k != "cosine_distribution"},
        "random_codebooks": [
            {k: v for k, v in r.items() if k != "cosine_distribution"}
            for r in random_results
        ],
        "random_codebook_avg_mean_cos": round(rand_mean_cos, 6),
        "random_codebook_std_mean_cos": round(rand_std_cos, 6),
        "cross_test": {k: v for k, v in cross_res.items() if k != "cosine_distribution"},
        "e8_advantage": round(delta, 6),
        "verdict": "specific" if delta > 0.01 else "marginal" if delta > 0.001 else "not_specific",
    }
    json_path = RESULTS_DIR / "e8_specificity_results.json"
    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"  Saved: {json_path}")


if __name__ == "__main__":
    main()
