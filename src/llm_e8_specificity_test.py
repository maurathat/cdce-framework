#!/usr/bin/env python3
"""
LLM Operator E8 Specificity Test
=================================
Tests whether the 0.82 mean cosine of LLM operator vectors to E8 roots
is meaningful, or if any codebook / random vectors with similar sparsity
would achieve the same.

Controls:
1. Real operator vectors → E8 roots (reproduce the 0.82 baseline)
2. Real operator vectors → random 240-vector codebook
3. Random vectors (same sparsity pattern) → E8 roots
4. Random vectors (same sparsity pattern) → random codebook
5. Dense random unit vectors → E8 roots
"""

import json
import numpy as np
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"


def load_e8_roots():
    with open(ROOT_DIR / "data" / "e8_roots.json") as f:
        roots = np.array(json.load(f), dtype=np.float64)
    norms = np.linalg.norm(roots, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return roots / norms


def build_random_codebook(n=240, dim=8, seed=42):
    rng = np.random.RandomState(seed)
    vecs = rng.randn(n, dim)
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def nearest_cosine_stats(vectors, codebook, label=""):
    """Compute cosine to nearest codebook vector for each input vector."""
    cos_sim = vectors @ codebook.T  # (N, K)
    nearest_cos = cos_sim.max(axis=1)
    nearest_idx = cos_sim.argmax(axis=1)

    stats = {
        "label": label,
        "n_vectors": len(vectors),
        "n_codebook": codebook.shape[0],
        "mean_cos": float(nearest_cos.mean()),
        "std_cos": float(nearest_cos.std()),
        "max_cos": float(nearest_cos.max()),
        "min_cos": float(nearest_cos.min()),
        "median_cos": float(np.median(nearest_cos)),
    }

    # If codebook is E8 (240 roots), report integer vs half-integer
    if codebook.shape[0] == 240:
        n_int = int(np.sum(nearest_idx < 112))
        stats["pct_integer"] = round(100 * n_int / len(nearest_idx), 1)
        stats["pct_half_integer"] = round(100 * (len(nearest_idx) - n_int) / len(nearest_idx), 1)

    print(f"  {label:<45} mean={stats['mean_cos']:.4f} ± {stats['std_cos']:.4f}  "
          f"max={stats['max_cos']:.4f}  median={stats['median_cos']:.4f}", end="")
    if "pct_integer" in stats:
        print(f"  int={stats['pct_integer']}%", end="")
    print()
    return stats


def extract_operator_vectors(tower_analysis):
    """Rebuild R^8 verb vectors from tower analysis comparisons."""
    from tower import embed_verbs_r8
    comps = tower_analysis["e8_comparisons"]
    vectors = []
    sparsities = []
    for c in comps:
        v = embed_verbs_r8(c["verbs"])
        if np.linalg.norm(v) > 0:
            vectors.append(v)
            sparsities.append(np.sum(v > 0))
    return np.array(vectors), sparsities


def make_sparsity_matched_random(sparsities, n_vectors, dim=8, seed=99):
    """Generate random unit vectors matching the sparsity pattern of real data."""
    rng = np.random.RandomState(seed)
    vectors = []
    for i in range(n_vectors):
        sp = sparsities[i % len(sparsities)]
        v = np.zeros(dim)
        active = rng.choice(dim, size=int(sp), replace=False)
        v[active] = rng.rand(int(sp))  # positive values like verb counts
        norm = np.linalg.norm(v)
        if norm > 0:
            v /= norm
        vectors.append(v)
    return np.array(vectors)


def main():
    print("=" * 70)
    print("LLM OPERATOR VECTOR E8 SPECIFICITY TEST")
    print("Is 0.82 mean cosine to E8 meaningful?")
    print("=" * 70)

    # Load data
    e8_roots = load_e8_roots()
    print(f"\nE8 roots: {e8_roots.shape}")

    # Load tower analysis
    tower_files = sorted(RESULTS_DIR.glob("tower_analysis_*.json"))
    if not tower_files:
        print("ERROR: No tower analysis files found")
        return
    with open(tower_files[-1]) as f:
        tower = json.load(f)
    print(f"Tower analysis: {tower_files[-1].name}")

    # Extract real operator vectors
    op_vectors, sparsities = extract_operator_vectors(tower)
    n = len(op_vectors)
    print(f"Operator vectors: {n}")
    mean_sp = np.mean(sparsities)
    print(f"Mean sparsity (nonzero dims): {mean_sp:.1f} / 8")

    # Build controls
    n_codebooks = 5
    random_codebooks = [build_random_codebook(240, 8, seed=i) for i in range(n_codebooks)]
    sparsity_matched = make_sparsity_matched_random(sparsities, n, seed=99)
    dense_random = np.random.RandomState(77).randn(n, 8)
    dense_random = dense_random / np.linalg.norm(dense_random, axis=1, keepdims=True)

    all_results = {}

    # ── Test 1: Real operators → E8 (reproduce baseline) ──
    print(f"\n{'─' * 70}")
    print("TEST 1: Real operator vectors → E8 roots")
    print(f"{'─' * 70}")
    all_results["real_vs_e8"] = nearest_cosine_stats(op_vectors, e8_roots, "Real operators → E8")

    # ── Test 2: Real operators → random codebooks ──
    print(f"\n{'─' * 70}")
    print("TEST 2: Real operator vectors → random 240-vector codebooks")
    print(f"{'─' * 70}")
    rand_cb_results = []
    for i, cb in enumerate(random_codebooks):
        r = nearest_cosine_stats(op_vectors, cb, f"Real operators → random CB #{i+1}")
        rand_cb_results.append(r)
    avg_rand_cb = np.mean([r["mean_cos"] for r in rand_cb_results])
    all_results["real_vs_random_cb_avg"] = round(avg_rand_cb, 4)

    # ── Test 3: Sparsity-matched random → E8 ──
    print(f"\n{'─' * 70}")
    print("TEST 3: Sparsity-matched random vectors → E8 roots")
    print(f"{'─' * 70}")
    all_results["sparse_random_vs_e8"] = nearest_cosine_stats(
        sparsity_matched, e8_roots, "Sparsity-matched random → E8")

    # ── Test 4: Sparsity-matched random → random codebook ──
    print(f"\n{'─' * 70}")
    print("TEST 4: Sparsity-matched random vectors → random codebook")
    print(f"{'─' * 70}")
    all_results["sparse_random_vs_random_cb"] = nearest_cosine_stats(
        sparsity_matched, random_codebooks[0], "Sparsity-matched random → random CB")

    # ── Test 5: Dense random → E8 ──
    print(f"\n{'─' * 70}")
    print("TEST 5: Dense random unit vectors → E8 roots")
    print(f"{'─' * 70}")
    all_results["dense_random_vs_e8"] = nearest_cosine_stats(
        dense_random, e8_roots, "Dense random → E8")

    # ── Test 6: Dense random → random codebook ──
    print(f"\n{'─' * 70}")
    print("TEST 6: Dense random unit vectors → random codebook")
    print(f"{'─' * 70}")
    all_results["dense_random_vs_random_cb"] = nearest_cosine_stats(
        dense_random, random_codebooks[0], "Dense random → random CB")

    # ── Summary ──
    r = all_results
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'─' * 80}")
    print(f"  {'Vectors':<30} {'Codebook':<20} {'Mean Cos':>10}")
    print(f"  {'─' * 62}")
    print(f"  {'Real operators':<30} {'E8 roots':<20} {r['real_vs_e8']['mean_cos']:>10.4f}")
    print(f"  {'Real operators':<30} {'Random CB (avg)':<20} {r['real_vs_random_cb_avg']:>10.4f}")
    print(f"  {'Sparsity-matched random':<30} {'E8 roots':<20} {r['sparse_random_vs_e8']['mean_cos']:>10.4f}")
    print(f"  {'Sparsity-matched random':<30} {'Random CB':<20} {r['sparse_random_vs_random_cb']['mean_cos']:>10.4f}")
    print(f"  {'Dense random':<30} {'E8 roots':<20} {r['dense_random_vs_e8']['mean_cos']:>10.4f}")
    print(f"  {'Dense random':<30} {'Random CB':<20} {r['dense_random_vs_random_cb']['mean_cos']:>10.4f}")
    print(f"  {'─' * 62}")

    # Key comparisons
    e8_vs_rand_cb = r["real_vs_e8"]["mean_cos"] - avg_rand_cb
    real_vs_sparse = r["real_vs_e8"]["mean_cos"] - r["sparse_random_vs_e8"]["mean_cos"]

    print(f"\n  KEY COMPARISONS:")
    print(f"  E8 advantage over random CB (same operators):  {e8_vs_rand_cb:+.4f}")
    print(f"  Real operators vs sparse-random (same E8):     {real_vs_sparse:+.4f}")

    if e8_vs_rand_cb > 0.05:
        print(f"\n  VERDICT: E8 IS SPECIFIC for LLM operators (+{e8_vs_rand_cb:.3f} over random CB)")
    elif e8_vs_rand_cb > 0.01:
        print(f"\n  VERDICT: MARGINAL E8 specificity (+{e8_vs_rand_cb:.3f})")
    else:
        print(f"\n  VERDICT: E8 NOT SPECIFIC for LLM operators (+{e8_vs_rand_cb:.3f})")

    if real_vs_sparse > 0.05:
        print(f"  VERDICT: OPERATOR STRUCTURE IS SPECIFIC (+{real_vs_sparse:.3f} over random with same sparsity)")
    elif real_vs_sparse > 0.01:
        print(f"  VERDICT: MARGINAL operator structure specificity (+{real_vs_sparse:.3f})")
    else:
        print(f"  VERDICT: OPERATOR STRUCTURE NOT SPECIFIC (+{real_vs_sparse:.3f})")

    print(f"{'=' * 80}")

    # Save
    json_path = RESULTS_DIR / "llm_e8_specificity_results.json"
    # Clean for JSON serialization
    json_out = {}
    for k, v in all_results.items():
        if isinstance(v, dict):
            json_out[k] = {kk: vv for kk, vv in v.items()}
        else:
            json_out[k] = v
    json_out["random_codebook_results"] = [
        {kk: vv for kk, vv in rr.items()} for rr in rand_cb_results
    ]
    json_out["e8_advantage_over_random_cb"] = round(e8_vs_rand_cb, 6)
    json_out["real_vs_sparse_random_advantage"] = round(real_vs_sparse, 6)

    with open(json_path, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"  Saved: {json_path}")


if __name__ == "__main__":
    main()
