#!/usr/bin/env python3
"""
E8 Specificity Test Suite — Frozen / Untrained Tests Only
==========================================================
No learned projections. Every test uses fixed linear maps (PCA, random
orthogonal) or raw geometric analysis. For each test we compare E8
against random spherical codebooks and sparsity-matched controls.

Tests
─────
1. ANGLE SPECTRUM (priority): pairwise cosine histogram of projected JEPA
   embeddings checked for E8-characteristic peaks at {-1, -0.5, 0, 0.5, 1}.
2. PCA PROJECTION: project 1280-D JEPA embeddings to 8-D via PCA, measure
   nearest-root cosine against E8 vs random codebooks.
3. RANDOM ORTHOGONAL PROJECTION: Haar-random orthogonal 1280→8, repeated
   with multiple seeds.
4. DIMENSIONAL ANOMALY SCAN: repeat PCA projection at dim 2..16, compare
   E8/random codebook gap at each dimension.
5. RANDOM CODEBOOK BASELINES for LLM operator vectors (no training).
6. LLM CROSS-COMPRESSION ANGLE PRESERVATION: do pairwise angles between
   operator vectors at budget=2000 survive compression to budget=125?
   Compare preservation against E8 vs random codebook nearest-root maps.
"""

import json
import ssl
import os
import sys
from pathlib import Path

import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())

import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance
from scipy.linalg import qr
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT_DIR / "results"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# Reuse JEPA infrastructure
sys.path.insert(0, str(ROOT_DIR / "src" / "jepa_test"))
from jepa_e8_test import (
    DEVICE, build_e8_roots, build_d4_roots,
    load_ijepa_encoder, load_cifar10_subset, extract_embeddings,
)
sys.path.insert(0, str(ROOT_DIR / "src"))
from tower import embed_verbs_r8, load_e8_roots

# ── E8 characteristic angles ───────────────────────────────────────────────
# The 240 E8 roots have pairwise inner products drawn from exactly these
# values (for unit-normalized roots): {-1, -0.5, 0, 0.5, 1}.
# Integer roots (norm √2 raw) → normalized to 1.
# Half-integer roots (norm √2 raw) → normalized to 1.
E8_CHARACTERISTIC_ANGLES = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
E8_ANGLE_TOLERANCE = 0.05  # bin width for peak detection


def random_codebook(n=240, dim=8, seed=42):
    rng = np.random.RandomState(seed)
    v = rng.randn(n, dim)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def random_orthogonal_proj(d_in, d_out, seed=0):
    """Haar-random orthogonal projection matrix (d_in → d_out)."""
    rng = np.random.RandomState(seed)
    M = rng.randn(d_in, d_out)
    Q, _ = qr(M, mode='economic')
    return Q  # shape (d_in, d_out), columns are orthonormal


def pairwise_cosines(vecs):
    """Compute all pairwise cosines (upper triangle)."""
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms == 0] = 1
    normed = vecs / norms
    G = normed @ normed.T
    idx = np.triu_indices(len(vecs), k=1)
    return G[idx]


def nearest_cos_stats(vectors, codebook):
    """Mean/std/max cosine to nearest codebook vector."""
    norms_v = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms_v[norms_v == 0] = 1
    vn = vectors / norms_v
    sim = vn @ codebook.T
    nc = sim.max(axis=1)
    return {
        "mean": float(nc.mean()),
        "std": float(nc.std()),
        "max": float(nc.max()),
        "median": float(np.median(nc)),
    }


def peak_score(cosines, targets=E8_CHARACTERISTIC_ANGLES, tol=E8_ANGLE_TOLERANCE):
    """Fraction of pairwise cosines within tol of any E8-characteristic angle."""
    hits = np.zeros(len(cosines), dtype=bool)
    for t in targets:
        hits |= np.abs(cosines - t) < tol
    return float(hits.mean())


def e8_pairwise_angle_distribution():
    """Ground truth: pairwise cosines among the 240 E8 roots."""
    roots = build_e8_roots()
    return pairwise_cosines(roots)


# ═══════════════════════════════════════════════════════════════════════════
# TEST 1: ANGLE SPECTRUM (priority)
# ═══════════════════════════════════════════════════════════════════════════

def test_angle_spectrum(embeddings_np):
    """Check if JEPA embedding pairwise angles show E8 peaks."""
    print("\n" + "=" * 70)
    print("TEST 1: ANGLE SPECTRUM — E8-characteristic peaks at {-1,-0.5,0,0.5,1}")
    print("=" * 70)

    results = {}

    # E8 ground truth
    e8_angles = e8_pairwise_angle_distribution()
    e8_peak = peak_score(e8_angles)
    results["e8_roots_peak_score"] = e8_peak
    print(f"  E8 roots pairwise peak score:      {e8_peak:.4f}")

    # Random codebook ground truth
    rcb = random_codebook(240, 8, seed=0)
    rcb_angles = pairwise_cosines(rcb)
    rcb_peak = peak_score(rcb_angles)
    results["random_cb_peak_score"] = rcb_peak
    print(f"  Random codebook pairwise peak:     {rcb_peak:.4f}")

    # PCA → 8D
    mean = embeddings_np.mean(axis=0)
    centered = embeddings_np - mean
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    proj_8d = centered @ Vt[:8].T  # (N, 8)
    jepa_angles = pairwise_cosines(proj_8d)
    jepa_peak = peak_score(jepa_angles)
    results["jepa_pca8d_peak_score"] = jepa_peak
    print(f"  JEPA PCA→8D pairwise peak:         {jepa_peak:.4f}")

    # Random orthogonal → 8D (average over 5 seeds)
    orth_peaks = []
    for seed in range(5):
        Q = random_orthogonal_proj(embeddings_np.shape[1], 8, seed=seed)
        proj = embeddings_np @ Q
        angles = pairwise_cosines(proj)
        orth_peaks.append(peak_score(angles))
    results["jepa_orth8d_peak_mean"] = float(np.mean(orth_peaks))
    results["jepa_orth8d_peak_std"] = float(np.std(orth_peaks))
    print(f"  JEPA random-orth→8D peak (5 seeds): {np.mean(orth_peaks):.4f} ± {np.std(orth_peaks):.4f}")

    # Dense random vectors in 8D
    rand8 = np.random.RandomState(77).randn(1000, 8)
    rand8 = rand8 / np.linalg.norm(rand8, axis=1, keepdims=True)
    rand_angles = pairwise_cosines(rand8)
    rand_peak = peak_score(rand_angles)
    results["random_8d_peak_score"] = rand_peak
    print(f"  Random unit vectors 8D peak:        {rand_peak:.4f}")

    # Statistical test: JEPA vs random
    ks_stat, ks_p = ks_2samp(jepa_angles, rand_angles)
    w_dist = wasserstein_distance(jepa_angles, rand_angles)
    results["ks_jepa_vs_random"] = {"statistic": float(ks_stat), "p_value": float(ks_p)}
    results["wasserstein_jepa_vs_random"] = float(w_dist)
    print(f"  KS test (JEPA vs random): stat={ks_stat:.4f}, p={ks_p:.2e}")
    print(f"  Wasserstein (JEPA vs random): {w_dist:.4f}")

    # Verdict
    if jepa_peak > rand_peak * 1.2 and jepa_peak > rcb_peak * 1.1:
        verdict = "PASS"
    else:
        verdict = "NULL"
    results["verdict"] = verdict
    print(f"  VERDICT: {verdict}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    axes[0, 0].hist(e8_angles, bins=100, density=True, alpha=0.7, color="#e63946", label="E8 roots")
    for a in E8_CHARACTERISTIC_ANGLES:
        axes[0, 0].axvline(a, color="black", ls="--", alpha=0.5, lw=0.8)
    axes[0, 0].set_title(f"E8 Root Pairwise Cosines (peak={e8_peak:.3f})")
    axes[0, 0].set_xlabel("Cosine similarity")
    axes[0, 0].legend()

    axes[0, 1].hist(jepa_angles, bins=100, density=True, alpha=0.7, color="#457b9d", label="JEPA PCA→8D")
    for a in E8_CHARACTERISTIC_ANGLES:
        axes[0, 1].axvline(a, color="black", ls="--", alpha=0.5, lw=0.8)
    axes[0, 1].set_title(f"JEPA PCA→8D Pairwise Cosines (peak={jepa_peak:.3f})")
    axes[0, 1].set_xlabel("Cosine similarity")
    axes[0, 1].legend()

    axes[1, 0].hist(rand_angles, bins=100, density=True, alpha=0.7, color="#a8dadc", label="Random 8D")
    for a in E8_CHARACTERISTIC_ANGLES:
        axes[1, 0].axvline(a, color="black", ls="--", alpha=0.5, lw=0.8)
    axes[1, 0].set_title(f"Random 8D Pairwise Cosines (peak={rand_peak:.3f})")
    axes[1, 0].set_xlabel("Cosine similarity")
    axes[1, 0].legend()

    # Overlay comparison
    axes[1, 1].hist(e8_angles, bins=100, density=True, alpha=0.5, color="#e63946", label=f"E8 ({e8_peak:.3f})")
    axes[1, 1].hist(jepa_angles, bins=100, density=True, alpha=0.5, color="#457b9d", label=f"JEPA ({jepa_peak:.3f})")
    axes[1, 1].hist(rand_angles, bins=100, density=True, alpha=0.5, color="#a8dadc", label=f"Random ({rand_peak:.3f})")
    for a in E8_CHARACTERISTIC_ANGLES:
        axes[1, 1].axvline(a, color="black", ls="--", alpha=0.5, lw=0.8)
    axes[1, 1].set_title("Overlay: E8 vs JEPA vs Random")
    axes[1, 1].set_xlabel("Cosine similarity")
    axes[1, 1].legend()

    plt.suptitle("Test 1: Angle Spectrum — E8 Characteristic Peaks", fontsize=14, y=1.01)
    plt.tight_layout()
    fig.savefig(str(PLOTS_DIR / "angle_spectrum.png"), dpi=150, bbox_inches="tight")
    print(f"  Saved: {PLOTS_DIR / 'angle_spectrum.png'}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# TEST 2: PCA PROJECTION → nearest root
# ═══════════════════════════════════════════════════════════════════════════

def test_pca_projection(embeddings_np):
    """PCA to 8D, measure nearest E8 root vs nearest random CB vector."""
    print("\n" + "=" * 70)
    print("TEST 2: PCA PROJECTION — frozen 1280→8D, nearest root cosine")
    print("=" * 70)

    e8 = build_e8_roots()
    mean = embeddings_np.mean(axis=0)
    centered = embeddings_np - mean
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)
    proj = centered @ Vt[:8].T

    results = {}

    # E8
    e8_stats = nearest_cos_stats(proj, e8)
    results["pca_vs_e8"] = e8_stats
    print(f"  PCA→8D vs E8:        mean={e8_stats['mean']:.4f} ± {e8_stats['std']:.4f}")

    # Random codebooks
    cb_means = []
    for seed in range(10):
        cb = random_codebook(240, 8, seed=seed)
        s = nearest_cos_stats(proj, cb)
        cb_means.append(s["mean"])
    results["pca_vs_random_cb_mean"] = float(np.mean(cb_means))
    results["pca_vs_random_cb_std"] = float(np.std(cb_means))
    print(f"  PCA→8D vs random CB: mean={np.mean(cb_means):.4f} ± {np.std(cb_means):.4f} (10 seeds)")

    delta = e8_stats["mean"] - np.mean(cb_means)
    results["e8_advantage"] = float(delta)

    verdict = "PASS" if delta > 0.03 else "NULL"
    results["verdict"] = verdict
    print(f"  E8 advantage: {delta:+.4f}  →  {verdict}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# TEST 3: RANDOM ORTHOGONAL PROJECTION
# ═══════════════════════════════════════════════════════════════════════════

def test_random_orthogonal(embeddings_np):
    """Haar-random orthogonal projection 1280→8, repeated 20 seeds."""
    print("\n" + "=" * 70)
    print("TEST 3: RANDOM ORTHOGONAL — Haar-random 1280→8D, 20 seeds")
    print("=" * 70)

    e8 = build_e8_roots()
    n_seeds = 20
    e8_means = []
    cb_means = []

    for seed in range(n_seeds):
        Q = random_orthogonal_proj(embeddings_np.shape[1], 8, seed=seed)
        proj = embeddings_np @ Q

        e8_s = nearest_cos_stats(proj, e8)
        e8_means.append(e8_s["mean"])

        cb = random_codebook(240, 8, seed=seed + 1000)
        cb_s = nearest_cos_stats(proj, cb)
        cb_means.append(cb_s["mean"])

    results = {
        "e8_mean": float(np.mean(e8_means)),
        "e8_std": float(np.std(e8_means)),
        "random_cb_mean": float(np.mean(cb_means)),
        "random_cb_std": float(np.std(cb_means)),
    }
    delta = results["e8_mean"] - results["random_cb_mean"]
    results["e8_advantage"] = float(delta)

    print(f"  Orth→8D vs E8:        {results['e8_mean']:.4f} ± {results['e8_std']:.4f}")
    print(f"  Orth→8D vs random CB: {results['random_cb_mean']:.4f} ± {results['random_cb_std']:.4f}")
    print(f"  E8 advantage: {delta:+.4f}")

    verdict = "PASS" if delta > 0.03 else "NULL"
    results["verdict"] = verdict
    print(f"  VERDICT: {verdict}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# TEST 4: DIMENSIONAL ANOMALY SCAN
# ═══════════════════════════════════════════════════════════════════════════

def test_dimensional_scan(embeddings_np):
    """PCA project to dim=2..16, look for anomaly at dim=8."""
    print("\n" + "=" * 70)
    print("TEST 4: DIMENSIONAL ANOMALY SCAN — PCA to dim 2..16")
    print("=" * 70)

    e8 = build_e8_roots()
    mean = embeddings_np.mean(axis=0)
    centered = embeddings_np - mean
    U, S, Vt = np.linalg.svd(centered, full_matrices=False)

    dims = list(range(2, 17))
    e8_gaps = []  # e8 advantage over random CB at each dim
    e8_scores = []
    cb_scores = []

    for dim in dims:
        proj = centered @ Vt[:dim].T

        if dim <= 8:
            # Use E8 roots truncated/projected to this dim
            e8_sub = e8[:, :dim]
            e8_sub = e8_sub / (np.linalg.norm(e8_sub, axis=1, keepdims=True) + 1e-8)
        else:
            # Pad E8 roots to this dim
            e8_sub = np.zeros((240, dim))
            e8_sub[:, :8] = e8
            e8_sub = e8_sub / (np.linalg.norm(e8_sub, axis=1, keepdims=True) + 1e-8)

        e8_s = nearest_cos_stats(proj, e8_sub)

        cb_avgs = []
        for seed in range(5):
            cb = random_codebook(240, dim, seed=seed)
            cb_avgs.append(nearest_cos_stats(proj, cb)["mean"])

        gap = e8_s["mean"] - np.mean(cb_avgs)
        e8_gaps.append(gap)
        e8_scores.append(e8_s["mean"])
        cb_scores.append(np.mean(cb_avgs))

        marker = " ◄◄◄" if dim == 8 else ""
        print(f"  dim={dim:>2}: E8={e8_s['mean']:.4f}  random_CB={np.mean(cb_avgs):.4f}  "
              f"gap={gap:+.4f}{marker}")

    results = {
        "dims": dims,
        "e8_scores": e8_scores,
        "cb_scores": cb_scores,
        "gaps": e8_gaps,
        "gap_at_8": float(e8_gaps[dims.index(8)]),
    }

    # Is dim=8 a local maximum in the gap?
    idx8 = dims.index(8)
    neighbors = [e8_gaps[i] for i in range(max(0, idx8-2), min(len(e8_gaps), idx8+3)) if i != idx8]
    is_local_max = e8_gaps[idx8] > max(neighbors) if neighbors else False
    results["dim8_is_local_max"] = is_local_max

    verdict = "PASS" if is_local_max and e8_gaps[idx8] > 0.01 else "NULL"
    results["verdict"] = verdict
    print(f"  Dim=8 gap: {e8_gaps[idx8]:+.4f}, local max: {is_local_max}  →  {verdict}")

    # Plot
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    ax.plot(dims, e8_scores, 'o-', color="#e63946", label="E8 roots", linewidth=2)
    ax.plot(dims, cb_scores, 's-', color="#457b9d", label="Random CB (avg)", linewidth=2)
    ax.fill_between(dims, e8_scores, cb_scores, alpha=0.15, color="#2a9d8f")
    ax.axvline(8, color="black", ls="--", alpha=0.4, label="dim=8 (E8)")
    ax.set_xlabel("Projection dimension", fontsize=12)
    ax.set_ylabel("Mean cosine to nearest codebook vector", fontsize=12)
    ax.set_title("Dimensional Anomaly Scan: E8 vs Random CB", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(str(PLOTS_DIR / "dimensional_scan.png"), dpi=150)
    print(f"  Saved: {PLOTS_DIR / 'dimensional_scan.png'}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# TEST 5: LLM OPERATOR RANDOM CODEBOOK BASELINES
# ═══════════════════════════════════════════════════════════════════════════

def test_llm_baselines():
    """LLM operator vectors vs E8/random codebooks, with sparsity controls."""
    print("\n" + "=" * 70)
    print("TEST 5: LLM OPERATOR BASELINES — E8 vs random CB vs sparsity match")
    print("=" * 70)

    # Load tower analysis
    tower_files = sorted(RESULTS_DIR.glob("tower_analysis_*.json"))
    if not tower_files:
        print("  SKIP: no tower analysis found")
        return {"verdict": "SKIP"}

    with open(tower_files[-1]) as f:
        tower = json.load(f)

    comps = tower["e8_comparisons"]
    vectors = []
    sparsities = []
    for c in comps:
        v = embed_verbs_r8(c["verbs"])
        if np.linalg.norm(v) > 0:
            vectors.append(v)
            sparsities.append(int(np.sum(v > 0)))
    vectors = np.array(vectors)
    n = len(vectors)

    e8 = build_e8_roots()

    # Real operators → E8
    real_e8 = nearest_cos_stats(vectors, e8)
    print(f"  Real operators → E8:              mean={real_e8['mean']:.4f}")

    # Real operators → random codebooks (10 seeds)
    real_cb_means = []
    for seed in range(10):
        cb = random_codebook(240, 8, seed=seed)
        real_cb_means.append(nearest_cos_stats(vectors, cb)["mean"])
    real_cb_avg = float(np.mean(real_cb_means))
    print(f"  Real operators → random CB (10):  mean={real_cb_avg:.4f} ± {np.std(real_cb_means):.4f}")

    # Sparsity-matched random → E8 (10 seeds)
    sparse_e8_means = []
    rng = np.random.RandomState(42)
    for seed in range(10):
        sv = []
        for i in range(n):
            sp = sparsities[i % len(sparsities)]
            v = np.zeros(8)
            active = rng.choice(8, size=sp, replace=False)
            v[active] = rng.rand(sp)
            norm = np.linalg.norm(v)
            if norm > 0:
                v /= norm
            sv.append(v)
        sv = np.array(sv)
        sparse_e8_means.append(nearest_cos_stats(sv, e8)["mean"])
    sparse_e8_avg = float(np.mean(sparse_e8_means))
    print(f"  Sparsity-matched rand → E8 (10): mean={sparse_e8_avg:.4f} ± {np.std(sparse_e8_means):.4f}")

    # Dense random → E8
    dense = rng.randn(n, 8)
    dense = dense / np.linalg.norm(dense, axis=1, keepdims=True)
    dense_e8 = nearest_cos_stats(dense, e8)
    print(f"  Dense random → E8:               mean={dense_e8['mean']:.4f}")

    results = {
        "n_vectors": n,
        "mean_sparsity": float(np.mean(sparsities)),
        "real_vs_e8": real_e8["mean"],
        "real_vs_random_cb": real_cb_avg,
        "sparse_random_vs_e8": sparse_e8_avg,
        "dense_random_vs_e8": dense_e8["mean"],
        "e8_advantage": float(real_e8["mean"] - real_cb_avg),
        "real_vs_sparse_advantage": float(real_e8["mean"] - sparse_e8_avg),
    }

    # E8 is specific if real operators prefer E8 over random codebook
    # AND real operators score higher than sparsity-matched random on E8
    e8_spec = results["e8_advantage"] > 0.02
    struct_spec = results["real_vs_sparse_advantage"] > 0.02
    verdict = "PASS" if (e8_spec and struct_spec) else "NULL"
    results["verdict"] = verdict
    print(f"  E8 advantage: {results['e8_advantage']:+.4f}, "
          f"structure advantage: {results['real_vs_sparse_advantage']:+.4f}  →  {verdict}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# TEST 6: LLM CROSS-COMPRESSION ANGLE PRESERVATION
# ═══════════════════════════════════════════════════════════════════════════

def test_cross_compression_angles():
    """Do pairwise angles between operator vectors survive compression?"""
    print("\n" + "=" * 70)
    print("TEST 6: CROSS-COMPRESSION ANGLE PRESERVATION")
    print("=" * 70)

    # Load all experiment data
    exp_files = sorted(RESULTS_DIR.glob("experiment_*.json"))
    if not exp_files:
        print("  SKIP: no experiment files")
        return {"verdict": "SKIP"}

    # Aggregate all runs
    all_runs = []
    for ef in exp_files:
        with open(ef) as f:
            data = json.load(f)
        all_runs.extend(data.get("runs", []))

    if not all_runs:
        print("  SKIP: no runs found")
        return {"verdict": "SKIP"}

    # Group by (model, task_id), get vectors at highest and lowest budget
    budgets = sorted(set(r["budget"] for r in all_runs))
    if len(budgets) < 2:
        print("  SKIP: need at least 2 budget levels")
        return {"verdict": "SKIP"}

    high_budget = max(budgets)
    low_budget = min(budgets)
    print(f"  Comparing budget={high_budget} vs budget={low_budget}")

    # Build paired vectors
    high_vecs = {}
    low_vecs = {}
    for r in all_runs:
        key = (r["model"], r["task_id"])
        verbs = r.get("unique_verbs", [])
        v = embed_verbs_r8(verbs)
        if np.linalg.norm(v) == 0:
            continue
        if r["budget"] == high_budget:
            high_vecs[key] = v
        elif r["budget"] == low_budget:
            low_vecs[key] = v

    # Find pairs that exist at both budgets
    shared_keys = sorted(set(high_vecs.keys()) & set(low_vecs.keys()))
    if len(shared_keys) < 3:
        print(f"  SKIP: only {len(shared_keys)} shared (model, task) pairs")
        return {"verdict": "SKIP"}

    print(f"  Shared (model, task) pairs: {len(shared_keys)}")

    high_mat = np.array([high_vecs[k] for k in shared_keys])
    low_mat = np.array([low_vecs[k] for k in shared_keys])

    # Pairwise angles at each budget level
    high_angles = pairwise_cosines(high_mat)
    low_angles = pairwise_cosines(low_mat)

    # Angle preservation: correlation between high-budget and low-budget pairwise cosines
    corr = float(np.corrcoef(high_angles, low_angles)[0, 1])
    print(f"  Pairwise angle correlation (high vs low budget): {corr:.4f}")

    # Now: do E8-nearest-root assignments preserve angles better than random CB?
    e8 = build_e8_roots()

    def nearest_root_vectors(vecs, codebook):
        sim = vecs @ codebook.T
        idx = sim.argmax(axis=1)
        return codebook[idx]

    high_e8 = nearest_root_vectors(high_mat, e8)
    low_e8 = nearest_root_vectors(low_mat, e8)
    e8_high_angles = pairwise_cosines(high_e8)
    e8_low_angles = pairwise_cosines(low_e8)
    e8_corr = float(np.corrcoef(e8_high_angles, e8_low_angles)[0, 1])
    print(f"  E8-quantized angle correlation:     {e8_corr:.4f}")

    cb_corrs = []
    for seed in range(10):
        cb = random_codebook(240, 8, seed=seed)
        h_cb = nearest_root_vectors(high_mat, cb)
        l_cb = nearest_root_vectors(low_mat, cb)
        h_a = pairwise_cosines(h_cb)
        l_a = pairwise_cosines(l_cb)
        if np.std(h_a) > 0 and np.std(l_a) > 0:
            cb_corrs.append(float(np.corrcoef(h_a, l_a)[0, 1]))
    cb_corr_avg = float(np.mean(cb_corrs)) if cb_corrs else 0.0
    print(f"  Random-CB-quantized angle corr:     {cb_corr_avg:.4f} ± {np.std(cb_corrs):.4f}")

    results = {
        "n_pairs": len(shared_keys),
        "raw_angle_correlation": corr,
        "e8_quantized_correlation": e8_corr,
        "random_cb_quantized_correlation": cb_corr_avg,
        "e8_advantage": float(e8_corr - cb_corr_avg),
    }

    verdict = "PASS" if results["e8_advantage"] > 0.05 else "NULL"
    results["verdict"] = verdict
    print(f"  E8 advantage: {results['e8_advantage']:+.4f}  →  {verdict}")

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].scatter(high_angles, low_angles, alpha=0.3, s=5, color="#457b9d")
    axes[0].plot([-1, 1], [-1, 1], 'k--', alpha=0.3)
    axes[0].set_title(f"Raw angles (r={corr:.3f})")
    axes[0].set_xlabel(f"Budget {high_budget}")
    axes[0].set_ylabel(f"Budget {low_budget}")

    axes[1].scatter(e8_high_angles, e8_low_angles, alpha=0.3, s=5, color="#e63946")
    axes[1].plot([-1, 1], [-1, 1], 'k--', alpha=0.3)
    axes[1].set_title(f"E8-quantized (r={e8_corr:.3f})")
    axes[1].set_xlabel(f"Budget {high_budget}")
    axes[1].set_ylabel(f"Budget {low_budget}")

    h_cb0 = nearest_root_vectors(high_mat, random_codebook(240, 8, seed=0))
    l_cb0 = nearest_root_vectors(low_mat, random_codebook(240, 8, seed=0))
    axes[2].scatter(pairwise_cosines(h_cb0), pairwise_cosines(l_cb0), alpha=0.3, s=5, color="#2a9d8f")
    axes[2].plot([-1, 1], [-1, 1], 'k--', alpha=0.3)
    axes[2].set_title(f"Random-CB-quantized (r={cb_corr_avg:.3f})")
    axes[2].set_xlabel(f"Budget {high_budget}")
    axes[2].set_ylabel(f"Budget {low_budget}")

    plt.suptitle("Cross-Compression Angle Preservation", fontsize=14)
    plt.tight_layout()
    fig.savefig(str(PLOTS_DIR / "cross_compression_angles.png"), dpi=150)
    print(f"  Saved: {PLOTS_DIR / 'cross_compression_angles.png'}")

    return results


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("E8 SPECIFICITY TEST SUITE — FROZEN / UNTRAINED ONLY")
    print("=" * 70)

    # Load JEPA embeddings
    print("\n[0] Loading I-JEPA encoder + CIFAR-10...")
    model, embed_dim = load_ijepa_encoder()
    loader = load_cifar10_subset(1000)
    embeddings = extract_embeddings(model, loader)
    embeddings_np = embeddings.numpy()
    print(f"  Embeddings: {embeddings_np.shape}")

    # Free GPU memory
    del model
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

    all_results = {}

    # Run all tests
    all_results["test1_angle_spectrum"] = test_angle_spectrum(embeddings_np)
    all_results["test2_pca_projection"] = test_pca_projection(embeddings_np)
    all_results["test3_random_orthogonal"] = test_random_orthogonal(embeddings_np)
    all_results["test4_dimensional_scan"] = test_dimensional_scan(embeddings_np)
    all_results["test5_llm_baselines"] = test_llm_baselines()
    all_results["test6_cross_compression"] = test_cross_compression_angles()

    # Final summary
    print(f"\n{'=' * 70}")
    print("FINAL SUMMARY")
    print(f"{'─' * 70}")
    print(f"  {'Test':<45} {'Verdict':>8}")
    print(f"  {'─' * 55}")
    for name, res in all_results.items():
        v = res.get("verdict", "???")
        label = name.replace("_", " ").replace("test", "Test")
        print(f"  {label:<45} {v:>8}")
    n_pass = sum(1 for r in all_results.values() if r.get("verdict") == "PASS")
    n_null = sum(1 for r in all_results.values() if r.get("verdict") == "NULL")
    n_skip = sum(1 for r in all_results.values() if r.get("verdict") == "SKIP")
    print(f"  {'─' * 55}")
    print(f"  PASS: {n_pass}  NULL: {n_null}  SKIP: {n_skip}")
    print(f"{'=' * 70}")

    # Save JSON (strip large arrays)
    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()
                    if not (isinstance(v, list) and len(v) > 50)}
        if isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, bool):
            return int(obj)
        return obj

    json_path = RESULTS_DIR / "e8_specificity_suite.json"
    with open(json_path, "w") as f:
        json.dump(clean(all_results), f, indent=2)
    print(f"\nSaved: {json_path}")
    print(f"Plots: {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
