#!/usr/bin/env python3
"""
check_2_1b_result.py
====================
Post-run sanity checks for the Test 2.1b result BEFORE it goes in the plan.

Answers three questions the single headline number (partial Spearman 0.455) does
NOT answer on its own:

  1. SCATTER — is the n_text vs d_act relationship a clean monotone trend, or a
     cluster plus a few influential points that the rank test happens to reward?
     (Saves a PNG you actually look at.)
  2. LAYER ROBUSTNESS — does the effect survive across last / midmean / allmean,
     or is it last-layer-only (fragile)? Requires those columns to exist in the
     CSV. If the run only produced d_act for one layer, this reports what's there
     and flags that the other modes still need running.
  3. OUTLIER INFLUENCE — leave-one-out: recompute the partial Spearman dropping
     each point in turn; if dropping any single point collapses significance,
     the result is fragile and you should say so.

Usage:
    python3 check_2_1b_result.py --csv results_2_1b.csv
    python3 check_2_1b_result.py --csv results_2_1b.csv --control token_count

Expects columns: n_text, token_count, and at least one of:
    d_act, d_act_last, d_act_midmean, d_act_allmean
(The activation script may name the single-mode column 'd_act' or 'd_act_last'.)
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression


def partial_corr(x, y, c, rank=False):
    """True residual-on-residual partial correlation of x,y controlling for c."""
    x = np.asarray(x, float); y = np.asarray(y, float); c = np.asarray(c, float)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(c)
    x, y, c = x[m], y[m], c[m]
    if len(x) < 4:
        return float("nan"), float("nan"), len(x)
    if rank:
        x, y, c = stats.rankdata(x), stats.rankdata(y), stats.rankdata(c)
    cm = c.reshape(-1, 1)
    xr = x - LinearRegression().fit(cm, x).predict(cm)
    yr = y - LinearRegression().fit(cm, y).predict(cm)
    if np.std(xr) == 0 or np.std(yr) == 0:
        return float("nan"), float("nan"), len(x)
    r, p = stats.pearsonr(xr, yr)
    return float(r), float(p), len(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results_2_1b.csv")
    ap.add_argument("--control", default="token_count")
    ap.add_argument("--x", default="n_text")
    ap.add_argument("--plot", default="check_2_1b_scatter.png")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    print(f"[load] {args.csv}: {len(df)} rows, columns={list(df.columns)}")

    # Find which d_act columns exist.
    candidates = ["d_act", "d_act_last", "d_act_midmean", "d_act_allmean"]
    act_cols = [c for c in candidates if c in df.columns]
    if not act_cols:
        raise SystemExit(f"No activation column found. Looked for {candidates}.")
    print(f"[cols] activation columns present: {act_cols}")
    if len([c for c in act_cols if c != "d_act"]) < 3 and "d_act" in act_cols:
        print("[layer] NOTE: only a single activation column ('d_act') is present. "
              "This is almost certainly ONE layer mode (likely 'last'). "
              "Layer-robustness across last/midmean/allmean is NOT yet testable from "
              "this CSV — re-run with all three modes to complete the robustness check.")

    x = df[args.x]
    ctrl = df[args.control]
    print(f"\n[ranges] {args.x}: {x.min()}–{x.max()}   "
          f"{args.control}: {ctrl.min()}–{ctrl.max()} (std={ctrl.std():.0f})")

    # ---- (2) Layer robustness: partial corr per available activation column ----
    print("\n=== partial correlations controlling for "
          f"{args.control} (per activation column) ===")
    for c in act_cols:
        pr_r, pr_p, n = partial_corr(x, df[c], ctrl, rank=False)
        ps_r, ps_p, _ = partial_corr(x, df[c], ctrl, rank=True)
        print(f"  [{c:14s}] n={n}  "
              f"partial Pearson r={pr_r:+.3f} (p={pr_p:.4f})   "
              f"partial Spearman r={ps_r:+.3f} (p={ps_p:.4f})")

    # ---- (3) Outlier influence: leave-one-out on the PRIMARY column's Spearman ----
    primary = "d_act_last" if "d_act_last" in act_cols else act_cols[0]
    print(f"\n=== leave-one-out partial Spearman on '{primary}' (fragility check) ===")
    base_r, base_p, n = partial_corr(x, df[primary], ctrl, rank=True)
    print(f"  full sample: r={base_r:+.3f} (p={base_p:.4f}), n={n}")
    loo_r = []
    loo_p = []
    idx = df.index.to_list()
    for drop in idx:
        keep = df.drop(index=drop)
        r, p, _ = partial_corr(keep[args.x], keep[primary], keep[args.control], rank=True)
        loo_r.append(r); loo_p.append(p)
    loo_r = np.array(loo_r); loo_p = np.array(loo_p)
    print(f"  leave-one-out r: min={np.nanmin(loo_r):+.3f}  max={np.nanmax(loo_r):+.3f}  "
          f"mean={np.nanmean(loo_r):+.3f}")
    print(f"  leave-one-out p: min={np.nanmin(loo_p):.4f}  max={np.nanmax(loo_p):.4f}")
    worst = int(np.nanargmax(loo_p))
    if np.nanmax(loo_p) > 0.05:
        print(f"  *** FRAGILE: dropping row {idx[worst]} pushes p above 0.05 "
              f"(p={np.nanmax(loo_p):.4f}). The result leans on that point — "
              f"report as fragile, inspect that row.")
    else:
        print(f"  ROBUST to single-point removal: p stays < 0.05 in every leave-one-out "
              f"(worst p={np.nanmax(loo_p):.4f}). Not outlier-driven.")

    # ---- (1) Scatter ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 5))
        sc = ax.scatter(x, df[primary], c=ctrl, cmap="viridis", s=40)
        ax.set_xlabel(args.x + " (text-parsed operator count)")
        ax.set_ylabel(primary + " (activation effective-dim, participation ratio)")
        ax.set_title(f"Test 2.1b: {args.x} vs {primary}\n(color = {args.control})")
        plt.colorbar(sc, label=args.control)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=120)
        print(f"\n[plot] wrote {args.plot} — OPEN IT. Look for: clean monotone rise vs. "
              f"a cluster + a few high points doing the rank work.")
    except ImportError:
        print("\n[plot] matplotlib not installed (pip install matplotlib) — "
              "skipping scatter; the leave-one-out above still tells you about fragility.")

    print("\n[verdict guide]")
    print("  - If all available layer modes agree AND leave-one-out is robust AND the")
    print("    scatter is a clean monotone -> solid 'consistent-with' result.")
    print("  - If only 'last' is present -> run midmean/allmean before claiming robustness.")
    print("  - If leave-one-out is fragile OR scatter is outlier-driven -> report as")
    print("    suggestive-but-fragile, n=45, do not overstate.")
    print("  - Either way: 'suggestive, consistent-with, arithmetic-reasoning, n=45' —")
    print("    not 'validated'.")


if __name__ == "__main__":
    main()
