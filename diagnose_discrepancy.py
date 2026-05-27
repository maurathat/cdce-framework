#!/usr/bin/env python3
"""
diagnose_discrepancy.py — find WHY the two partial-Spearman numbers differ.
Computes the partial correlation under every plausible variation so the cause
is unambiguous. Run:  python3 diagnose_discrepancy.py --csv results_2_1b.csv
"""
import argparse
import numpy as np
import pandas as pd
from scipy import stats


def resid(a, c):
    """OLS residual of a on control c (single column). pinv and lstsq agree here."""
    c = np.asarray(c, float).reshape(-1, 1)
    c1 = np.hstack([np.ones((len(c), 1)), c])      # intercept + control
    beta = np.linalg.pinv(c1) @ np.asarray(a, float)
    return np.asarray(a, float) - c1 @ beta


def partial(x, y, c, rank_first):
    x = np.asarray(x, float); y = np.asarray(y, float); c = np.asarray(c, float)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(c)
    x, y, c = x[m], y[m], c[m]
    if rank_first:
        x, y, c = stats.rankdata(x), stats.rankdata(y), stats.rankdata(c)
    xr, yr = resid(x, c), resid(y, c)
    r, p = stats.pearsonr(xr, yr)
    return r, p, len(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results_2_1b.csv")
    a = ap.parse_args()
    df = pd.read_csv(a.csv)
    print(f"columns: {list(df.columns)}\n")

    controls = [c for c in ["token_count", "prompt_token_count", "total_token_count"]
                if c in df.columns]
    x = df["n_text"]
    y = df["d_act"] if "d_act" in df.columns else df[[c for c in df.columns if c.startswith("d_act")][0]]

    print("PARTIAL CORRELATION under each variation:")
    print(f"{'control':22s} {'rank_first':10s} {'r':>8s} {'p':>8s}  n")
    print("-" * 60)
    for ctrl in controls:
        for rf in (True, False):
            r, p, n = partial(x, y, df[ctrl], rank_first=rf)
            tag = "SPEARMAN" if rf else "pearson "
            print(f"{ctrl:22s} {tag:10s} {r:+8.3f} {p:8.4f}  {n}")
    print("\nThe row matching 0.455 and the row matching 0.268 tell you EXACTLY")
    print("which (control variable, rank-or-not) each script used. That is the cause.")
    print("Correct choice = control on GENERATED token_count, rank_first=True (true Spearman).")


if __name__ == "__main__":
    main()
