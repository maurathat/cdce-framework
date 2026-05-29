#!/usr/bin/env python3
"""
analyze_section_6_4_corpus.py — Reproduce all numerical claims in §6.4 of the
HELM paper from the v0.4.1 content-addressed corpus.

Given the manifest JSON files and the published routing_store/, this script:
  1. Loads and verifies every trace_κ referenced in the manifests.
  2. Re-scores each trace from stored output (no API calls).
  3. Produces:
     - Per-cell success rates with Wilson 95% CIs
     - Specific lift (correct − shuffled) with CIs
     - Format-vs-method decomposition for opt_routing cells
     - Replay-agreement table (unique trace_κs vs unique grading outcomes)
  4. Outputs results as JSON and markdown.

Usage:
    python3 analyze_section_6_4_corpus.py \\
        manifests/section_6_4_v04_20260528_152116.json \\
        manifests/section_6_4_v04_20260528_152507.json \\
        manifests/section_6_4_v04_20260528_155940.json

All manifests are merged; duplicate trace_κs are deduplicated. The script is
deterministic and reproducible given the same manifests + routing_store.
"""

import argparse, hashlib, json, math, os, sys, time
from collections import defaultdict

from cdce_routing_objects import verify, TRACE_FIELDS, JsonStore
from src.scorers import (
    score as score_fn, _extract_tour_distance, _numbers_from_text,
)

OPT_ROUTING_OPTIMAL = 70

SCORERS_PATH = os.path.join(os.path.dirname(__file__), "src", "scorers.py")


def scorer_hash():
    """SHA-256 of src/scorers.py contents. Pins the grading logic version."""
    with open(SCORERS_PATH, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# ---- statistics ----

def wilson_ci(successes, n, z=1.96):
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return None, None, None
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return p, max(0.0, center - spread), min(1.0, center + spread)


def lift_ci(p1, n1, p2, n2, z=1.96):
    """Approximate CI on the difference p1 - p2 (independent binomials)."""
    if n1 == 0 or n2 == 0:
        return None, None, None
    diff = p1 - p2
    se = math.sqrt(p1 * (1 - p1) / n1 + p2 * (1 - p2) / n2)
    return diff, diff - z * se, diff + z * se


# ---- data loading ----

def load_manifests(paths):
    """Load and merge results from multiple manifest files."""
    all_results = []
    manifest_meta = []
    for path in paths:
        with open(path) as f:
            m = json.load(f)
        manifest_meta.append({
            "path": path,
            "run_id": m.get("run_id"),
            "schema_version": m.get("schema_version"),
            "n_results": len(m.get("results", [])),
        })
        for r in m.get("results", []):
            if r.get("status") == "ok" and r.get("trace_kappa"):
                all_results.append(r)
    return all_results, manifest_meta


def load_and_verify_traces(results, store):
    """Load traces from store, verify each, re-score from output."""
    rows = []
    n_verified = 0
    n_failed = 0

    for r in results:
        tk = r["trace_kappa"]
        trace = store.get(tk)
        if trace is None:
            n_failed += 1
            continue

        ok = verify(trace, TRACE_FIELDS, "trace_kappa")
        if not ok:
            n_failed += 1
            continue
        n_verified += 1

        output = trace.get("output", "")
        task_id = r["task_id"]
        model = r["model"]
        condition = r["condition"]
        budget = r["budget"]

        # Re-score from output
        input_obj = store.get(trace["input_kappa"]) or {}
        sc, su, gr = score_fn(task_id, input_obj, output)

        # opt_routing format-vs-method decomposition
        parsed_dist = None
        failure_mode = None
        if task_id == "opt_routing":
            parsed_dist = _extract_tour_distance(output)
            if parsed_dist is None:
                failure_mode = "format"
            elif abs(parsed_dist - OPT_ROUTING_OPTIMAL) < 0.5:
                failure_mode = "optimal"
            else:
                failure_mode = "suboptimal"

        rows.append({
            "trace_kappa": tk,
            "task_id": task_id,
            "model": model,
            "condition": condition,
            "budget": budget,
            "score": sc,
            "success": su,
            "graded": gr,
            "output_len": len(output),
            "parsed_dist": parsed_dist,
            "failure_mode": failure_mode,
        })

    return rows, n_verified, n_failed


# ---- analysis ----

def build_cell_stats(rows):
    """Group rows by cell, compute per-cell statistics."""
    cells = defaultdict(list)
    for r in rows:
        key = (r["task_id"], r["model"], r["condition"], r["budget"])
        cells[key].append(r)

    stats = {}
    for key, reps in cells.items():
        n = len(reps)
        graded = [r for r in reps if r["graded"]]
        n_graded = len(graded)
        n_success = sum(1 for r in graded if r["success"])

        p, ci_lo, ci_hi = wilson_ci(n_success, n_graded)

        unique_kappas = len(set(r["trace_kappa"] for r in reps))
        unique_grades = len(set((r["score"], r["success"]) for r in reps))

        mean_output_len = sum(r["output_len"] for r in reps) / n if n else 0

        # opt_routing decomposition
        n_format_fail = sum(1 for r in reps if r["failure_mode"] == "format")
        n_suboptimal = sum(1 for r in reps if r["failure_mode"] == "suboptimal")
        n_optimal = sum(1 for r in reps if r["failure_mode"] == "optimal")
        parsed_dists = [r["parsed_dist"] for r in reps if r["parsed_dist"] is not None]

        stats[key] = {
            "task_id": key[0], "model": key[1], "condition": key[2], "budget": key[3],
            "n_reps": n, "n_graded": n_graded, "n_success": n_success,
            "success_rate": p, "ci_lo": ci_lo, "ci_hi": ci_hi,
            "unique_kappas": unique_kappas, "unique_grades": unique_grades,
            "mean_output_len": mean_output_len,
            "n_format_fail": n_format_fail, "n_suboptimal": n_suboptimal,
            "n_optimal": n_optimal, "parsed_dists": parsed_dists,
        }

    return stats


def compute_specific_lifts(stats):
    """Compute specific lift (correct − shuffled) for each (task, model, budget)."""
    lifts = {}
    for key, s in stats.items():
        if s["condition"] != "correct":
            continue
        shuf_key = (s["task_id"], s["model"], "shuffled", s["budget"])
        shuf = stats.get(shuf_key)
        if shuf is None or shuf["success_rate"] is None or s["success_rate"] is None:
            continue

        diff, lo, hi = lift_ci(
            s["success_rate"], s["n_graded"],
            shuf["success_rate"], shuf["n_graded"],
        )
        if diff is None:
            continue

        lift_key = (s["task_id"], s["model"], s["budget"])
        lifts[lift_key] = {
            "task_id": s["task_id"], "model": s["model"], "budget": s["budget"],
            "correct_rate": s["success_rate"], "correct_n": s["n_graded"],
            "shuffled_rate": shuf["success_rate"], "shuffled_n": shuf["n_graded"],
            "specific_lift": diff, "lift_ci_lo": lo, "lift_ci_hi": hi,
        }
    return lifts


# ---- output formatting ----

def fmt_rate(s):
    """Format a cell stat as 'pct [lo, hi]' or 'ng'."""
    if s["success_rate"] is None:
        return "ng"
    return f"{s['success_rate']:.0%} [{s['ci_lo']:.0%},{s['ci_hi']:.0%}]"


def fmt_rate_short(s):
    if s["success_rate"] is None:
        return "ng"
    return f"{s['success_rate']:.0%}"


def generate_markdown(stats, lifts, manifest_meta, n_verified, n_failed):
    """Generate markdown report."""
    lines = []
    lines.append("# §6.4 Cross-Model Strategy Transfer — v0.4.1 Corpus Analysis")
    lines.append("")
    lines.append("Generated by `analyze_section_6_4_corpus.py`. All numbers derived")
    lines.append("from content-addressed traces; every trace_κ verified by recompute.")
    lines.append("")
    sh = scorer_hash()
    lines.append(f"**Analysis date:** {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}")
    lines.append(f"**Scorer hash:** `{sh[:16]}…` (SHA-256 of `src/scorers.py`)")
    lines.append(f"**Traces:** {n_verified} verified, {n_failed} failed verification")
    lines.append(f"**Manifests:** {len(manifest_meta)}")
    for m in manifest_meta:
        lines.append(f"  - `{m['path']}` ({m['n_results']} results, {m['schema_version']})")
    lines.append("")

    # ---- Success rates with CIs ----
    for task_id in ["opt_routing", "trans_nl_code"]:
        task_stats = {k: v for k, v in stats.items() if k[0] == task_id}
        if not task_stats:
            continue
        models = sorted(set(k[1] for k in task_stats))
        budgets = sorted(set(k[3] for k in task_stats), reverse=True)

        lines.append(f"## {task_id} — Success Rates (Wilson 95% CI)")
        lines.append("")
        header = f"| Model | Condition |"
        for b in budgets:
            header += f" {b} |"
        lines.append(header)
        sep = "|-------|-----------|"
        for b in budgets:
            sep += "------|"
        lines.append(sep)

        for model in models:
            for cond in ["no_strategy", "shuffled", "correct"]:
                row = f"| {model} | {cond} |"
                for b in budgets:
                    s = stats.get((task_id, model, cond, b))
                    row += f" {fmt_rate(s) if s else '—'} |"
                lines.append(row)
        lines.append("")

    # ---- Specific lift ----
    lines.append("## Specific Lift (correct − shuffled) with 95% CI")
    lines.append("")
    lines.append("| Task | Model | Budget | Lift | CI | Correct | Shuffled |")
    lines.append("|------|-------|--------|------|----|---------|----------|")
    for lk in sorted(lifts.keys()):
        l = lifts[lk]
        ci_str = f"[{l['lift_ci_lo']:+.0%}, {l['lift_ci_hi']:+.0%}]"
        lines.append(
            f"| {l['task_id']} | {l['model']} | {l['budget']} | "
            f"**{l['specific_lift']:+.0%}** | {ci_str} | "
            f"{l['correct_rate']:.0%} (n={l['correct_n']}) | "
            f"{l['shuffled_rate']:.0%} (n={l['shuffled_n']}) |"
        )
    lines.append("")

    # ---- Format vs method (opt_routing only) ----
    lines.append("## opt_routing — Format vs Method Decomposition")
    lines.append("")
    lines.append("Failure modes: **optimal** = stated distance == 70, "
                 "**suboptimal** = stated distance != 70, "
                 "**format** = no parseable distance in output.")
    lines.append("")
    lines.append("| Model | Condition | Budget | n | Optimal | Suboptimal | Format |")
    lines.append("|-------|-----------|--------|---|---------|------------|--------|")
    for key in sorted(stats.keys()):
        s = stats[key]
        if s["task_id"] != "opt_routing":
            continue
        if s["n_reps"] < 2:
            continue
        lines.append(
            f"| {s['model']} | {s['condition']} | {s['budget']} | "
            f"{s['n_reps']} | {s['n_optimal']} | {s['n_suboptimal']} | "
            f"{s['n_format_fail']} |"
        )
    lines.append("")

    # ---- Replay agreement ----
    lines.append("## Replay Agreement (per cell)")
    lines.append("")
    lines.append("| Task | Model | Condition | Budget | Reps | Unique κ | "
                 "Unique Grade | Success % |")
    lines.append("|------|-------|-----------|--------|------|----------|"
                 "-------------|-----------|")
    for key in sorted(stats.keys()):
        s = stats[key]
        if s["n_reps"] < 2:
            continue
        sr = fmt_rate_short(s)
        lines.append(
            f"| {s['task_id']} | {s['model']} | {s['condition']} | "
            f"{s['budget']} | {s['n_reps']} | {s['unique_kappas']}/{s['n_reps']} | "
            f"{s['unique_grades']} | {sr} |"
        )
    lines.append("")

    # ---- Mean output length ----
    lines.append("## Mean Output Length (chars)")
    lines.append("")
    for task_id in ["opt_routing", "trans_nl_code"]:
        task_stats = {k: v for k, v in stats.items() if k[0] == task_id}
        if not task_stats:
            continue
        models = sorted(set(k[1] for k in task_stats))
        budgets = sorted(set(k[3] for k in task_stats), reverse=True)

        lines.append(f"### {task_id}")
        lines.append("")
        header = "| Model | Condition |"
        for b in budgets:
            header += f" {b} |"
        lines.append(header)
        sep = "|-------|-----------|"
        for b in budgets:
            sep += "------|"
        lines.append(sep)

        for model in models:
            for cond in ["no_strategy", "shuffled", "correct"]:
                row = f"| {model} | {cond} |"
                for b in budgets:
                    s = stats.get((task_id, model, cond, b))
                    if s:
                        row += f" {s['mean_output_len']:.0f} |"
                    else:
                        row += " — |"
                lines.append(row)
        lines.append("")

    return "\n".join(lines)


def generate_json(stats, lifts, manifest_meta, n_verified, n_failed):
    """Generate JSON output for downstream consumers."""
    return {
        "analysis_version": "section_6_4_v041",
        "analysis_date": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scorer_version": scorer_hash(),
        "n_verified": n_verified,
        "n_failed": n_failed,
        "manifests": manifest_meta,
        "cells": {f"{k[0]}|{k[1]}|{k[2]}|{k[3]}": v for k, v in stats.items()},
        "specific_lifts": {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in lifts.items()},
    }


# ---- main ----

def main():
    ap = argparse.ArgumentParser(
        description="Reproduce §6.4 numerical claims from the v0.4.1 corpus")
    ap.add_argument("manifests", nargs="+", help="manifest JSON paths")
    ap.add_argument("--store", default="./routing_store",
                    help="path to the routing_store directory")
    ap.add_argument("--out-json", default="results/section_6_4_analysis.json")
    ap.add_argument("--out-md", default="results/section_6_4_analysis.md")
    args = ap.parse_args()

    print(f"Loading {len(args.manifests)} manifests...")
    results, manifest_meta = load_manifests(args.manifests)
    print(f"  {len(results)} results loaded")

    # Deduplicate by trace_kappa
    seen = set()
    deduped = []
    for r in results:
        tk = r["trace_kappa"]
        if tk not in seen:
            seen.add(tk)
            deduped.append(r)
    print(f"  {len(deduped)} unique traces after dedup")

    store = JsonStore(args.store)

    print("Loading and verifying traces...")
    rows, n_verified, n_failed = load_and_verify_traces(deduped, store)
    print(f"  {n_verified} verified, {n_failed} failed")

    print("Computing statistics...")
    stats = build_cell_stats(rows)
    lifts = compute_specific_lifts(stats)

    # Write outputs
    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)

    json_out = generate_json(stats, lifts, manifest_meta, n_verified, n_failed)
    with open(args.out_json, "w") as f:
        json.dump(json_out, f, indent=2, default=str)
    print(f"  JSON: {args.out_json}")

    md_out = generate_markdown(stats, lifts, manifest_meta, n_verified, n_failed)
    with open(args.out_md, "w") as f:
        f.write(md_out)
    print(f"  Markdown: {args.out_md}")

    # Print markdown to stdout
    print()
    print(md_out)


if __name__ == "__main__":
    main()
