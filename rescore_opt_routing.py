#!/usr/bin/env python3
"""
rescore_opt_routing.py — Re-score existing opt_routing traces with the
scorer, emit new trace_κs, update the manifest.

No API calls. Walks the routing_store for opt_routing traces, recomputes
score/success/graded from stored output, emits new Traces with new trace_κs.
Old traces stay on disk (not deleted).

Usage:
    python3 rescore_opt_routing.py --manifest manifests/section_6_4_v04_20260528_152116.json
"""

import argparse, json, os, sys
from collections import defaultdict

from cdce_routing_objects import (
    make_trace, verify, TRACE_FIELDS, JsonStore,
)
from src.scorers import score_opt_routing, _extract_tour_distance


def main():
    ap = argparse.ArgumentParser(description="Re-score opt_routing traces")
    ap.add_argument("--manifest", required=True, help="manifest JSON to rescore")
    args = ap.parse_args()

    with open(args.manifest) as f:
        manifest = json.load(f)

    store = JsonStore("./routing_store")

    # Capture old rescore results for diffing
    old_by_tk = {}
    for rr in manifest.get("rescore_runs", []):
        old_by_tk[rr["old_trace_kappa"]] = rr

    rescore_runs = []
    table_rows = []

    for result in manifest["results"]:
        if result.get("task_id") != "opt_routing":
            continue
        if result.get("status") != "ok":
            continue

        old_tk = result["trace_kappa"]
        trace = store.get(old_tk)
        if trace is None:
            print(f"  SKIP: trace not found in store: {old_tk[:40]}...")
            continue

        input_obj = store.get(trace["input_kappa"])
        if input_obj is None:
            print(f"  SKIP: input not found: {trace['input_kappa'][:40]}...")
            continue

        output = trace["output"]
        new_score, new_success, new_graded = score_opt_routing(input_obj, output)
        parsed_dist = _extract_tour_distance(output)

        new_trace = make_trace(
            task_kappa=trace["task_kappa"],
            model=trace["model"],
            strategy_kappa=trace["strategy_kappa"],
            input_kappa=trace["input_kappa"],
            output=output,
            score=new_score,
            success=new_success,
            budget=trace["budget"],
            graded=new_graded,
        )
        new_tk = new_trace["trace_kappa"]
        store.put(new_tk, new_trace)

        ok = verify(new_trace, TRACE_FIELDS, "trace_kappa")
        if not ok:
            print(f"  FAIL: new trace did not verify: {new_tk[:40]}...")
            continue

        old_parsed = old_by_tk.get(old_tk, {}).get("parsed_distance")

        rescore_runs.append({
            "old_trace_kappa": old_tk,
            "new_trace_kappa": new_tk,
            "old_graded": trace.get("graded", False),
            "new_graded": new_graded,
            "score": new_score,
            "success": new_success,
            "parsed_distance": parsed_dist,
            "old_parsed_distance": old_parsed,
            "scorer_version": "opt_routing_v2",
        })

        table_rows.append({
            "model": result["model"],
            "condition": result["condition"],
            "budget": result["budget"],
            "replication": result.get("replication", 0),
            "parsed_dist": parsed_dist,
            "old_parsed_dist": old_parsed,
            "success": new_success,
            "graded": new_graded,
        })

    # Replace rescore_runs in manifest
    manifest["rescore_runs"] = rescore_runs
    with open(args.manifest, "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    # ---- Print results ----
    models = ["claude", "gpt4o", "gemini_flash"]
    conditions = ["no_strategy", "shuffled", "correct"]
    budgets = [2000, 1000, 500, 250, 125]

    by_cell = defaultdict(list)
    for r in table_rows:
        by_cell[(r["model"], r["condition"], r["budget"])].append(r)

    # Success rates
    print(f"\n=== OPT_ROUTING SUCCESS RATES (optimal=70, rescored v2) ===")
    header = f"  {'model':15s} {'condition':12s}"
    for b in budgets:
        header += f" {b:>5}"
    print(header)
    print("  " + "-" * (30 + 6 * len(budgets)))

    for model in models:
        for cond in conditions:
            row = f"  {model:15s} {cond:12s}"
            for b in budgets:
                cells = by_cell.get((model, cond, b), [])
                if not cells:
                    row += "     —"
                else:
                    graded_cells = [c for c in cells if c["graded"]]
                    if not graded_cells:
                        row += "    ng"
                    else:
                        rate = sum(1 for c in graded_cells if c["success"]) / len(graded_cells)
                        row += f" {rate:>4.0%} "
            print(row)

    # Parsed distances
    print(f"\n=== PARSED DISTANCES (model × condition × budget) ===")
    header = f"  {'model':15s} {'condition':12s}"
    for b in budgets:
        header += f" {b:>5}"
    print(header)
    print("  " + "-" * (30 + 6 * len(budgets)))

    for model in models:
        for cond in conditions:
            row = f"  {model:15s} {cond:12s}"
            for b in budgets:
                cells = by_cell.get((model, cond, b), [])
                if not cells:
                    row += "     —"
                else:
                    dists = [c["parsed_dist"] for c in cells]
                    dist_strs = [f"{d:.0f}" if d is not None else "?" for d in dists]
                    row += f" {','.join(dist_strs):>5}"
            print(row)

    # Cells where parsing changed
    changed = [r for r in table_rows if r["parsed_dist"] != r["old_parsed_dist"]]
    print(f"\n=== PARSING CHANGES (v1 → v2): {len(changed)} cells changed ===")
    if changed:
        print(f"  {'model':15s} {'condition':12s} {'budget':>6} {'rep':>3} "
              f"{'old':>5} {'new':>5} {'old_ok':>6} {'new_ok':>6}")
        print("  " + "-" * 60)
        for r in sorted(changed, key=lambda r: (r["model"], r["condition"], -r["budget"])):
            old_d = f"{r['old_parsed_dist']:.0f}" if r["old_parsed_dist"] is not None else "?"
            new_d = f"{r['parsed_dist']:.0f}" if r["parsed_dist"] is not None else "?"
            old_ok = "—"
            if r["old_parsed_dist"] is not None:
                old_ok = "OK" if abs(r["old_parsed_dist"] - 70) < 0.5 else "FAIL"
            new_ok = "—"
            if r["parsed_dist"] is not None:
                new_ok = "OK" if abs(r["parsed_dist"] - 70) < 0.5 else "FAIL"
            print(f"  {r['model']:15s} {r['condition']:12s} {r['budget']:>6} "
                  f"{r['replication']:>3} {old_d:>5} {new_d:>5} {old_ok:>6} {new_ok:>6}")

    # Gemini Flash detail
    print(f"\n=== GEMINI FLASH DETAIL (per cell) ===")
    print(f"  {'condition':12s} {'budget':>6} {'rep':>3} {'dist':>5} "
          f"{'graded':>6} {'success':>7}")
    print("  " + "-" * 45)
    for r in sorted(table_rows, key=lambda r: (r["condition"], -r["budget"], r["replication"])):
        if r["model"] != "gemini_flash":
            continue
        dist_str = f"{r['parsed_dist']:.0f}" if r["parsed_dist"] is not None else "?"
        print(f"  {r['condition']:12s} {r['budget']:>6} {r['replication']:>3} "
              f"{dist_str:>5} {'yes' if r['graded'] else 'no':>6} "
              f"{'OK' if r['success'] else 'FAIL':>7}")

    print(f"\nManifest updated: {args.manifest}")
    print(f"  {len(rescore_runs)} traces rescored (v2), new trace_κs written.")


if __name__ == "__main__":
    main()
