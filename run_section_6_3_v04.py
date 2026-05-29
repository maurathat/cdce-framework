#!/usr/bin/env python3
"""
run_section_6_3_v04.py — §6.3 Outcome-Lossless (single-model) under v0.4.1
content-addressed traces.

§6.3 tests whether a single model (Claude) can reuse its own compressed
strategy to solve a task (trans_nl_code / cumavg). This is the single-model
companion to §6.4's cross-model transfer.

Grid:
  - 1 task: trans_nl_code (cumavg)
  - 1 model: claude
  - 3 conditions: no_strategy, shuffled, correct
  - 5 budgets: 2000, 1000, 500, 250, 125
  - 10 reps per cell
  - Total: 150 calls

Strategy reuse: this script loads the SAME strategy_κ values that §6.4 used
for trans_nl_code correct and shuffled conditions. A precondition check at
startup verifies the match against the §6.4 Phase 2 manifest. Both papers
reference the same canonical strategies.

Usage:
    python3 run_section_6_3_v04.py --dry-run
    python3 run_section_6_3_v04.py --limit 6
    python3 run_section_6_3_v04.py
"""

import argparse, json, math, os, sys, time

# ---- §6.3 grid ----

TASK_63 = "trans_nl_code"
MODEL_63 = "claude"
CONDITIONS = ["no_strategy", "shuffled", "correct"]
BUDGETS = [2000, 1000, 500, 250, 125]
REPS = 10

# §6.4 Phase 2 manifest — source of truth for strategy_κ values
SECTION_64_MANIFEST = "manifests/section_6_4_v04_20260528_155940.json"

# ---- cost model ----
BASE_INPUT_TOKENS = 500
COST_PER_1K_INPUT = 0.003   # Claude Sonnet 4
COST_PER_1K_OUTPUT = 0.015
COST_CEILING = 15.00
RETRY_BACKOFF_S = 30
MAX_RETRIES = 1


# ---- strategy loading from §6.4 manifest ----

def load_64_strategies():
    """Load strategy_κ values from the §6.4 manifest.
    Returns (correct, shuffled) dicts keyed by budget."""
    if not os.path.exists(SECTION_64_MANIFEST):
        print(f"FATAL: §6.4 manifest not found: {SECTION_64_MANIFEST}")
        print("Run §6.4 first, or fix the path.")
        sys.exit(1)

    with open(SECTION_64_MANIFEST) as f:
        m64 = json.load(f)

    ss = m64["strategy_selections"]

    correct = {}
    for key, val in ss["correct"].items():
        if key.startswith("trans_nl_code@"):
            budget = int(key.split("@")[1])
            correct[budget] = val["strategy_kappa"]

    shuffled = {}
    for key, val in ss["shuffled"].items():
        budget = int(key)
        shuffled[budget] = val["strategy_kappa"]

    return correct, shuffled


def load_strategy_entries(memory_dir, correct_kappas, shuffled_kappas):
    """Load strategy entries from memory, matched by κ-label.
    Returns dicts mapping budget -> entry."""
    from cdce_routing_objects import canonical_bytes, kappa_addr

    entries = []
    for fn in sorted(os.listdir(memory_dir)):
        if not fn.endswith(".json") or fn == "memory_index.json":
            continue
        with open(os.path.join(memory_dir, fn)) as f:
            e = json.load(f)
        if isinstance(e, dict) and e.get("strategy_text"):
            canon = canonical_bytes(e, ["strategy_text"])
            e["_kappa"] = kappa_addr(canon)
            entries.append(e)

    correct_entries = {}
    shuffled_entries = {}

    for e in entries:
        k = e["_kappa"]
        for b, ck in correct_kappas.items():
            if k == ck:
                correct_entries[b] = e
        for b, sk in shuffled_kappas.items():
            if k == sk:
                shuffled_entries[b] = e

    return correct_entries, shuffled_entries


def verify_strategy_match(correct_kappas, shuffled_kappas,
                          correct_entries, shuffled_entries):
    """Hard-fail if strategy_κ values don't match §6.4."""
    errors = []
    for b in BUDGETS:
        if b not in correct_kappas:
            errors.append(f"correct@{b}: missing from §6.4 manifest")
        elif b not in correct_entries:
            errors.append(f"correct@{b}: κ={correct_kappas[b][:30]}… not found in memory")
        if b not in shuffled_kappas:
            errors.append(f"shuffled@{b}: missing from §6.4 manifest")
        elif b not in shuffled_entries:
            errors.append(f"shuffled@{b}: κ={shuffled_kappas[b][:30]}… not found in memory")

    if errors:
        print("FATAL: §6.3 strategy_κ values do not match §6.4 manifest.")
        print("Strategies must be identical across both sections.")
        for e in errors:
            print(f"  {e}")
        sys.exit(1)

    # Verify κ-labels match
    for b in BUDGETS:
        ck = correct_entries[b]["_kappa"]
        expected = correct_kappas[b]
        if ck != expected:
            print(f"FATAL: correct@{b} κ mismatch:")
            print(f"  computed: {ck}")
            print(f"  §6.4:    {expected}")
            sys.exit(1)
        sk = shuffled_entries[b]["_kappa"]
        expected_s = shuffled_kappas[b]
        if sk != expected_s:
            print(f"FATAL: shuffled@{b} κ mismatch:")
            print(f"  computed: {sk}")
            print(f"  §6.4:    {expected_s}")
            sys.exit(1)


def compute_strategy_kappa(entry):
    """Compute the κ-label for a memory entry."""
    from cdce_routing_objects import canonical_bytes, kappa_addr
    return kappa_addr(canonical_bytes(entry, ["strategy_text"]))


# ---- cost estimation ----

def estimate_cell_cost(budget, condition, strategy_entry):
    input_tokens = BASE_INPUT_TOKENS
    if condition != "no_strategy" and strategy_entry is not None:
        input_tokens += int(strategy_entry.get("budget_at_creation", budget))
    input_cost = (input_tokens / 1000) * COST_PER_1K_INPUT
    output_cost = (budget / 1000) * COST_PER_1K_OUTPUT
    return input_cost + output_cost


# ---- cell building ----

def build_cells(correct_entries, shuffled_entries):
    cells = []
    for budget in BUDGETS:
        for cond in CONDITIONS:
            if cond == "no_strategy":
                strat = None
            elif cond == "shuffled":
                strat = shuffled_entries.get(budget)
            else:
                strat = correct_entries.get(budget)
            cells.append({
                "task_id": TASK_63,
                "model": MODEL_63,
                "condition": cond,
                "budget": budget,
                "strategy_entry": strat,
                "replications": REPS,
            })
    return cells


# ---- execution ----

def execute_cell(cell, rep_index, store, cumulative_cost, dry_run):
    from cdce_routing_objects import (
        make_taskspec, make_input_kappa, run_trace, verify, TRACE_FIELDS,
    )
    from src.tasks import get_tasks

    task_id = cell["task_id"]
    budget = cell["budget"]
    condition = cell["condition"]
    strat_entry = cell["strategy_entry"]

    est_cost = estimate_cell_cost(budget, condition, strat_entry)

    if dry_run:
        return None, est_cost, "dry_run"

    if cumulative_cost + est_cost > COST_CEILING:
        return None, 0.0, "cost_ceiling"

    task_def = None
    for t in get_tasks():
        if t["id"] == task_id:
            task_def = t
            break
    if task_def is None:
        return None, 0.0, f"task_not_found:{task_id}"

    task_spec = make_taskspec(
        family=task_def.get("family", "translation"),
        input_schema=f"{task_id}:canonical",
        success_predicate="scorer_registry",
        scorer="src.scorers",
        params={},
        task_id=task_id,
        prompt=task_def["prompt"],
    )
    store.put(task_spec["task_kappa"], task_spec)

    # §6.3 uses one canonical instance per task — input_κ is task-invariant.
    input_obj, input_kappa = make_input_kappa(
        task_spec["task_kappa"], task_def["prompt"],
    )
    store.put(input_kappa, input_obj)

    strategy_kappa = "none"
    if strat_entry is not None:
        sk = compute_strategy_kappa(strat_entry)
        store.put(sk, strat_entry)
        strategy_kappa = sk

    for attempt in range(1 + MAX_RETRIES):
        try:
            trace = run_trace(
                task_spec, MODEL_63, strategy_kappa,
                input_kappa, budget, store,
            )
            store.put(trace["trace_kappa"], trace)
            ok = verify(trace, TRACE_FIELDS, "trace_kappa")
            if not ok:
                return trace, est_cost, "verify_failed"
            return trace, est_cost, "ok"
        except Exception as e:
            if attempt < MAX_RETRIES:
                print(f"      retry in {RETRY_BACKOFF_S}s ({e})")
                time.sleep(RETRY_BACKOFF_S)
            else:
                return None, est_cost, f"error:{e}"

    return None, est_cost, "exhausted_retries"


# ---- manifest ----

def build_manifest(cells, correct_kappas, shuffled_kappas, results,
                   run_id, ts, dry_run):
    strategy_selections = {"correct": {}, "shuffled": {}}
    for b, k in correct_kappas.items():
        strategy_selections["correct"][f"trans_nl_code@{b}"] = {
            "strategy_kappa": k,
            "source": "reused from §6.4 manifest",
        }
    for b, k in shuffled_kappas.items():
        strategy_selections["shuffled"][str(b)] = {
            "strategy_kappa": k,
            "source": "reused from §6.4 manifest",
        }

    return {
        "schema_version": "v0.4.1",
        "run_id": run_id,
        "timestamp": ts,
        "dry_run": dry_run,
        "section": "6.3",
        "grid": {
            "task": TASK_63,
            "model": MODEL_63,
            "conditions": CONDITIONS,
            "budgets": BUDGETS,
            "reps_per_cell": REPS,
            "note": "Single-model outcome-lossless. Strategies reused from "
                    "§6.4 Phase 2 manifest to ensure cross-section consistency.",
            "section_64_manifest": SECTION_64_MANIFEST,
        },
        "strategy_selections": strategy_selections,
        "cost_ceiling": COST_CEILING,
        "results": results,
    }


# ---- main ----

def main():
    ap = argparse.ArgumentParser(
        description="§6.3 Outcome-Lossless under v0.4.1 content-addressed traces")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--memory", default="./memory")
    ap.add_argument("--limit", type=int, default=0,
                    help="max number of cells to execute (0 = all)")
    args = ap.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"section_6_3_v04_{ts}"

    print(f"{'DRY RUN — ' if args.dry_run else ''}§6.3 v0.4.1 corpus run")
    print(f"run_id: {run_id}")
    print(f"task: {TASK_63}, model: {MODEL_63}")
    print()

    # Load and verify strategies against §6.4
    print(f"Loading strategies from §6.4 manifest: {SECTION_64_MANIFEST}")
    correct_kappas, shuffled_kappas = load_64_strategies()
    correct_entries, shuffled_entries = load_strategy_entries(
        args.memory, correct_kappas, shuffled_kappas)
    verify_strategy_match(correct_kappas, shuffled_kappas,
                          correct_entries, shuffled_entries)
    print("  Strategy κ-labels match §6.4 manifest. Precondition passed.")
    print()

    # Build cells
    all_cells = build_cells(correct_entries, shuffled_entries)
    if args.limit > 0:
        cells = all_cells[:args.limit]
        print(f"  --limit {args.limit}: running {len(cells)} of {len(all_cells)} cells")
    else:
        cells = all_cells

    total_calls = sum(c["replications"] for c in cells)
    total_est = sum(estimate_cell_cost(c["budget"], c["condition"],
                    c["strategy_entry"]) * c["replications"] for c in cells)

    print(f"\n=== GRID SUMMARY ===")
    print(f"  Task:       {TASK_63}")
    print(f"  Model:      {MODEL_63}")
    print(f"  Conditions: {CONDITIONS}")
    print(f"  Budgets:    {BUDGETS}")
    print(f"  Reps/cell:  {REPS}")
    print(f"  Total cells: {len(cells)}")
    print(f"  Total API calls: {total_calls}")
    print(f"  Estimated cost: ${total_est:.2f}")
    print(f"  Ceiling: ${COST_CEILING:.2f}")
    print()

    # Strategy selections
    print("=== STRATEGY SELECTIONS (reused from §6.4) ===")
    print("  CORRECT:")
    for b in BUDGETS:
        print(f"    @{b:>5}: {correct_kappas[b]}")
    print("  SHUFFLED:")
    for b in BUDGETS:
        print(f"    @{b:>5}: {shuffled_kappas[b]}")
    print()

    # Cell plan
    print("=== CELL PLAN ===")
    print(f"  {'#':>3} {'cond':12s} {'budget':>6} {'reps':>4} {'est$':>6}")
    print("  " + "-" * 35)
    for i, c in enumerate(cells):
        est = estimate_cell_cost(c["budget"], c["condition"],
                                 c["strategy_entry"]) * c["replications"]
        print(f"  {i+1:3d} {c['condition']:12s} {c['budget']:>6} "
              f"{c['replications']:>4} ${est:>5.3f}")
    print()

    if args.dry_run:
        results = []
        for i, c in enumerate(cells):
            sk = compute_strategy_kappa(c["strategy_entry"]) if c["strategy_entry"] else None
            for r in range(c["replications"]):
                results.append({
                    "cell_index": i, "replication": r,
                    "task_id": TASK_63, "model": MODEL_63,
                    "condition": c["condition"], "budget": c["budget"],
                    "strategy_kappa": sk,
                    "trace_kappa": None, "status": "dry_run",
                    "est_cost": estimate_cell_cost(c["budget"], c["condition"],
                                                   c["strategy_entry"]),
                })

        manifest = build_manifest(cells, correct_kappas, shuffled_kappas,
                                  results, run_id, ts, dry_run=True)
        os.makedirs("manifests", exist_ok=True)
        path = f"manifests/{run_id}.json"
        with open(path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"Dry-run manifest: {path}")
        print(f"Total calls: {total_calls}, est cost: ${total_est:.2f}")
        return

    # ---- LIVE EXECUTION ----
    from cdce_routing_objects import JsonStore
    store = JsonStore("./routing_store")

    results = []
    cumulative_cost = 0.0
    ok_count = 0
    fail_count = 0

    for i, c in enumerate(cells):
        for r in range(c["replications"]):
            label = (f"[{i+1}/{len(cells)} rep {r+1}/{c['replications']}] "
                     f"{c['condition']} @ {c['budget']}")

            if cumulative_cost >= COST_CEILING:
                print(f"  {label}: COST CEILING (${cumulative_cost:.2f})")
                results.append({
                    "cell_index": i, "replication": r,
                    "task_id": TASK_63, "model": MODEL_63,
                    "condition": c["condition"], "budget": c["budget"],
                    "strategy_kappa": (compute_strategy_kappa(c["strategy_entry"])
                                      if c["strategy_entry"] else None),
                    "trace_kappa": None, "status": "cost_ceiling",
                    "est_cost": 0.0,
                })
                fail_count += 1
                continue

            trace, cost, status = execute_cell(
                c, r, store, cumulative_cost, dry_run=False)
            cumulative_cost += cost

            trace_k = trace["trace_kappa"] if trace else None
            score = trace["score"] if trace else None
            success = trace["success"] if trace else None

            results.append({
                "cell_index": i, "replication": r,
                "task_id": TASK_63, "model": MODEL_63,
                "condition": c["condition"], "budget": c["budget"],
                "strategy_kappa": (compute_strategy_kappa(c["strategy_entry"])
                                   if c["strategy_entry"] else None),
                "trace_kappa": trace_k,
                "score": score, "success": success,
                "status": status, "est_cost": cost,
            })

            if status == "ok":
                ok_count += 1
                score_str = (f"{score}" if score is not None
                             and not math.isnan(score) else "NaN")
                print(f"  {label}: {trace_k[:30]}… score={score_str} "
                      f"[${cumulative_cost:.2f}]")
            else:
                fail_count += 1
                print(f"  {label}: {status} [${cumulative_cost:.2f}]")

            time.sleep(0.3)

    manifest = build_manifest(cells, correct_kappas, shuffled_kappas,
                              results, run_id, ts, dry_run=False)
    os.makedirs("manifests", exist_ok=True)
    path = f"manifests/{run_id}.json"
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print()
    print(f"=== COMPLETE ===")
    print(f"  OK: {ok_count}, Failed: {fail_count}")
    print(f"  Cumulative cost: ${cumulative_cost:.2f}")
    print(f"  Manifest: {path}")


if __name__ == "__main__":
    main()
