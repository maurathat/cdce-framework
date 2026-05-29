#!/usr/bin/env python3
"""
run_section_6_4_v04.py — §6.4 Cross-Model Strategy Transfer under v0.4
content-addressed traces.

Replicates §6.4 as written in Section_6_4_CrossModel_Transfer_Draft.md:
  - 2 tasks: trans_nl_code (cumavg), opt_routing (TSP)
  - 3 models: claude, gpt4o, gemini_flash
  - 3 conditions: no-strategy, shuffled, correct
  - 5 budgets: 2000, 1000, 500, 250, 125
  - Replications: 1x baseline; 2x on the 125-token cells (most compressed
    slice) so replay-agreement is measurable there.

§6.4 is a CROSS-MODEL transfer study: the "correct" strategy may have been
compressed by a different model than the one being tested. This is the point
of the section — it asks whether compressed strategies carry method across
model boundaries. Same-model transfer is §6.3.

Every cell produces a content-addressed Trace (trace_κ) via the v0.4
routing objects. The manifest records the full grid plan, chosen strategies,
and per-cell results.

Usage:
    python3 run_section_6_4_v04.py --dry-run      # inspect plan, no API calls
    python3 run_section_6_4_v04.py                 # execute (requires API keys)
"""

import argparse, json, math, os, random, sys, time

# ---- §6.4 grid (matches the draft exactly) ----

TASKS_64 = ["trans_nl_code", "opt_routing"]

MODELS_64 = ["claude", "gpt4o", "gemini_flash"]

CONDITIONS = ["no_strategy", "shuffled", "correct"]

BUDGETS = [2000, 1000, 500, 250, 125]

# Budget at which we double replications for replay-agreement measurement.
REPLAY_BUDGET = 125

# ---- cost model (conservative estimates per call) ----
# Prices per 1K tokens (approx).
BASE_INPUT_TOKENS = 500  # prompt overhead without strategy text
COST_PER_1K_INPUT = {
    "claude": 0.003,        # Sonnet 4
    "gpt4o": 0.0025,        # GPT-4o
    "gemini_flash": 0.0001, # Gemini 2.5 Flash
}
COST_PER_1K_OUTPUT = {
    "claude": 0.015,
    "gpt4o": 0.01,
    "gemini_flash": 0.0004,
}

COST_CEILING = 15.00    # hard stop
COST_RESERVE = 5.00     # effective ceiling = $10, $5 reserve for retries

RETRY_BACKOFF_S = 30
MAX_RETRIES = 1


# ---- κ-label computation for strategy entries ----

def compute_strategy_kappa(entry):
    """Compute the κ-label for a memory entry's strategy_text.

    The memory store doesn't carry uor_addr / κ-labels — entries have only
    a 20-char content_hash. We compute the real κ-label from a canonical
    JSON form so the v0.4 store keys on κ, not on the legacy hash.

    Canonical form: {"strategy_text": <text>} — sorted-key JSON, compact
    separators. This matches the cdce_routing_objects.canonical_bytes style.
    """
    from cdce_routing_objects import canonical_bytes, kappa_addr
    canon = canonical_bytes(entry, ["strategy_text"])
    return kappa_addr(canon)


# ---- strategy selection from memory ----

def load_memory(memory_dir):
    """Load all strategy entries from the memory store."""
    entries = []
    for fn in sorted(os.listdir(memory_dir)):
        if not fn.endswith(".json") or fn == "memory_index.json":
            continue
        with open(os.path.join(memory_dir, fn)) as f:
            e = json.load(f)
        if isinstance(e, dict) and e.get("strategy_text"):
            entries.append(e)
    return entries


def select_strategies(entries):
    """Select one canonical strategy per (task, budget) for the correct
    condition, and one foreign strategy per budget for the shuffled condition.

    For correct: pick the highest-generation strategy at each (task, budget).
    §6.4 is cross-model, so the source model is intentionally not constrained
    to match the receiving model — the strategy may have been compressed by
    any model in the corpus.

    For shuffled: pick one random strategy from a foreign task at each budget.

    Returns:
        correct: dict[(task_id, budget)] -> entry
        shuffled: dict[budget] -> entry (foreign to both §6.4 tasks)
    """
    random.seed(2026)  # reproducible selection

    correct = {}
    by_task_budget = {}
    for e in entries:
        tid = e.get("task_id")
        b = int(e.get("budget_at_creation", 0))
        if tid in TASKS_64 and b in BUDGETS:
            key = (tid, b)
            by_task_budget.setdefault(key, []).append(e)

    for key, candidates in by_task_budget.items():
        best = max(candidates, key=lambda e: e.get("generation", 0))
        correct[key] = best

    # Shuffled: foreign strategies (not from either §6.4 task)
    shuffled = {}
    foreign_by_budget = {}
    for e in entries:
        tid = e.get("task_id")
        b = int(e.get("budget_at_creation", 0))
        if tid not in TASKS_64 and b in BUDGETS:
            foreign_by_budget.setdefault(b, []).append(e)

    for b in BUDGETS:
        candidates = foreign_by_budget.get(b, [])
        if candidates:
            shuffled[b] = random.choice(candidates)

    return correct, shuffled


# ---- cell generation ----

def build_cells(correct, shuffled):
    """Build the full list of cells to execute.

    Each cell is a dict with: task_id, model, condition, budget,
    strategy_entry (or None), replications.
    """
    cells = []
    for task_id in TASKS_64:
        for model in MODELS_64:
            for budget in BUDGETS:
                reps = 2 if budget == REPLAY_BUDGET else 1

                # no_strategy
                cells.append({
                    "task_id": task_id,
                    "model": model,
                    "condition": "no_strategy",
                    "budget": budget,
                    "strategy_entry": None,
                    "replications": reps,
                })

                # shuffled
                sh = shuffled.get(budget)
                cells.append({
                    "task_id": task_id,
                    "model": model,
                    "condition": "shuffled",
                    "budget": budget,
                    "strategy_entry": sh,
                    "replications": reps,
                })

                # correct
                cr = correct.get((task_id, budget))
                cells.append({
                    "task_id": task_id,
                    "model": model,
                    "condition": "correct",
                    "budget": budget,
                    "strategy_entry": cr,
                    "replications": reps,
                })

    return cells


# Phase 2: targeted replication on cells where signal can appear.
# Reuses the same canonical strategies from Phase 1 (no fresh selection).
PHASE2_CELLS = [
    # Gemini Flash × opt_routing × {2000, 1000} — only budgets where output fits
    {"task_id": "opt_routing", "model": "gemini_flash", "budgets": [2000, 1000]},
    # GPT-4o × opt_routing × {500, 250} — transition zone
    {"task_id": "opt_routing", "model": "gpt4o", "budgets": [500, 250]},
    # Gemini Flash × trans_nl_code × {2000, 1000} — §6.4 reports +34% specific lift
    {"task_id": "trans_nl_code", "model": "gemini_flash", "budgets": [2000, 1000]},
]
PHASE2_REPS = 10


def build_phase2_cells(correct, shuffled):
    """Build Phase 2 cells: targeted replication on gradient-visible cells.

    Skip Claude routing (no headroom), Gemini ≤500 (truncates), budget 125
    (cliff). Uses the same strategies as Phase 1.
    """
    cells = []
    for spec in PHASE2_CELLS:
        task_id = spec["task_id"]
        model = spec["model"]
        for budget in spec["budgets"]:
            for cond in CONDITIONS:
                if cond == "no_strategy":
                    strat = None
                elif cond == "shuffled":
                    strat = shuffled.get(budget)
                else:
                    strat = correct.get((task_id, budget))
                cells.append({
                    "task_id": task_id,
                    "model": model,
                    "condition": cond,
                    "budget": budget,
                    "strategy_entry": strat,
                    "replications": PHASE2_REPS,
                })
    return cells


def estimate_cell_cost(model, budget, condition, strategy_entry):
    """Estimate cost for one API call to this model at this budget.

    When condition != no_strategy, the strategy text is included in the
    input prompt, so input tokens = base overhead + strategy text length
    (estimated as budget tokens, since strategies were compressed to that
    budget). This makes the estimate ~15-30% higher than base-only.
    """
    input_tokens = BASE_INPUT_TOKENS
    if condition != "no_strategy" and strategy_entry is not None:
        # Strategy text was compressed at this budget, so its token count
        # is roughly proportional to the budget it was created at.
        input_tokens += int(strategy_entry.get("budget_at_creation", budget))
    input_cost = (input_tokens / 1000) * COST_PER_1K_INPUT.get(model, 0.01)
    output_cost = (budget / 1000) * COST_PER_1K_OUTPUT.get(model, 0.01)
    return input_cost + output_cost


# ---- task prompt lookup ----

def get_task_prompt(task_id):
    """Get the canonical prompt for a §6.4 task."""
    from src.tasks import get_tasks
    for t in get_tasks():
        if t["id"] == task_id:
            return t["prompt"]
    raise ValueError(f"Task {task_id} not found in src.tasks")


# ---- execution ----

def execute_cell(cell, rep_index, store, cumulative_cost, dry_run):
    """Execute one cell replication. Returns (trace_dict, cost, status)."""
    from cdce_routing_objects import (
        make_taskspec, make_input_kappa, run_trace, verify,
        TRACE_FIELDS,
    )
    from src.tasks import get_tasks

    task_id = cell["task_id"]
    model = cell["model"]
    budget = cell["budget"]
    condition = cell["condition"]
    strat_entry = cell["strategy_entry"]

    est_cost = estimate_cell_cost(model, budget, condition, strat_entry)

    if dry_run:
        return None, est_cost, "dry_run"

    # Check cost ceiling
    if cumulative_cost + est_cost > COST_CEILING:
        return None, 0.0, "cost_ceiling"

    # Find the task definition
    task_def = None
    for t in get_tasks():
        if t["id"] == task_id:
            task_def = t
            break
    if task_def is None:
        return None, 0.0, f"task_not_found:{task_id}"

    # Build TaskSpec
    task_spec = make_taskspec(
        family=task_def.get("family", task_id.split("_")[0]),
        input_schema=f"{task_id}:canonical",
        success_predicate="scorer_registry",
        scorer="src.scorers",
        params={},
        task_id=task_id,
        prompt=task_def["prompt"],
    )
    store.put(task_spec["task_kappa"], task_spec)

    # §6.4 uses one canonical instance per task (the prompt IS the instance).
    # input_κ is therefore task-invariant: every cell for the same task_id
    # shares the same input_κ. If the grid is ever extended to multiple
    # instances per task, make_input_kappa must be called per-instance instead.
    input_obj, input_kappa = make_input_kappa(
        task_spec["task_kappa"], task_def["prompt"],
    )
    store.put(input_kappa, input_obj)

    # Store strategy if applicable, keyed by κ-label
    strategy_kappa = "none"
    if strat_entry is not None:
        sk = compute_strategy_kappa(strat_entry)
        store.put(sk, strat_entry)
        strategy_kappa = sk

    # Execute with retry
    for attempt in range(1 + MAX_RETRIES):
        try:
            trace = run_trace(
                task_spec, model, strategy_kappa,
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

def build_manifest(cells, correct, shuffled, results, run_id, ts, dry_run):
    """Build the manifest dict."""
    strategy_selections = {
        "correct": {},
        "shuffled": {},
    }
    for (tid, b), entry in correct.items():
        strategy_selections["correct"][f"{tid}@{b}"] = {
            "strategy_kappa": compute_strategy_kappa(entry),
            "content_hash": entry.get("content_hash"),
            "source_model": entry.get("model"),
            "generation": entry.get("generation"),
            "text_len": len(entry.get("strategy_text", "")),
        }
    for b, entry in shuffled.items():
        strategy_selections["shuffled"][str(b)] = {
            "strategy_kappa": compute_strategy_kappa(entry),
            "content_hash": entry.get("content_hash"),
            "task_id": entry.get("task_id"),
            "source_model": entry.get("model"),
            "generation": entry.get("generation"),
            "text_len": len(entry.get("strategy_text", "")),
        }

    return {
        "schema_version": "v0.4.1",
        "run_id": run_id,
        "timestamp": ts,
        "dry_run": dry_run,
        "grid": {
            "tasks": TASKS_64,
            "models": MODELS_64,
            "conditions": CONDITIONS,
            "budgets": BUDGETS,
            "replay_budget": REPLAY_BUDGET,
            "source": "Section_6_4_CrossModel_Transfer_Draft.md",
            "note_cross_model": "Cross-model transfer: correct strategies may "
                    "come from any source model, not necessarily the "
                    "receiving model.",
            "note_source_homogeneity": "All correct-strategy selections happen "
                    "to come from GPT-4o because that model reached the "
                    "deepest generation in the v0.3 corpus. This is an "
                    "artifact of which models were run longest, not a design "
                    "choice. The cross-model transfer claim holds (strategies "
                    "compressed by GPT-4o are tested on Claude and Gemini), "
                    "but the source-model diversity is zero. A future run "
                    "should select per-(source_model, task, budget) to test "
                    "whether source model identity affects transfer.",
        },
        "strategy_selections": strategy_selections,
        "cost_ceiling": COST_CEILING,
        "cost_reserve": COST_RESERVE,
        "results": results,
    }


# ---- main ----

def main():
    ap = argparse.ArgumentParser(
        description="§6.4 Cross-Model Transfer under v0.4 content-addressed traces")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the full plan without making API calls")
    ap.add_argument("--memory", default="./memory",
                    help="path to the memory store")
    ap.add_argument("--limit", type=int, default=0,
                    help="max number of cells to execute (0 = all)")
    ap.add_argument("--tasks", nargs="+", default=None,
                    help="filter to specific task_ids (e.g. --tasks opt_routing)")
    ap.add_argument("--phase2", action="store_true",
                    help="run Phase 2 targeted replication (Gemini routing/trans "
                         "+ GPT-4o routing transition zone, 10 reps each)")
    args = ap.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    run_id = f"section_6_4_v04_{ts}"

    print(f"{'DRY RUN — ' if args.dry_run else ''}§6.4 v0.4 corpus run")
    print(f"run_id: {run_id}")
    print()

    # Load and select strategies
    entries = load_memory(args.memory)
    correct, shuffled = select_strategies(entries)

    # Check strategy coverage
    missing = []
    for tid in TASKS_64:
        for b in BUDGETS:
            if (tid, b) not in correct:
                missing.append(f"correct:{tid}@{b}")
    for b in BUDGETS:
        if b not in shuffled:
            missing.append(f"shuffled@{b}")
    if missing:
        print(f"WARNING: missing strategies: {missing}")
        print("Cells with missing strategies will run as no_strategy.")
        print()

    # Build cells
    if args.phase2:
        all_cells = build_phase2_cells(correct, shuffled)
        print(f"  PHASE 2: {len(all_cells)} targeted cells, {PHASE2_REPS} reps each")
        print(f"  Cells: {[(s['task_id'], s['model'], s['budgets']) for s in PHASE2_CELLS]}")
    else:
        all_cells = build_cells(correct, shuffled)
    if args.tasks:
        all_cells = [c for c in all_cells if c["task_id"] in args.tasks]
        print(f"  --tasks {args.tasks}: {len(all_cells)} cells after filter")
    if args.limit > 0:
        cells = all_cells[:args.limit]
        print(f"  --limit {args.limit}: running {len(cells)} of {len(all_cells)} cells")
    else:
        cells = all_cells
    print()

    # Count and cost summary
    total_calls = sum(c["replications"] for c in cells)
    cost_by_model = {}
    for c in cells:
        m = c["model"]
        n = c["replications"]
        est = estimate_cell_cost(m, c["budget"], c["condition"],
                                 c["strategy_entry"]) * n
        cost_by_model[m] = cost_by_model.get(m, 0.0) + est
    total_est = sum(cost_by_model.values())

    print("=== GRID SUMMARY ===")
    print(f"  Tasks:       {TASKS_64}")
    print(f"  Models:      {MODELS_64}")
    print(f"  Conditions:  {CONDITIONS}")
    print(f"  Budgets:     {BUDGETS}")
    print(f"  Replay 2x at budget: {REPLAY_BUDGET}")
    print()
    print(f"  Total cells: {len(cells)}")
    print(f"  Total API calls (incl. replications): {total_calls}")
    print()
    print("  Estimated cost by model:")
    for m in MODELS_64:
        print(f"    {m:15s}: ${cost_by_model.get(m, 0):.2f}")
    print(f"    {'TOTAL':15s}: ${total_est:.2f}")
    print(f"    Ceiling:         ${COST_CEILING:.2f} (reserve ${COST_RESERVE:.2f})")
    print()

    # Strategy selections
    print("=== STRATEGY SELECTIONS ===")
    print("  CORRECT (one per task × budget, highest generation, any source model):")
    for tid in TASKS_64:
        for b in BUDGETS:
            entry = correct.get((tid, b))
            if entry:
                sk = compute_strategy_kappa(entry)
                print(f"    {tid} @ {b:>5}: {sk}")
                print(f"      content_hash={entry.get('content_hash')}, "
                      f"source_model={entry.get('model')}, "
                      f"gen={entry.get('generation')}, "
                      f"len={len(entry.get('strategy_text', ''))}")
            else:
                print(f"    {tid} @ {b:>5}: MISSING")
    print()
    print("  SHUFFLED (one foreign strategy per budget):")
    for b in BUDGETS:
        entry = shuffled.get(b)
        if entry:
            sk = compute_strategy_kappa(entry)
            print(f"    budget {b:>5}: {sk}")
            print(f"      content_hash={entry.get('content_hash')}, "
                  f"task={entry.get('task_id')}, "
                  f"source_model={entry.get('model')}, "
                  f"len={len(entry.get('strategy_text', ''))}")
        else:
            print(f"    budget {b:>5}: MISSING")
    print()

    # Cell plan
    print("=== CELL PLAN ===")
    print(f"  {'#':>3} {'task':16s} {'model':15s} {'cond':12s} "
          f"{'budget':>6} {'reps':>4} {'est$':>6}")
    print("  " + "-" * 70)
    for i, c in enumerate(cells):
        est = estimate_cell_cost(c["model"], c["budget"], c["condition"],
                                 c["strategy_entry"]) * c["replications"]
        print(f"  {i+1:3d} {c['task_id']:16s} {c['model']:15s} "
              f"{c['condition']:12s} {c['budget']:>6} {c['replications']:>4} "
              f"${est:>5.3f}")
    print()

    if args.dry_run:
        # Write manifest with dry_run=True
        results = []
        for i, c in enumerate(cells):
            sk = (compute_strategy_kappa(c["strategy_entry"])
                  if c["strategy_entry"] else None)
            for r in range(c["replications"]):
                results.append({
                    "cell_index": i,
                    "replication": r,
                    "task_id": c["task_id"],
                    "model": c["model"],
                    "condition": c["condition"],
                    "budget": c["budget"],
                    "strategy_kappa": sk,
                    "trace_kappa": None,
                    "status": "dry_run",
                    "est_cost": estimate_cell_cost(
                        c["model"], c["budget"], c["condition"],
                        c["strategy_entry"]),
                })

        manifest = build_manifest(
            cells, correct, shuffled, results, run_id, ts, dry_run=True)
        os.makedirs("manifests", exist_ok=True)
        manifest_path = f"manifests/{run_id}.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"Dry-run manifest written to {manifest_path}")
        print(f"Total calls that would be made: {total_calls}")
        print(f"Estimated total cost: ${total_est:.2f}")
        print()
        print("Inspect the manifest, then re-run without --dry-run to execute.")
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
                     f"{c['task_id']} × {c['model']} × {c['condition']} "
                     f"@ {c['budget']}")

            if cumulative_cost >= COST_CEILING:
                print(f"  {label}: COST CEILING (${cumulative_cost:.2f})")
                results.append({
                    "cell_index": i, "replication": r,
                    "task_id": c["task_id"], "model": c["model"],
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
                "task_id": c["task_id"], "model": c["model"],
                "condition": c["condition"], "budget": c["budget"],
                "strategy_kappa": (compute_strategy_kappa(c["strategy_entry"])
                                   if c["strategy_entry"] else None),
                "trace_kappa": trace_k,
                "score": score,
                "success": success,
                "status": status,
                "est_cost": cost,
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

            time.sleep(0.3)  # rate-limit courtesy

    # Write manifest
    manifest = build_manifest(
        cells, correct, shuffled, results, run_id, ts, dry_run=False)
    os.makedirs("manifests", exist_ok=True)
    manifest_path = f"manifests/{run_id}.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    print()
    print(f"=== COMPLETE ===")
    print(f"  OK: {ok_count}, Failed: {fail_count}")
    print(f"  Cumulative cost: ${cumulative_cost:.2f}")
    print(f"  Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
