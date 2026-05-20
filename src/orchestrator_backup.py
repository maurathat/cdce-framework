"""
CDCE Compression Harness — Orchestrator

The main experiment loop:
1. For each budget level (high → low):
2.   For each task family:
3.     For each task:
4.       Call all active models in parallel
5.       Extract compression metrics
6.       Feed prior strategy into next compression level (recursive)
7. Compute cross-family convergence at each level
8. Store results and generate plots
"""
import os
import json
import time
from datetime import datetime

from src.config import (
    BUDGET_LEVELS, TASK_FAMILIES, RESULTS_DIR, get_active_models,
)
from src.tasks import get_tasks, build_compression_prompt, content_hash, ALL_TASKS
from src.llm_clients import call_models_parallel, init_keys
from src.metrics import extract_metrics, compute_convergence, metrics_to_dict
from src.visualize import generate_all_plots
from src.memory import MemoryStore, MemoryEntry, compute_hash, UORBridge
from src.energy import compute_energy, energy_to_dict, compute_phase_transition

def run_experiment():
    """
    Execute the full compression experiment with persistent memory.
    
    Two layers of recursion:
    1. WITHIN a run: each budget level feeds prior strategy to the next
    2. ACROSS runs: memory store persists compressed geometry between sessions
    
    The memory store IS the geometric structure accumulating over time.
    Each run compresses further. Over many runs, we track whether the
    stored structures converge toward exceptional algebraic signatures.
    """
    print("=" * 60)
    print("CDCE COMPRESSION HARNESS")
    print("Recursive compression experiment for AGI architecture")
    print("=" * 60)

    # Initialize
    init_keys()
    active_models = get_active_models()

    # Initialize persistent memory
    memory = MemoryStore()
    uor = UORBridge()
    print(f"\nMemory store: {memory.index['total_entries']} prior geometries loaded")

    if not active_models:
        print("\n[ERROR] No API keys configured!")
        print("Copy .env.example to .env and add at least one key.")
        return

    print(f"\nActive models: {', '.join(active_models.keys())}")
    print(f"Budget levels: {BUDGET_LEVELS}")
    print(f"Task families: {TASK_FAMILIES}")

    total_calls = (
        len(BUDGET_LEVELS)
        * sum(len(get_tasks(f)) for f in TASK_FAMILIES)
        * len(active_models)
    )
    print(f"Total API calls: {total_calls}")
    print()

    # Results storage
    all_runs = []
    convergence_by_budget = {}

    # Prior strategies for within-run recursive compression
    # Key: (model, task_id) -> strategy text from previous budget level
    prior_strategies = {}

    # Load cross-session priors from memory store
    # This is the ACROSS-RUN recursion
    for model_name in active_models:
        for family in TASK_FAMILIES:
            for task in get_tasks(family):
                prior_memory = memory.retrieve(model_name, task["id"])
                if prior_memory:
                    prior_strategies[(model_name, task["id"])] = prior_memory.strategy_text
                    print(f"  Loaded prior geometry: {model_name}/{task['id']} "
                          f"(gen {prior_memory.generation}, "
                          f"{prior_memory.operator_count} ops)")

    # ── MAIN LOOP ──
    for budget_idx, budget in enumerate(BUDGET_LEVELS):
        print(f"\n{'─' * 50}")
        print(f"COMPRESSION LEVEL {budget_idx + 1}/{len(BUDGET_LEVELS)}: "
              f"Budget = {budget} tokens")
        print(f"{'─' * 50}")

        level_metrics = []

        for family in TASK_FAMILIES:
            tasks = get_tasks(family)
            print(f"\n  Family: {family} ({len(tasks)} tasks)")

            for task in tasks:
                task_id = task["id"]

                # For models that have prior strategies, build custom prompts
                # Prior comes from EITHER previous budget level OR memory store
                model_prompts = {}
                for model_name in active_models:
                    prior_key = (model_name, task_id)
                    prior = prior_strategies.get(prior_key)
                    model_prompts[model_name] = build_compression_prompt(
                        task, budget, prior_strategy=prior
                    )

                # Call all models
                print(f"    → {task['name']} ({task_id})...", end=" ", flush=True)

                responses = []
                for model_name, cfg in active_models.items():
                    from src.llm_clients import call_model
                    resp = call_model(
                        provider=cfg["provider"],
                        model=cfg["model"],
                        prompt=model_prompts[model_name],
                        max_tokens=budget,
                        task_id=task_id,
                    )
                    responses.append((model_name, resp))

                    # Extract metrics, store prior strategies, persist to memory
                     for model_name, resp in responses:
                        content=resp.content,
                        task_id=task_id,
                        task_family=family,
                        model=model_name,
                        budget=budget,
                        output_tokens=resp.output_tokens,
                    )
                    level_metrics.append(m)
                    all_runs.append(metrics_to_dict(m))
                # Compute energy metrics
                    em = compute_energy(
                        model=model_name,
                        task_id=task_id,
                        budget=budget,
                        input_tokens=resp.input_tokens,
                        output_tokens=resp.output_tokens,
                        latency_ms=resp.latency_ms,
                        operator_count=m.operator_count,
                        reuse_ratio=m.reuse_ratio,
                        word_count=m.word_count,
                        step_count=m.step_count,
                    )
                    all_runs[-1].update(energy_to_dict(em))    
                    # Store this strategy as prior for next compression level
                    prior_strategies[(model_name, task_id)] = resp.content

                    # ── PERSIST TO MEMORY STORE ──
                    if not resp.content.startswith("[ERROR"):
                        prior_mem = memory.retrieve(model_name, task_id)
                        prior_hash = prior_mem.content_hash if prior_mem else None
                        lineage = (prior_mem.lineage if prior_mem else [])
                        if prior_hash:
                            lineage = lineage + [prior_hash]

                        entry = MemoryEntry(
                            content_hash=compute_hash(resp.content),
                            model=model_name,
                            task_id=task_id,
                            task_family=family,
                            generation=memory.get_generation(model_name, task_id),
                            budget_at_creation=budget,
                            strategy_text=resp.content,
                            operator_count=m.operator_count,
                            reuse_ratio=m.reuse_ratio,
                            unique_verbs=m.unique_verbs,
                            lineage=lineage,
                        )
                        memory.store(entry)

                        # Optional: also store to UOR
                        if uor.available:
                            uor.store_to_uor(resp.content, {
                                "model": model_name,
                                "task": task_id,
                                "generation": entry.generation,
                                "operator_count": m.operator_count,
                            })

                print(f"done ({len(responses)} models)")

        # Compute cross-family convergence at this budget level
        for model_name in active_models:
            model_metrics = [m for m in level_metrics if m.model == model_name]
            if model_metrics:
                conv = compute_convergence(model_metrics)
                key = f"{model_name}@{budget}"
                convergence_by_budget[key] = conv

                avg_ops = sum(m.operator_count for m in model_metrics) / len(model_metrics)
                print(f"\n  {model_name}: avg operators={avg_ops:.1f}, "
                      f"convergence={conv['jaccard_mean']:.3f}")

    # ── SAVE RESULTS ──
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    memory_report = memory.convergence_report()

    results = {
        "experiment": "cdce_compression_harness",
        "timestamp": timestamp,
        "models": list(active_models.keys()),
        "budget_levels": BUDGET_LEVELS,
        "task_families": TASK_FAMILIES,
        "total_runs": len(all_runs),
        "runs": all_runs,
        "convergence_by_budget": convergence_by_budget,
        "memory": {
            "total_stored": memory_report["total_stored"],
            "convergence_events": memory_report["convergence_events"],
            "cross_model_convergence": memory_report["cross_model_convergence"],
            "operator_by_generation": memory_report["operator_by_generation"],
        },
    }

    results_path = os.path.join(RESULTS_DIR, f"experiment_{timestamp}.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {results_path}")

    # ── GENERATE PLOTS ──
    generate_all_plots(results, RESULTS_DIR)

    # ── MEMORY CONVERGENCE REPORT ──
    memory.print_report()

    # ── PRINT SUMMARY ──
    print_summary(results)

    return results


def print_summary(results: dict):
    """Print a human-readable summary of key findings."""
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

    # Group by model and budget
    model_budget_ops = {}
    for run in results["runs"]:
        m, b = run["model"], run["budget"]
        model_budget_ops.setdefault(m, {}).setdefault(b, []).append(run["operator_count"])

    for model in sorted(model_budget_ops.keys()):
        print(f"\n  Model: {model}")
        print(f"  {'Budget':>8} | {'Avg Operators':>14} | {'Δ':>6} | {'Signal':>20}")
        print(f"  {'─' * 55}")

        prev_ops = None
        for budget in sorted(model_budget_ops[model].keys(), reverse=True):
            ops_list = model_budget_ops[model][budget]
            avg = sum(ops_list) / len(ops_list)
            delta = ""
            signal = ""
            if prev_ops is not None:
                diff = avg - prev_ops
                delta = f"{diff:+.1f}"
                if diff < -2:
                    signal = "← compression"
                elif abs(diff) < 1:
                    signal = "← PLATEAU?"
            prev_ops = avg
            print(f"  {budget:>8} | {avg:>14.1f} | {delta:>6} | {signal:>20}")

    # Convergence trend
    conv = results.get("convergence_by_budget", {})
    if conv:
        print(f"\n  Cross-Family Convergence Trend:")
        for key in sorted(conv.keys(), key=lambda k: (k.split("@")[0], -int(k.split("@")[1]))):
            model, budget = key.rsplit("@", 1)
            j = conv[key]["jaccard_mean"]
            bar = "█" * int(j * 30)
            print(f"    {model:>10} @ {budget:>5}: {j:.3f} {bar}")

    print("\n" + "=" * 60)
    print("CDCE PREDICTIONS TO CHECK:")
    print("  P1: Does operator count decrease monotonically?")
    print("  P2: Are there plateaus (stage transitions)?")
    print("  P3: Does cross-family convergence increase under pressure?")
    print("  P4: Is the pattern consistent across models (substrate-independent)?")
    print("=" * 60)
