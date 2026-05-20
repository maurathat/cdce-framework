import os
import json
import time
from datetime import datetime

from src.config import (
    BUDGET_LEVELS, TASK_FAMILIES, RESULTS_DIR, get_active_models,
)
from src.tasks import get_tasks, build_compression_prompt, content_hash, ALL_TASKS
from src.llm_clients import call_models_parallel, init_keys, call_model
from src.metrics import extract_metrics, compute_convergence, metrics_to_dict
from src.visualize import generate_all_plots
from src.memory import MemoryStore, MemoryEntry, compute_hash, UORBridge
from src.energy import compute_energy, energy_to_dict, compute_phase_transition


def run_experiment():
    print("=" * 60)
    print("CDCE COMPRESSION HARNESS")
    print("Recursive compression experiment for AGI architecture")
    print("=" * 60)

    init_keys()
    active_models = get_active_models()
    memory = MemoryStore()
    uor = UORBridge()
    print(f"\nMemory store: {memory.index['total_entries']} prior geometries loaded")

    if not active_models:
        print("\n[ERROR] No API keys configured!")
        return

    print(f"\nActive models: {', '.join(active_models.keys())}")
    print(f"Budget levels: {BUDGET_LEVELS}")
    print(f"Task families: {TASK_FAMILIES}")
    total_calls = len(BUDGET_LEVELS) * sum(len(get_tasks(f)) for f in TASK_FAMILIES) * len(active_models)
    print(f"Total API calls: {total_calls}\n")

    prior_strategies = {}
    for model_name in active_models:
        for family in TASK_FAMILIES:
            for task in get_tasks(family):
                prior_memory = memory.retrieve(model_name, task["id"])
                if prior_memory:
                    prior_strategies[(model_name, task["id"])] = prior_memory.strategy_text
                    print(f"  Loaded prior geometry: {model_name}/{task['id']} "
                          f"(gen {prior_memory.generation}, {prior_memory.operator_count} ops)")

    all_runs = []
    all_energy = []
    convergence_by_budget = {}

    for budget_idx, budget in enumerate(BUDGET_LEVELS):
        print(f"\n{'─' * 50}")
        print(f"COMPRESSION LEVEL {budget_idx + 1}/{len(BUDGET_LEVELS)}: Budget = {budget} tokens")
        print(f"{'─' * 50}")

        level_metrics = []

        for family in TASK_FAMILIES:
            tasks = get_tasks(family)
            print(f"\n  Family: {family} ({len(tasks)} tasks)")

            for task in tasks:
                task_id = task["id"]

                model_prompts = {}
                for model_name in active_models:
                    prior_key = (model_name, task_id)
                    prior = prior_strategies.get(prior_key)
                    model_prompts[model_name] = build_compression_prompt(task, budget, prior_strategy=prior)

                print(f"    → {task['name']} ({task_id})...", end=" ", flush=True)

                responses = []
                for model_name, cfg in active_models.items():
                    resp = call_model(
                        provider=cfg["provider"],
                        model=cfg["model"],
                        prompt=model_prompts[model_name],
                        max_tokens=budget,
                        task_id=task_id,
                    )
                    responses.append((model_name, resp))

                for model_name, resp in responses:
                    m = extract_metrics(
                        content=resp.content,
                        task_id=task_id,
                        task_family=family,
                        model=model_name,
                        budget=budget,
                        output_tokens=resp.output_tokens,
                    )
                    level_metrics.append(m)
                    all_runs.append(metrics_to_dict(m))

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
                    all_energy.append(energy_to_dict(em))

                    prior_strategies[(model_name, task_id)] = resp.content

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

                        if uor.available:
                            uor.address(resp.content, {
                                "model": model_name,
                                "task": task_id,
                                "generation": entry.generation,
                                "operator_count": m.operator_count,
                            })

                print(f"done ({len(responses)} models)")

        for model_name in active_models:
            model_metrics = [m for m in level_metrics if m.model == model_name]
            if model_metrics:
                conv = compute_convergence(model_metrics)
                key = f"{model_name}@{budget}"
                convergence_by_budget[key] = conv
                avg_ops = sum(m.operator_count for m in model_metrics) / len(model_metrics)
                print(f"\n  {model_name}: avg operators={avg_ops:.1f}, convergence={conv['jaccard_mean']:.3f}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    memory_report = memory.convergence_report()

    # Compute energy phase transitions per model
    energy_by_model = {}
    for e in all_energy:
        m = e["model"]
        b = e["budget"]
        energy_by_model.setdefault(m, {}).setdefault(b, []).append(e)

    phase_transitions = {}
    for model, budget_data in energy_by_model.items():
        budget_summary = {}
        for b, entries in budget_data.items():
            effs = [x["efficiency"] for x in entries]
            budget_summary[b] = {
                "mean_efficiency": sum(effs) / len(effs) if effs else 0,
                "mean_free_energy": sum(x["free_energy"] for x in entries) / len(entries),
                "mean_dissipation": sum(x["dissipation"] for x in entries) / len(entries),
                "total_cost": sum(x["dollar_cost"] for x in entries),
            }
        phase_transitions[model] = compute_phase_transition(budget_summary)

    results = {
        "experiment": "cdce_compression_harness",
        "timestamp": timestamp,
        "models": list(active_models.keys()),
        "budget_levels": BUDGET_LEVELS,
        "task_families": TASK_FAMILIES,
        "total_runs": len(all_runs),
        "runs": all_runs,
        "convergence_by_budget": convergence_by_budget,
        "energy": {
            "phase_transitions": phase_transitions,
            "by_model_budget": {
                m: {str(b): s for b, s in bd.items()}
                for m, bd in {
                    model: {
                        b: {
                            "mean_efficiency": sum(x["efficiency"] for x in entries) / len(entries),
                            "mean_free_energy": sum(x["free_energy"] for x in entries) / len(entries),
                            "mean_dissipation": sum(x["dissipation"] for x in entries) / len(entries),
                            "mean_quality": sum(x["quality"] for x in entries) / len(entries),
                            "total_cost": sum(x["dollar_cost"] for x in entries),
                        }
                        for b, entries in budget_data.items()
                    }
                    for model, budget_data in energy_by_model.items()
                }.items()
            },
        },
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

    generate_all_plots(results, RESULTS_DIR)
    memory.print_report()

    # Energy summary
    print("\n" + "=" * 60)
    print("ENERGY / THERMODYNAMIC SUMMARY")
    print("=" * 60)
    for model, pt in phase_transitions.items():
        print(f"\n  {model}:")
        if pt["detected"]:
            print(f"    ★ Phase transition detected at budget={pt['critical_budget']}")
            print(f"      Peak efficiency: {pt['peak_efficiency']:.4f}")
            print(f"      Efficiency drop: {pt['efficiency_drop']:.4f}")
        else:
            print(f"    No clear phase transition detected")
        if "efficiency_curve" in pt:
            print(f"    Efficiency curve: ", end="")
            for b, e in sorted(pt["efficiency_curve"].items(), key=lambda x: -int(x[0])):
                bar = "█" * int(e * 30)
                print(f"\n      {b:>6}: {e:.4f} {bar}", end="")
            print()
    print("=" * 60)

    print_summary(results)
    return results


def print_summary(results):
    print("\n" + "=" * 60)
    print("EXPERIMENT SUMMARY")
    print("=" * 60)

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
