"""
CDCE Compression Harness — Visualization

Generates plots of compression metrics across budget levels and models.
Key plots:
1. Operator count vs budget (the primary CDCE curve)
2. Cross-family convergence vs budget
3. Reuse ratio vs budget
4. Multi-model comparison overlay
"""
import os
import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

matplotlib.use("Agg")  # Non-interactive backend for file output


def plot_operator_curves(results: dict, output_dir: str):
    """
    Plot operator count vs budget level for each model.
    CDCE predicts: power-law decay with plateaus at stage transitions.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    models = {}
    for entry in results.get("runs", []):
        model = entry["model"]
        budget = entry["budget"]
        op_count = entry["operator_count"]
        models.setdefault(model, {"budgets": [], "ops": []})
        models[model]["budgets"].append(budget)
        models[model]["ops"].append(op_count)

    colors = ["#E8403F", "#3B82F6", "#10B981", "#F59E0B", "#8B5CF6"]
    for i, (model, data) in enumerate(sorted(models.items())):
        # Average across tasks at each budget level
        budget_ops = {}
        for b, o in zip(data["budgets"], data["ops"]):
            budget_ops.setdefault(b, []).append(o)
        budgets = sorted(budget_ops.keys(), reverse=True)
        means = [np.mean(budget_ops[b]) for b in budgets]
        stds = [np.std(budget_ops[b]) for b in budgets]

        color = colors[i % len(colors)]
        ax.errorbar(
            budgets, means, yerr=stds,
            marker="o", linewidth=2, markersize=8,
            color=color, label=model, capsize=4,
        )

    ax.set_xlabel("Token Budget", fontsize=12)
    ax.set_ylabel("Distinct Operator Count", fontsize=12)
    ax.set_title("CDCE Prediction: Operator Reduction Under Compression", fontsize=14)
    ax.legend(fontsize=10)
    ax.invert_xaxis()  # Higher budget on left, compressed on right
    ax.grid(True, alpha=0.3)

    # Add CDCE stage annotations
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
    ax.text(
        0.98, 0.02,
        "→ Increasing compression pressure",
        transform=ax.transAxes, ha="right", va="bottom",
        fontsize=9, color="gray", style="italic",
    )

    plt.tight_layout()
    path = os.path.join(output_dir, "operator_reduction.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_convergence(results: dict, output_dir: str):
    """
    Plot cross-family Jaccard convergence vs budget.
    CDCE predicts: convergence increases as budget decreases.
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    convergence = results.get("convergence_by_budget", {})
    if not convergence:
        print("  No convergence data to plot.")
        return

    # Group by model
    model_data = {}
    for key, data in convergence.items():
        # key format: "model_name@budget"
        parts = key.rsplit("@", 1)
        if len(parts) != 2:
            continue
        model, budget = parts[0], int(parts[1])
        model_data.setdefault(model, {"budgets": [], "jaccard": []})
        model_data[model]["budgets"].append(budget)
        model_data[model]["jaccard"].append(data.get("jaccard_mean", 0))

    colors = ["#E8403F", "#3B82F6", "#10B981", "#F59E0B"]
    for i, (model, data) in enumerate(sorted(model_data.items())):
        budgets = data["budgets"]
        jaccards = data["jaccard"]
        # Sort by budget descending
        paired = sorted(zip(budgets, jaccards), reverse=True)
        budgets = [p[0] for p in paired]
        jaccards = [p[1] for p in paired]

        color = colors[i % len(colors)]
        ax.plot(
            budgets, jaccards,
            marker="s", linewidth=2, markersize=8,
            color=color, label=model,
        )

    ax.set_xlabel("Token Budget", fontsize=12)
    ax.set_ylabel("Cross-Family Jaccard Similarity", fontsize=12)
    ax.set_title("CDCE Prediction: Strategy Convergence Across Task Families", fontsize=14)
    ax.legend(fontsize=10)
    ax.invert_xaxis()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "convergence.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_reuse_ratio(results: dict, output_dir: str):
    """
    Plot reuse ratio vs budget.
    Higher reuse = more compression (fewer unique ops, more repetition).
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    models = {}
    for entry in results.get("runs", []):
        model = entry["model"]
        budget = entry["budget"]
        reuse = entry["reuse_ratio"]
        models.setdefault(model, {"budgets": [], "reuse": []})
        models[model]["budgets"].append(budget)
        models[model]["reuse"].append(reuse)

    colors = ["#E8403F", "#3B82F6", "#10B981", "#F59E0B"]
    for i, (model, data) in enumerate(sorted(models.items())):
        budget_reuse = {}
        for b, r in zip(data["budgets"], data["reuse"]):
            budget_reuse.setdefault(b, []).append(r)
        budgets = sorted(budget_reuse.keys(), reverse=True)
        means = [np.mean(budget_reuse[b]) for b in budgets]

        color = colors[i % len(colors)]
        ax.plot(
            budgets, means,
            marker="D", linewidth=2, markersize=8,
            color=color, label=model,
        )

    ax.set_xlabel("Token Budget", fontsize=12)
    ax.set_ylabel("Reuse Ratio (higher = more compressed)", fontsize=12)
    ax.set_title("Operation Reuse Under Compression Pressure", fontsize=14)
    ax.legend(fontsize=10)
    ax.invert_xaxis()
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "reuse_ratio.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def plot_summary_dashboard(results: dict, output_dir: str):
    """Generate a combined 2x2 dashboard."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("CDCE Compression Harness — Experiment Results", fontsize=16, y=0.98)

    # Collect model data
    model_ops = {}
    model_reuse = {}
    model_steps = {}
    for entry in results.get("runs", []):
        m = entry["model"]
        b = entry["budget"]
        model_ops.setdefault(m, {}).setdefault(b, []).append(entry["operator_count"])
        model_reuse.setdefault(m, {}).setdefault(b, []).append(entry["reuse_ratio"])
        model_steps.setdefault(m, {}).setdefault(b, []).append(entry["step_count"])

    colors = ["#E8403F", "#3B82F6", "#10B981", "#F59E0B"]

    # Panel 1: Operator count
    ax = axes[0][0]
    for i, (model, bdata) in enumerate(sorted(model_ops.items())):
        budgets = sorted(bdata.keys(), reverse=True)
        means = [np.mean(bdata[b]) for b in budgets]
        ax.plot(budgets, means, marker="o", color=colors[i % 4], label=model, linewidth=2)
    ax.set_title("Operator Count")
    ax.set_xlabel("Budget")
    ax.invert_xaxis()
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Reuse ratio
    ax = axes[0][1]
    for i, (model, bdata) in enumerate(sorted(model_reuse.items())):
        budgets = sorted(bdata.keys(), reverse=True)
        means = [np.mean(bdata[b]) for b in budgets]
        ax.plot(budgets, means, marker="s", color=colors[i % 4], label=model, linewidth=2)
    ax.set_title("Reuse Ratio")
    ax.set_xlabel("Budget")
    ax.set_ylim(0, 1)
    ax.invert_xaxis()
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 3: Step count
    ax = axes[1][0]
    for i, (model, bdata) in enumerate(sorted(model_steps.items())):
        budgets = sorted(bdata.keys(), reverse=True)
        means = [np.mean(bdata[b]) for b in budgets]
        ax.plot(budgets, means, marker="D", color=colors[i % 4], label=model, linewidth=2)
    ax.set_title("Reasoning Steps")
    ax.set_xlabel("Budget")
    ax.invert_xaxis()
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 4: Convergence
    ax = axes[1][1]
    convergence = results.get("convergence_by_budget", {})
    conv_model_data = {}
    for key, data in convergence.items():
        parts = key.rsplit("@", 1)
        if len(parts) != 2:
            continue
        model, budget = parts[0], int(parts[1])
        conv_model_data.setdefault(model, {"b": [], "j": []})
        conv_model_data[model]["b"].append(budget)
        conv_model_data[model]["j"].append(data.get("jaccard_mean", 0))

    for i, (model, d) in enumerate(sorted(conv_model_data.items())):
        paired = sorted(zip(d["b"], d["j"]), reverse=True)
        ax.plot(
            [p[0] for p in paired], [p[1] for p in paired],
            marker="^", color=colors[i % 4], label=model, linewidth=2,
        )
    ax.set_title("Cross-Family Convergence")
    ax.set_xlabel("Budget")
    ax.set_ylim(0, 1)
    ax.invert_xaxis()
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(output_dir, "dashboard.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


def generate_all_plots(results: dict, output_dir: str):
    """Generate all visualization plots."""
    os.makedirs(output_dir, exist_ok=True)
    print("\nGenerating plots...")
    plot_operator_curves(results, output_dir)
    plot_convergence(results, output_dir)
    plot_reuse_ratio(results, output_dir)
    plot_summary_dashboard(results, output_dir)
    print("Done.\n")
