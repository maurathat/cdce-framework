#!/usr/bin/env python3
"""
Prompt Ablation Test
====================
Tests whether low operator counts at budget=125 are a compression attractor
or a prompt artifact.

Three conditions:
  A) COMPRESS: current prompt with "use fewest operations" (system + user)
  B) BUDGET-ONLY: token limit mentioned in prompt, no operation pressure
  C) NEUTRAL: bare task, max_tokens=125 is the only constraint

All run at budget=125 on Claude Sonnet across all 9 tasks.
If B and C also show low operator counts, the attractor is real.
If only A shows it, it's a prompt artifact.
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path

from src.config import ANTHROPIC_API_KEY, RESULTS_DIR
from src.tasks import ALL_TASKS
from src.metrics import extract_metrics, metrics_to_dict

BUDGET = 125
MODEL = "claude-sonnet-4-20250514"


# ── Three prompt modes ─────────────────────────────────────────────────────

SYSTEM_PROMPTS = {
    "A_compress": (
        "You are solving a task under strict token constraints. "
        "Be maximally compressed and efficient. Show your reasoning, "
        "but use the fewest possible distinct operations and steps."
    ),
    "B_budget_only": (
        "You are solving a task. Answer within the space available."
    ),
    "C_neutral": (
        "You are a helpful assistant."
    ),
}


def build_user_prompt(task: dict, mode: str) -> str:
    """Build user prompt for each mode."""
    if mode == "A_compress":
        return (
            task["prompt"]
            + f"\n\n[CONSTRAINT: Your COMPLETE response must be under {BUDGET} tokens. "
            f"Be maximally compressed. Every word must earn its place.]"
        )
    elif mode == "B_budget_only":
        return (
            task["prompt"]
            + f"\n\n[Note: Please keep your response under {BUDGET} tokens.]"
        )
    elif mode == "C_neutral":
        return task["prompt"]
    else:
        raise ValueError(f"Unknown mode: {mode}")


def call_anthropic_with_mode(task: dict, mode: str):
    """Call Claude Sonnet with the given prompt mode."""
    import anthropic

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    system = SYSTEM_PROMPTS[mode]
    user = build_user_prompt(task, mode)

    start = time.time()
    response = client.messages.create(
        model=MODEL,
        max_tokens=BUDGET,
        messages=[{"role": "user", "content": user}],
        system=system,
    )
    latency = (time.time() - start) * 1000

    content = response.content[0].text if response.content else ""
    return {
        "content": content,
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
        "latency_ms": latency,
    }


def main():
    if not ANTHROPIC_API_KEY or len(ANTHROPIC_API_KEY) < 10:
        print("ERROR: ANTHROPIC_API_KEY not set")
        return

    print("=" * 70)
    print("PROMPT ABLATION TEST")
    print(f"Budget={BUDGET}, Model={MODEL}")
    print("=" * 70)

    modes = ["A_compress", "B_budget_only", "C_neutral"]
    all_results = []

    for family_name, tasks in ALL_TASKS.items():
        for task in tasks:
            print(f"\n{'─' * 60}")
            print(f"Task: {task['name']} ({task['id']}) — family: {family_name}")
            print(f"{'─' * 60}")

            for mode in modes:
                print(f"  Mode {mode}...", end=" ", flush=True)

                try:
                    resp = call_anthropic_with_mode(task, mode)
                except Exception as e:
                    print(f"ERROR: {e}")
                    all_results.append({
                        "task_id": task["id"],
                        "task_name": task["name"],
                        "family": family_name,
                        "mode": mode,
                        "error": str(e),
                    })
                    continue

                m = extract_metrics(
                    content=resp["content"],
                    task_id=task["id"],
                    task_family=family_name,
                    model="claude",
                    budget=BUDGET,
                    output_tokens=resp["output_tokens"],
                )

                result = {
                    "task_id": task["id"],
                    "task_name": task["name"],
                    "family": family_name,
                    "mode": mode,
                    "operator_count": m.operator_count,
                    "unique_verbs": m.unique_verbs,
                    "word_count": m.word_count,
                    "output_tokens": resp["output_tokens"],
                    "reuse_ratio": m.reuse_ratio,
                    "content": resp["content"],
                }
                all_results.append(result)
                print(f"ops={m.operator_count}, verbs={m.unique_verbs}, "
                      f"words={m.word_count}, tokens={resp['output_tokens']}")

    # ── Analysis ────────────────────────────────────────────────────────────
    print(f"\n{'=' * 80}")
    print("PROMPT ABLATION RESULTS")
    print(f"{'─' * 80}")

    # Group by mode
    by_mode = {}
    for r in all_results:
        if "error" in r:
            continue
        by_mode.setdefault(r["mode"], []).append(r)

    print(f"\n  {'Mode':<20} | {'Mean Ops':>9} | {'Median Ops':>11} | "
          f"{'Mean Words':>11} | {'Mean Tokens':>12} | {'N':>3}")
    print(f"  {'─' * 72}")

    mode_summaries = {}
    for mode in modes:
        entries = by_mode.get(mode, [])
        if not entries:
            continue
        ops = [e["operator_count"] for e in entries]
        words = [e["word_count"] for e in entries]
        tokens = [e["output_tokens"] for e in entries]
        import numpy as np
        summary = {
            "mean_ops": float(np.mean(ops)),
            "median_ops": float(np.median(ops)),
            "std_ops": float(np.std(ops)),
            "mean_words": float(np.mean(words)),
            "mean_tokens": float(np.mean(tokens)),
            "n": len(entries),
            "ops_list": ops,
        }
        mode_summaries[mode] = summary
        print(f"  {mode:<20} | {summary['mean_ops']:>9.1f} | {summary['median_ops']:>11.1f} | "
              f"{summary['mean_words']:>11.1f} | {summary['mean_tokens']:>12.1f} | {summary['n']:>3}")

    print(f"  {'─' * 72}")

    # Per-task comparison
    print(f"\n  Per-task operator counts:")
    print(f"  {'Task':<20} | {'A (compress)':>13} | {'B (budget)':>13} | {'C (neutral)':>13}")
    print(f"  {'─' * 65}")

    task_ids = sorted(set(r["task_id"] for r in all_results if "error" not in r))
    for tid in task_ids:
        row = {}
        for r in all_results:
            if r.get("task_id") == tid and "error" not in r:
                row[r["mode"]] = r["operator_count"]
        a_ops = row.get("A_compress", "—")
        b_ops = row.get("B_budget_only", "—")
        c_ops = row.get("C_neutral", "—")
        print(f"  {tid:<20} | {str(a_ops):>13} | {str(b_ops):>13} | {str(c_ops):>13}")

    # Verdict
    print(f"\n  {'─' * 65}")
    if mode_summaries:
        a_mean = mode_summaries.get("A_compress", {}).get("mean_ops", 0)
        b_mean = mode_summaries.get("B_budget_only", {}).get("mean_ops", 0)
        c_mean = mode_summaries.get("C_neutral", {}).get("mean_ops", 0)

        print(f"\n  Mean operator counts: A={a_mean:.1f}, B={b_mean:.1f}, C={c_mean:.1f}")

        if b_mean <= a_mean * 1.3 and c_mean <= a_mean * 1.3:
            verdict = "ATTRACTOR IS REAL — low ops persist without compression prompt"
        elif b_mean <= a_mean * 1.3:
            verdict = "PARTIAL — budget constraint alone drives low ops"
        elif c_mean > a_mean * 1.5:
            verdict = "PROMPT ARTIFACT — only explicit compression instruction drives low ops"
        else:
            verdict = "AMBIGUOUS — need more data"

        print(f"  VERDICT: {verdict}")
    print(f"{'=' * 80}")

    # Save
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Strip raw content for JSON (keep it small)
    json_results = []
    for r in all_results:
        entry = {k: v for k, v in r.items() if k != "content"}
        json_results.append(entry)

    output = {
        "experiment": "prompt_ablation",
        "timestamp": timestamp,
        "budget": BUDGET,
        "model": MODEL,
        "modes": modes,
        "results": json_results,
        "summaries": mode_summaries,
    }

    out_path = os.path.join(RESULTS_DIR, f"prompt_ablation_{timestamp}.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
