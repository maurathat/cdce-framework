#!/usr/bin/env python3
"""
Floor-only probes for candidate Test 6.3 tasks.

Runs no-strategy baseline (10 trials each) on three harder corpus tasks:
  - opt_routing (TSP/route planning) — optimization family
  - pred_music (cyclic MIDI intervals) — prediction family
  - pred_math (powers-of-2 difference sequence) — prediction family

Reports: floor success rate and parse rate. Lowest floor = best candidate
for a full 6.3 run (most room for strategy lift).
"""

import re, sys, time

# ---- harness glue ---------------------------------------------------------
def solve(prompt, model_key):
    from src.llm_clients import call_model
    from src.config import MODELS
    cfg = MODELS[model_key]
    resp = call_model(
        provider=cfg["provider"],
        model=cfg["model"],
        prompt=prompt,
        max_tokens=2000,
        task_id="floor_probe",
    )
    return resp.content


# ---- TSP (opt_routing) ----------------------------------------------------
# Brute-force the optimal route length.
# Graph from the harness prompt (undirected; Depot=0, A=1..F=6):
DIST = {}
def _d(a, b, d):
    DIST[(a,b)] = d; DIST[(b,a)] = d
_d("Depot","A",10); _d("Depot","B",15); _d("Depot","C",20)
_d("A","B",12); _d("A","C",8); _d("A","D",15); _d("A","E",18); _d("A","F",22)
_d("B","C",10); _d("B","D",14); _d("B","E",9); _d("B","F",16)
_d("C","D",7); _d("C","E",11); _d("C","F",14)
_d("D","E",6); _d("D","F",13)
_d("E","F",8)

STOPS = ["A","B","C","D","E","F"]

def tsp_optimal():
    from itertools import permutations
    best = 9999
    for perm in permutations(STOPS):
        route = ["Depot"] + list(perm) + ["Depot"]
        cost = sum(DIST.get((route[i], route[i+1]), 9999) for i in range(len(route)-1))
        best = min(best, cost)
    return best

TSP_PROMPT = (
    "Find the shortest route visiting all 6 stops exactly once, starting and ending "
    "at the depot. Return ONLY the total distance as a single number.\n\n"
    "Distances: Depot-A:10, Depot-B:15, Depot-C:20, A-B:12, A-C:8, A-D:15, "
    "B-C:10, B-D:14, B-E:9, C-D:7, C-E:11, D-E:6, D-F:13, E-F:8, "
    "A-E:18, A-F:22, B-F:16, C-F:14."
)

def check_tsp(text, truth, tol=1):
    nums = re.findall(r"\b(\d+)\b", text)
    if not nums:
        return None, False
    # take the last number (models often reason then state final answer last)
    val = int(nums[-1])
    return val, abs(val - truth) <= tol


# ---- pred_music (cyclic MIDI intervals) -----------------------------------
# Pattern: +4, -2, +5, -3 repeating. Next 4 intervals: +5, -3, +4, -2
MUSIC_TRUTH = [5, -3, 4, -2]

MUSIC_PROMPT = (
    "A melody follows this pattern of intervals (in semitones):\n"
    "+4, -2, +5, -3, +4, -2, +5, -3, +4, -2, ?\n\n"
    "Predict the next 4 intervals. Return ONLY a comma-separated list of signed "
    "integers (e.g. +5, -3, +4, -2)."
)

def check_music(text, truth):
    # extract signed integers
    vals = re.findall(r"[+-]?\d+", text)
    if len(vals) < len(truth):
        return None, False
    # take the last len(truth) values
    got = [int(v) for v in vals[-len(truth):]]
    return got, got == truth


# ---- pred_math (powers-of-2 differences) ---------------------------------
# Sequence: 2, 6, 14, 30, 62, ... rule: a(n) = 2*a(n-1) + 2, or a(n) = 2^(n+1) - 2
# Next 3: 126, 254, 510
MATH_TRUTH = [126, 254, 510]

MATH_PROMPT = (
    "Find the pattern and predict the next 3 values:\n"
    "2, 6, 14, 30, 62, ?\n\n"
    "Return ONLY the next 3 values as a comma-separated list of integers."
)

def check_math(text, truth):
    vals = re.findall(r"\b(\d+)\b", text)
    if len(vals) < len(truth):
        return None, False
    # look for the truth subsequence anywhere in the extracted numbers
    ints = [int(v) for v in vals]
    for i in range(len(ints) - len(truth) + 1):
        if ints[i:i+len(truth)] == truth:
            return truth, True
    # fallback: take last 3
    got = ints[-len(truth):]
    return got, got == truth


# ---- run ------------------------------------------------------------------
def probe(name, prompt, checker, truth_arg, model, trials):
    print(f"\n--- {name} ---")
    ok = 0; parsed = 0
    for i in range(trials):
        ans = solve(prompt, model)
        result, correct = checker(ans, truth_arg)
        if result is not None:
            parsed += 1
        if correct:
            ok += 1
        time.sleep(0.3)
    floor = ok / trials
    parse_rate = parsed / trials
    print(f"  floor: {ok}/{trials} = {floor:.0%}   parse rate: {parse_rate:.0%}")
    return name, floor, parse_rate


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="claude")
    ap.add_argument("--trials", type=int, default=10)
    args = ap.parse_args()

    # compute TSP optimal once
    opt = tsp_optimal()
    print(f"TSP optimal distance (brute force): {opt}")

    results = []
    results.append(probe("opt_routing (TSP)", TSP_PROMPT, check_tsp, opt,
                         args.model, args.trials))
    results.append(probe("pred_music (MIDI intervals)", MUSIC_PROMPT, check_music,
                         MUSIC_TRUTH, args.model, args.trials))
    results.append(probe("pred_math (powers-of-2 seq)", MATH_PROMPT, check_math,
                         MATH_TRUTH, args.model, args.trials))

    print("\n=== SUMMARY ===")
    print(f"{'task':<30} | {'floor':>6} | {'parse':>6}")
    print("-" * 50)
    for name, floor, parse in results:
        print(f"{name:<30} | {floor:>5.0%} | {parse:>5.0%}")

    lowest = min(results, key=lambda r: r[1])
    print(f"\nLowest floor: {lowest[0]} at {lowest[1]:.0%}")
    print("-> Best candidate for full Test 6.3 run.")


if __name__ == "__main__":
    main()
