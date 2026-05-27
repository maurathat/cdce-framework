#!/usr/bin/env python3
"""Pre-sweep checks: Claude cumavg floor (20 trials) + GPT-4o routing error diagnosis."""

import re, time

def solve(prompt, model_key, task_id="precheck"):
    from src.llm_clients import call_model
    from src.config import MODELS
    cfg = MODELS[model_key]
    resp = call_model(provider=cfg["provider"], model=cfg["model"],
                      prompt=prompt, max_tokens=2000, task_id=task_id)
    return resp.content

def is_api_error(text):
    if not text or len(text.strip()) < 5:
        return True
    t = text.lower()
    return any(s in t for s in ["[error", "credit balance", "invalid_request_error",
                                 "rate_limit", "server_error", "api_key",
                                 "status_code", "authenticate"])

# ---- Check 1: Claude cumavg floor at 20 trials ----
CUMAVG_TRUTH = [4.0, 6.0, 6.0, 5.0, 6.0]
CUMAVG_PROMPT = ("Given the list [4, 8, 6, 2, 10], return the cumulative average at "
                 "each position (running sum divided by count so far). Return ONLY a "
                 "Python list of numbers.")

def check_cumavg(text):
    m = re.search(r"\[([^\[\]]+)\]", text)
    span = m.group(1) if m else text
    vals = [float(v) for v in re.findall(r"-?\d+\.?\d*", span)]
    if len(vals) != len(CUMAVG_TRUTH):
        return False
    return all(abs(a - b) <= 0.01 for a, b in zip(vals, CUMAVG_TRUTH))

print("=== Check 1: Claude cumavg floor (20 trials) ===")
ok = errors = 0
for i in range(20):
    ans = solve(CUMAVG_PROMPT, "claude", task_id="cumavg")
    if is_api_error(ans):
        errors += 1
        print(f"  trial {i+1}: API ERROR")
    elif check_cumavg(ans):
        ok += 1
    else:
        print(f"  trial {i+1}: WRONG — {ans[:100]}")
    time.sleep(0.3)
valid = 20 - errors
rate = ok / valid if valid else 0
print(f"\n  Result: {ok}/{valid} correct ({rate:.0%}), {errors} errors")
print(f"  Earlier full run: 70%. This probe: {rate:.0%}.")
if rate > 0.85:
    print("  -> Claude is ABOVE on cumavg. The 90% probe was real.")
elif rate > 0.50:
    print("  -> Claude is MID on cumavg. Split is weaker than it looked.")
else:
    print("  -> Claude is BELOW on cumavg. Earlier 90% was noise.")

# ---- Check 2: GPT-4o routing error diagnosis ----
ROUTING_PROMPT = (
    "Find the shortest route visiting all 6 stops exactly once, starting and ending "
    "at the depot. Return ONLY the total distance as a single number.\n\n"
    "Distances: Depot-A:10, Depot-B:15, Depot-C:20, A-B:12, A-C:8, A-D:15, "
    "B-C:10, B-D:14, B-E:9, C-D:7, C-E:11, D-E:6, D-F:13, E-F:8, "
    "A-E:18, A-F:22, B-F:16, C-F:14.")

print("\n=== Check 2: GPT-4o routing (10 trials, print errors) ===")
gpt_ok = gpt_errors = 0
for i in range(10):
    ans = solve(ROUTING_PROMPT, "gpt4o", task_id="routing")
    if is_api_error(ans):
        gpt_errors += 1
        print(f"  trial {i+1}: API ERROR — {ans[:200]}")
    else:
        nums = re.findall(r"\b(\d+)\b", ans)
        val = int(nums[-1]) if nums else None
        correct = val is not None and abs(val - 70) <= 1
        if correct:
            gpt_ok += 1
        print(f"  trial {i+1}: {ans[:100]}  -> parsed={val}, correct={correct}")
    time.sleep(0.3)
gpt_valid = 10 - gpt_errors
print(f"\n  Result: {gpt_ok}/{gpt_valid} correct, {gpt_errors} errors")
if gpt_errors > 2:
    print("  -> GPT-4o routing errors are PERSISTENT, not transient. Check config.")
else:
    print("  -> Errors were transient. GPT-4o routing is usable.")
