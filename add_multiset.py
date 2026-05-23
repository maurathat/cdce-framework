#!/usr/bin/env python3
"""
Re-extract operator_multiset from strategy_text for each memory entry.

Uses the exact same 70-verb vocabulary and inflection logic as
src/metrics.py:extract_metrics, but stores the full count-weighted list
(operator_multiset) rather than just the unique set.

Writes updated entries back to the memory directory in-place,
adding the operator_multiset field without touching other fields.
"""

import json
import os
import re
from collections import Counter

# Exact vocabulary from src/metrics.py
OPERATION_VERBS = {
    "compare", "evaluate", "assess", "check", "verify", "test", "measure",
    "rank", "score", "weigh",
    "break", "split", "separate", "decompose", "divide", "partition",
    "isolate", "extract",
    "combine", "merge", "join", "aggregate", "sum", "total", "accumulate",
    "integrate", "unify",
    "search", "find", "look", "scan", "explore", "try", "iterate",
    "enumerate", "traverse",
    "choose", "select", "pick", "decide", "assign", "allocate", "place",
    "swap", "move",
    "convert", "transform", "translate", "map", "encode", "decode",
    "reduce", "simplify", "compress", "abstract",
    "calculate", "compute", "add", "subtract", "multiply", "divide",
    "average", "minimize", "maximize", "optimize",
    "if", "then", "else", "because", "therefore", "since", "implies",
    "assume", "given", "conclude",
    "pattern", "rule", "formula", "sequence", "repeat", "recurse",
    "generalize", "observe", "notice", "identify",
}


def extract_multiset(text):
    """Return a list of verbs with repeats, matching metrics.py logic exactly."""
    cleaned = text.lower()
    for ch in ['*', '#', '`', '>', '|', '-', '_', ':', '"', "'",
               '(', ')', '[', ']', '{', '}', ',', '.', '!', '?',
               ';', '\n', '\t']:
        cleaned = cleaned.replace(ch, ' ')
    words = cleaned.split()

    multiset = []
    for verb in sorted(OPERATION_VERBS):
        forms = {verb, verb + "s", verb + "ed", verb + "ing", verb + "e",
                 verb + "es", verb + "d"}
        count = 0
        for form in forms:
            count += words.count(form)
        # metrics.py breaks after first matching form; we sum all forms
        # for the same base verb to get the true total-occurrence count
        for _ in range(count):
            multiset.append(verb)

    return multiset


def main():
    memdir = "memory"
    updated = 0
    skipped_no_text = 0
    total = 0

    for fn in sorted(os.listdir(memdir)):
        if not fn.endswith(".json") or fn == "memory_index.json":
            continue
        total += 1
        fpath = os.path.join(memdir, fn)
        with open(fpath) as f:
            entry = json.load(f)

        text = entry.get("strategy_text", "")
        if not text:
            skipped_no_text += 1
            continue

        ms = extract_multiset(text)
        entry["operator_multiset"] = ms
        with open(fpath, "w") as f:
            json.dump(entry, f, indent=2)
        updated += 1

    print(f"Total entries: {total}")
    print(f"Updated with operator_multiset: {updated}")
    print(f"Skipped (no strategy_text): {skipped_no_text}")

    # Sanity check: show one example
    sample = os.path.join(memdir, sorted(
        fn for fn in os.listdir(memdir)
        if fn.endswith(".json") and fn != "memory_index.json"
    )[0])
    with open(sample) as f:
        e = json.load(f)
    print(f"\nSample entry ({os.path.basename(sample)}):")
    print(f"  unique_verbs:      {e.get('unique_verbs')}")
    print(f"  operator_multiset: {e.get('operator_multiset')}")
    print(f"  len(multiset):     {len(e.get('operator_multiset', []))}")


if __name__ == "__main__":
    main()
