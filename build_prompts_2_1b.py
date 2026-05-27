#!/usr/bin/env python3
"""
build_prompts_2_1b.py
=====================
Build a stratified prompt set for Test 2.1b from the PLAIN GSM8K dataset.

WHY THIS EXISTS
  Test 2.1b correlates text-parsed operator count against activation
  effective-dimension. For that to work the prompt set must:
    (a) span a RANGE of reasoning-operation counts (the independent variable),
    (b) produce a RANGE of response lengths (so the length-control has variance).
  GSM8K problems take "between 2 and 8 steps" — so step count is a natural,
  citable proxy for operation count. This script stratifies the sample across
  low / medium / high step counts so the operator-count axis has deliberate
  spread instead of clustering.

DELIBERATE CHOICES (and their reasons)
  - Uses openai/gsm8k "main" — the PLAIN config, NOT the <thinking>/<reflection>
    CoT-formatted derivatives. Those impose a fixed reasoning FORMAT, which would
    contaminate both the operator count and the activation trajectory with the
    template's structure rather than the model's spontaneous reasoning. Plain only.
  - "Step count" is read from the reference SOLUTION (count of GSM8K's calculator
    annotations  <<...>>  , the gold-standard step markers). This is used ONLY to
    stratify the SAMPLE — it is NOT your operator metric and never enters the
    correlation. It's a sampling aid, full stop.
  - Deterministic: a fixed --seed gives the same sample every run, so the prompt
    set is reproducible and the provenance hash is stable.

SCOPE CAVEAT baked into the manifest: GSM8K is ARITHMETIC reasoning. A 2.1b result
on this set is about arithmetic operators specifically, not reasoning in general.

OUTPUT
  - data/prompts_2_1b.txt   : one prompt (question) per line — the format the
                              activation script reads.
  - data/prompts_2_1b.manifest.json : per-prompt record (gsm8k index, step bucket,
                              reference step count, char length of question) +
                              run params, so the sample is fully reproducible and
                              the sidecar can pin it.

USAGE
  pip install datasets
  python3 build_prompts_2_1b.py                 # 45 prompts, 15 per bucket, seed 42
  python3 build_prompts_2_1b.py --n-per-bucket 10 --seed 7 --split test
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections import defaultdict


# GSM8K marks each calculation step as <<expression=result>>. Counting these in
# the reference answer is the standard cheap proxy for "number of reasoning steps".
_STEP_MARK = re.compile(r"<<.*?>>")


def reference_step_count(answer: str) -> int:
    return len(_STEP_MARK.findall(answer))


def bucket_for(step_count: int) -> str:
    """Low / medium / high step-count buckets. GSM8K is ~2-8 steps."""
    if step_count <= 2:
        return "low"
    if step_count <= 4:
        return "medium"
    return "high"


def main():
    ap = argparse.ArgumentParser(description="Build stratified GSM8K prompt set for Test 2.1b")
    ap.add_argument("--dataset", default="openai/gsm8k")
    ap.add_argument("--config", default="main", help="use 'main' (plain), NOT a CoT variant")
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--n-per-bucket", type=int, default=15,
                    help="prompts per step-count bucket (low/medium/high). 15 -> 45 total")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--out-name", default="prompts_2_1b")
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        raise SystemExit("Install the datasets library first:  pip install datasets")

    print(f"[load] {args.dataset} ({args.config}) split={args.split}  "
          f"[cached at ~/.cache/huggingface/datasets after first download]")
    ds = load_dataset(args.dataset, args.config, split=args.split)

    # Guard against accidentally pointing at a CoT-formatted variant.
    cols = set(ds.column_names)
    if not {"question", "answer"}.issubset(cols):
        raise SystemExit(
            f"Dataset columns {sorted(cols)} are not plain GSM8K (expected question/answer). "
            "Use the plain 'openai/gsm8k' main config, not a <thinking>/CoT derivative."
        )

    # Bucket every example by reference step count, preserving original indices.
    buckets: dict[str, list[int]] = defaultdict(list)
    step_counts: dict[int, int] = {}
    for idx in range(len(ds)):
        sc = reference_step_count(ds[idx]["answer"])
        step_counts[idx] = sc
        buckets[bucket_for(sc)].append(idx)

    print("[strata] available per bucket: "
          + ", ".join(f"{b}={len(buckets[b])}" for b in ("low", "medium", "high")))

    # Deterministic stratified sample.
    import random
    rng = random.Random(args.seed)
    chosen: list[int] = []
    for b in ("low", "medium", "high"):
        pool = buckets[b]
        if not pool:
            print(f"[warn] bucket '{b}' empty — skipping")
            continue
        take = min(args.n_per_bucket, len(pool))
        if take < args.n_per_bucket:
            print(f"[warn] bucket '{b}' has only {len(pool)} (< {args.n_per_bucket}); taking all")
        chosen.extend(rng.sample(pool, take))
    rng.shuffle(chosen)  # mix buckets so order doesn't encode difficulty

    os.makedirs(args.outdir, exist_ok=True)
    prompts_path = os.path.join(args.outdir, f"{args.out_name}.txt")
    manifest_path = os.path.join(args.outdir, f"{args.out_name}.manifest.json")

    # Write prompts: one question per line. Newlines within a question are collapsed
    # so the "one prompt per line" contract the activation script relies on holds.
    records = []
    with open(prompts_path, "w", encoding="utf-8") as f:
        for idx in chosen:
            q = " ".join(ds[idx]["question"].split())  # collapse internal whitespace/newlines
            f.write(q + "\n")
            records.append({
                "gsm8k_index": idx,
                "step_bucket": bucket_for(step_counts[idx]),
                "reference_step_count": step_counts[idx],
                "question_char_len": len(q),
            })

    manifest = {
        "schema": "cdce.test_2_1b.prompt_manifest/v1",
        "source": {"dataset": args.dataset, "config": args.config, "split": args.split},
        "sampling": {
            "seed": args.seed,
            "n_per_bucket": args.n_per_bucket,
            "n_total": len(records),
            "buckets": "low(<=2 steps) / medium(3-4) / high(5+)",
        },
        "scope_caveat": (
            "GSM8K is ARITHMETIC reasoning. A 2.1b result on this set concerns arithmetic "
            "operators specifically, not reasoning in general. reference_step_count is a "
            "SAMPLING aid only — it is NOT the operator metric and does not enter the correlation."
        ),
        "prompts_file": os.path.basename(prompts_path),
        "records": records,
    }
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, sort_keys=True)

    # Report the spread you actually got — this is what makes/breaks the length control.
    by_bucket = defaultdict(int)
    for r in records:
        by_bucket[r["step_bucket"]] += 1
    lens = [r["question_char_len"] for r in records]
    print(f"\n[out] {prompts_path}  ({len(records)} prompts)")
    print(f"[out] {manifest_path}")
    print(f"[strata] sampled: " + ", ".join(f"{b}={by_bucket[b]}" for b in ("low","medium","high")))
    print(f"[len] question chars: min={min(lens)} max={max(lens)} "
          f"mean={sum(lens)//len(lens)}")
    print("\n[next] swap into the activation script:  --prompts "
          f"{prompts_path}")
    print("[next] then the provenance sidecar will hash this exact prompt file.")


if __name__ == "__main__":
    main()
