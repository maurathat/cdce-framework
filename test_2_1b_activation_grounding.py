#!/usr/bin/env python3
"""
test_2_1b_activation_grounding.py
=================================
Test 2.1b — Activation Grounding Probe (sourcing half).

Question: does the operator text-count (n_text) correlate with the
effective dimensionality (d_act) of the model's hidden-state activations?

This is the sourcing half only.  It does NOT validate the operator metric
against human judgment (validity half, still requires annotation).  It does
NOT address any JEPA question (Gemma is an autoregressive LLM, not a JEPA).

Method
------
1. Feed each prompt through an open-weight model (default: google/gemma-2-2b).
2. Generate a completion and collect hidden-state activations.
3. Compute effective dimensionality via participation ratio:
       d_act = (sum lambda_i)^2 / sum(lambda_i^2)
   where lambda_i are eigenvalues of the activation covariance matrix.
4. Count distinct CDCE operation verbs in the generated text (n_text).
5. Compute raw and partial correlations (controlling for generated token count).
6. Emit results CSV + provenance sidecar.

Usage
-----
    python3 test_2_1b_activation_grounding.py --device mps     # real run: gemma-2-2b-it + 45 GSM8K prompts
    python3 test_2_1b_activation_grounding.py --smoke --device mps   # pipeline smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter

import numpy as np
import pandas as pd

# ── Operator verb vocabulary ────────────────────────────────────────────────
# Canonical source: src/metrics.py OPERATION_VERBS.
# Duplicated here so the script stays self-contained (no harness import
# required).  The provenance sidecar hashes the parser source, so any
# divergence between this copy and src/metrics.py is detectable.

OPERATION_VERBS: set[str] = {
    # Comparison / evaluation
    "compare", "evaluate", "assess", "check", "verify", "test", "measure",
    "rank", "score", "weigh",
    # Decomposition
    "break", "split", "separate", "decompose", "divide", "partition",
    "isolate", "extract",
    # Composition
    "combine", "merge", "join", "aggregate", "sum", "total", "accumulate",
    "integrate", "unify",
    # Search
    "search", "find", "look", "scan", "explore", "try", "iterate",
    "enumerate", "traverse",
    # Selection
    "choose", "select", "pick", "decide", "assign", "allocate", "place",
    "swap", "move",
    # Transformation
    "convert", "transform", "translate", "map", "encode", "decode",
    "reduce", "simplify", "compress", "abstract",
    # Computation
    "calculate", "compute", "add", "subtract", "multiply", "divide",
    "average", "minimize", "maximize", "optimize",
    # Logical
    "if", "then", "else", "because", "therefore", "since", "implies",
    "assume", "given", "conclude",
    # Pattern
    "pattern", "rule", "formula", "sequence", "repeat", "recurse",
    "generalize", "observe", "notice", "identify",
}


# ── Operator parser ────────────────────────────────────────────────────────
def _clean_words(text: str) -> list[str]:
    """Lowercase, strip punctuation, split — mirrors src/metrics.py cleaning."""
    cleaned = text.lower()
    for ch in (
        "*", "#", "`", ">", "|", "-", "_", ":", '"', "'",
        "(", ")", "[", "]", "{", "}", ",", ".", "!", "?", ";", "\n", "\t",
    ):
        cleaned = cleaned.replace(ch, " ")
    return cleaned.split()


def parse_operator_count(text: str) -> tuple[int, dict]:
    """
    Count distinct CDCE operator verbs in *text*.

    Uses the same inflection logic as src/metrics.extract_metrics so the
    two parsers produce identical operator_count for the same input.

    Returns (n_distinct_verbs, {verb: count, ...}).
    """
    words = _clean_words(text)
    verb_counts: Counter[str] = Counter()

    for verb in OPERATION_VERBS:
        forms = {verb, verb + "s", verb + "ed", verb + "ing",
                 verb + "e", verb + "es", verb + "d"}
        for form in forms:
            count = words.count(form)
            if count > 0:
                verb_counts[verb] += count
                break

    return len(verb_counts), dict(verb_counts)


def parse_operator_count_fallback(text: str) -> tuple[int, dict]:
    """Trivial word-intersection fallback.  Named so provenance flags it."""
    words = set(_clean_words(text))
    found = words & OPERATION_VERBS
    return len(found), {v: 1 for v in sorted(found)}


# ── Activation extraction ──────────────────────────────────────────────────
def load_model(model_name: str, device: str):
    """Load model + tokenizer."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=device if device != "cpu" else None,
    )
    if device == "cpu":
        model = model.to("cpu")
    model.eval()
    return model, tokenizer


def format_prompt(tokenizer, prompt: str, device) -> tuple:
    """
    Apply chat template if the tokenizer has one (TinyLlama-Chat, gemma-2-2b-it,
    etc.), otherwise tokenize raw.

    Returns (input_ids: Tensor[1, seq_len], prompt_len: int).
    prompt_len is the POST-template token count — the correct slice boundary
    so that template tokens are never leaked into the activation matrix.
    """
    import torch

    if getattr(tokenizer, "chat_template", None):
        chat = [{"role": "user", "content": prompt}]
        result = tokenizer.apply_chat_template(
            chat, return_tensors="pt", add_generation_prompt=True,
        )
        # transformers 5.x may return BatchEncoding instead of a bare tensor
        input_ids = (result["input_ids"] if hasattr(result, "keys") else result).to(device)
    else:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    return input_ids, int(input_ids.shape[1])


def generate_and_get_hidden_states(
    model, tokenizer, prompt: str, max_new_tokens: int, seed: int,
    device: str, layer_mode: str, n_layers: int,
) -> tuple[str, np.ndarray, dict]:
    """
    Generate text, then do a CLEAN re-forward pass of the full sequence
    to collect hidden states.

    Why re-forward instead of generation-time cache:
      - generation cache contains KV states, not per-token hidden vectors
        from a single clean pass
      - a re-forward with output_hidden_states=True, use_cache=False gives
        the full (n_layers+1, 1, seq_len, hidden_dim) tuple in one shot
      - slicing at prompt_len (post-template) ensures only generated-token
        activations enter the PR calculation

    Returns (generated_text, gen_activations[n_gen_tokens, hidden_dim], token_counts).
    """
    import torch

    torch.manual_seed(seed)
    input_ids, prompt_len = format_prompt(tokenizer, prompt, device)

    # ── generate ─────────────────────────────────────────────────────────
    with torch.no_grad():
        gen_output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    full_ids = gen_output                           # [1, prompt_len + gen_len]
    generated_text = tokenizer.decode(full_ids[0, prompt_len:], skip_special_tokens=True)
    gen_token_count = int(full_ids.shape[1] - prompt_len)

    # ── clean re-forward for hidden states ───────────────────────────────
    with torch.no_grad():
        fwd = model(
            input_ids=full_ids,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

    # fwd.hidden_states is a tuple of (n_layers+1) tensors, each [1, seq_len, dim].
    # Index 0 = embedding layer; 1..n_layers = transformer layers.
    gen_activations = select_layer_activations(
        fwd.hidden_states, layer_mode, n_layers, prompt_len,
    )

    token_counts = {
        "prompt_token_count": prompt_len,
        "generated_token_count": gen_token_count,
        "total_token_count": int(full_ids.shape[1]),
    }
    return generated_text, gen_activations, token_counts


def select_layer_activations(
    hidden_states: tuple, layer_mode: str, n_layers: int, prompt_len: int,
) -> np.ndarray:
    """
    Extract generated-token activations from a clean forward pass.

    hidden_states: tuple of (n_layers+1) tensors, each [1, seq_len, hidden_dim].
    Returns ndarray [n_generated_tokens, hidden_dim].

    layer_mode
      "last"    — final transformer layer only.
      "midmean" — mean of the middle third of layers.
      "allmean"  — mean of all transformer layers (skip embedding layer 0).
    """
    import torch

    # Select which layers to average.
    if layer_mode == "last":
        layer_indices = [n_layers]
    elif layer_mode == "midmean":
        third = max(1, n_layers // 3)
        layer_indices = list(range(third, 2 * third + 1))
    elif layer_mode == "allmean":
        layer_indices = list(range(1, n_layers + 1))
    else:
        raise ValueError(f"unknown layer_mode: {layer_mode}")

    # Stack selected layers → [n_selected, 1, seq_len, dim], mean → [1, seq_len, dim]
    stacked = torch.stack([hidden_states[li] for li in layer_indices], dim=0)
    averaged = stacked.mean(dim=0)

    # Slice to generated tokens only, drop batch dim → [n_gen_tokens, dim]
    gen_hidden = averaged[0, prompt_len:, :].to(torch.float32).cpu().numpy()

    if gen_hidden.shape[0] == 0:
        return np.zeros((1, gen_hidden.shape[1]))
    return gen_hidden


def participation_ratio(activations: np.ndarray) -> float:
    """
    Effective dimensionality via participation ratio:
        PR = (sum lambda_i)^2 / sum(lambda_i^2)

    PR >= 1.  PR == 1 when all variance lies on a single axis.
    PR == n  when variance is uniform across n dimensions.
    Note: PR <= n_tokens (the number of rows), which is a hard ceiling
    acknowledged in the provenance sidecar.
    """
    n = activations.shape[0]
    if n < 2:
        return 1.0

    centered = activations - activations.mean(axis=0, keepdims=True)

    # Use the Gram matrix when n < d (cheaper eigendecomposition).
    _, d = centered.shape
    if n < d:
        gram = (centered @ centered.T) / (n - 1)
        eigvals = np.linalg.eigvalsh(gram)
    else:
        cov = (centered.T @ centered) / (n - 1)
        eigvals = np.linalg.eigvalsh(cov)

    eigvals = eigvals[eigvals > 0]
    if len(eigvals) == 0:
        return 1.0

    s1 = eigvals.sum()
    s2 = (eigvals ** 2).sum()
    return float(s1 ** 2 / s2)


# ── Partial correlation ────────────────────────────────────────────────────
def partial_corr(x, y, z, method: str = "pearson"):
    """
    Partial correlation of *x* and *y* controlling for *z*.
    Returns (r, p-value).
    """
    from scipy import stats

    x, y, z = np.asarray(x, dtype=float), np.asarray(y, dtype=float), np.asarray(z, dtype=float)

    if method == "spearman":
        x, y, z = stats.rankdata(x), stats.rankdata(y), stats.rankdata(z)

    z_design = np.column_stack([np.ones(len(z)), z])
    z_pinv = np.linalg.pinv(z_design)
    x_resid = x - (z_design @ (z_pinv @ x)).ravel()
    y_resid = y - (z_design @ (z_pinv @ y)).ravel()

    r, p = stats.pearsonr(x_resid, y_resid)
    return float(r), float(p)


# ── Prompts ────────────────────────────────────────────────────────────────
SMOKE_PROMPTS = [
    "Explain how to sort a list of numbers efficiently.",
    "Compare and contrast two approaches to solving a maze.",
    "Break down the steps to compute a running average of a stream of values.",
    "Describe how to find the shortest route between six cities.",
    "Translate this math into plain English: y = (1/n) * sum(x_i, i=1..n).",
    "Combine flour, sugar, butter, and eggs into a recipe. List every step.",
    "Simplify the expression (a + b)^2 - (a - b)^2 and show each algebraic step.",
    "Decompose the problem of scheduling five meetings into three rooms with no overlaps.",
]


def load_prompts(path: str) -> list[str]:
    """One prompt per line (.txt) or a JSON list (.json)."""
    if path.endswith(".json"):
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, list):
            return [str(p).strip() for p in data if str(p).strip()]
        raise ValueError(f"expected a JSON list in {path}")
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Test 2.1b — activation grounding probe")
    ap.add_argument("--model", default="google/gemma-2-2b-it",
                    help="HuggingFace model id (default: google/gemma-2-2b-it)")
    ap.add_argument("--prompts", default="data/prompts_2_1b.txt",
                    help="Prompt file (.txt one-per-line, .json list). "
                         "Default: data/prompts_2_1b.txt (45 stratified GSM8K)")
    ap.add_argument("--smoke", action="store_true",
                    help="Use built-in 8-prompt smoke set (overrides --prompts)")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--layer-mode", default="last",
                    choices=["last", "midmean", "allmean"])
    ap.add_argument("--device", default="cpu",
                    help="cpu | cuda | mps")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="results_2_1b.csv",
                    help="Output CSV path (default: results_2_1b.csv)")
    args = ap.parse_args()

    # ── resolve prompts ──────────────────────────────────────────────────
    if args.smoke:
        prompts = SMOKE_PROMPTS
        prompts_path = None
    else:
        prompts = load_prompts(args.prompts)
        prompts_path = args.prompts

    # ── choose operator parser ───────────────────────────────────────────
    OPERATOR_PARSER = parse_operator_count

    print(f"[*] Model:          {args.model}")
    print(f"[*] Prompts:        {len(prompts)} ({'smoke' if prompts_path is None else prompts_path})")
    print(f"[*] Layer mode:     {args.layer_mode}")
    print(f"[*] Max new tokens: {args.max_new_tokens}")
    print(f"[*] Device:         {args.device}")
    print(f"[*] Seed:           {args.seed}")
    print()

    # ── load model ───────────────────────────────────────────────────────
    print("[*] Loading model...")
    model, tokenizer = load_model(args.model, device=args.device)
    n_layers = model.config.num_hidden_layers
    hidden_dim = model.config.hidden_size
    print(f"[+] Loaded: {n_layers} layers, hidden_dim={hidden_dim}\n")

    # ── run prompts ──────────────────────────────────────────────────────
    rows = []
    for i, prompt in enumerate(prompts):
        tag = prompt[:60].replace("\n", " ")
        print(f"[{i + 1}/{len(prompts)}] {tag}...")

        generated_text, gen_activations, token_counts = generate_and_get_hidden_states(
            model, tokenizer, prompt, args.max_new_tokens, args.seed + i,
            device=args.device, layer_mode=args.layer_mode, n_layers=n_layers,
        )

        # Sanity print on first prompt so the user can eyeball shapes.
        if i == 0:
            print(f"    [shape] gen_hidden={gen_activations.shape}  "
                  f"prompt_len={token_counts['prompt_token_count']}  "
                  f"hidden_dim={hidden_dim}")

        d_act = participation_ratio(gen_activations)
        n_text, verb_detail = OPERATOR_PARSER(generated_text)

        rows.append({
            "prompt_id": i,
            "prompt": prompt[:200],
            "n_text": n_text,
            "d_act": round(d_act, 6),
            "token_count": token_counts["generated_token_count"],
            "prompt_token_count": token_counts["prompt_token_count"],
            "total_token_count": token_counts["total_token_count"],
            "verbs_found": ",".join(sorted(verb_detail.keys())),
        })
        print(f"    n_text={n_text}  d_act={d_act:.2f}  "
              f"tokens={token_counts['generated_token_count']}")

    # ── write CSV ────────────────────────────────────────────────────────
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\n[+] Wrote {len(df)} rows to {args.out}")

    # ── correlations ─────────────────────────────────────────────────────
    from scipy import stats

    v_n = df["n_text"].values.astype(float)
    v_d = df["d_act"].values.astype(float)
    v_t = df["token_count"].values.astype(float)

    if len(df) < 4:
        print("[!] < 4 rows — skipping correlation (need >= 4).")
        raw_pearson_r = raw_pearson_p = 0.0
        raw_spearman_r = raw_spearman_p = 0.0
        partial_pearson_r = partial_pearson_p = 0.0
        partial_spearman_r = partial_spearman_p = 0.0
    else:
        raw_pearson_r, raw_pearson_p = stats.pearsonr(v_n, v_d)
        raw_spearman_r, raw_spearman_p = stats.spearmanr(v_n, v_d)
        partial_pearson_r, partial_pearson_p = partial_corr(v_n, v_d, v_t, "pearson")
        partial_spearman_r, partial_spearman_p = partial_corr(v_n, v_d, v_t, "spearman")

    print(f"\n{'=' * 60}")
    print(f"  Raw Pearson:      r={raw_pearson_r:.4f}  p={raw_pearson_p:.4f}")
    print(f"  Raw Spearman:     r={raw_spearman_r:.4f}  p={raw_spearman_p:.4f}")
    print(f"  Partial Pearson:  r={partial_pearson_r:.4f}  p={partial_pearson_p:.4f}"
          f"  (control: token_count)")
    print(f"  Partial Spearman: r={partial_spearman_r:.4f}  p={partial_spearman_p:.4f}"
          f"  (control: token_count)")
    print(f"{'=' * 60}")

    # ── provenance sidecar ───────────────────────────────────────────────
    from provenance import build_provenance, write_sidecar

    prov = build_provenance(
        results_csv_path=args.out,
        df=df,
        prompts_path=prompts_path,
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
        layer_mode=args.layer_mode,
        operator_parser=OPERATOR_PARSER,
        stats_summary={
            "raw_pearson_r": raw_pearson_r,
            "raw_pearson_p": raw_pearson_p,
            "raw_spearman_r": raw_spearman_r,
            "raw_spearman_p": raw_spearman_p,
            "partial_pearson_r": partial_pearson_r,
            "partial_pearson_p": partial_pearson_p,
            "partial_spearman_r": partial_spearman_r,
            "partial_spearman_p": partial_spearman_p,
        },
        control_var="generated_token_count",
        seed=args.seed,
        extra={
            "notes": "primary control = generated token_count; "
                     "PR <= n_tokens ceiling acknowledged",
        },
    )
    sidecar_path, digest = write_sidecar(prov, results_csv_path=args.out)
    print(f"[provenance] sidecar: {sidecar_path}")
    print(f"[provenance] sha256:  {digest}")


if __name__ == "__main__":
    main()
