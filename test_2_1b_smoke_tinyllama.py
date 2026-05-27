#!/usr/bin/env python3
"""
test_2_1b_smoke_tinyllama.py
============================
SMOKE TEST for the Test 2.1b activation-grounding pipeline.

Purpose: prove the whole pipeline runs end-to-end on a small, UNGATED model
(TinyLlama-1.1B-Chat) BEFORE introducing any Gemma/gating/thinking-mode complexity.
A failure here is unambiguously a pipeline bug. Once this runs clean, the real run
on gemma-2-2b-it differs only by the --model string (and the chat template is
already handled the same way).

WHAT THIS IS / IS NOT
  - This is a PIPELINE smoke test. The operator parser here is a FALLBACK
    (regex-counting arithmetic/logic words). It is NOT your real operator metric.
    Any correlation printed is therefore MEANINGLESS as science — it only proves
    the numbers flow through. Swap OPERATOR_PARSER for the real one for a real run.
  - Effective-dim proxy = participation ratio of the generated-token hidden-state
    trajectory. Headline statistic = PARTIAL correlation controlling for token count
    (raw correlation is expected to be confounded by length; that's the point).

KEY CORRECTNESS POINTS (the places this kind of script silently breaks):
  1. Hidden states come from a CLEAN re-forward pass of the full sequence
     (output_hidden_states=True, use_cache=False), NOT from generation-time cache.
  2. Only GENERATED tokens are analyzed: we slice at prompt_len computed AFTER
     applying the chat template (template adds tokens; slicing at raw-prompt length
     would leak prompt/template tokens into the activation matrix and inflate PR).
  3. participation_ratio divides by matrix.shape[0] (NOT an undefined `num_tokens`),
     and PR is scale-invariant so the divisor can't corrupt the result anyway.
  4. partial correlation is true residual-on-residual (residualize x and y against
     the control separately, then correlate residuals).
  5. PR <= n_generated_tokens is a hard ceiling -> raw r will look good for a
     meaningless reason -> the length control is load-bearing. We also print the
     token_count distribution so you can confirm there's length VARIANCE to work with.

Run (smoke test, no args needed):
    pip install torch transformers accelerate numpy pandas scipy scikit-learn
    python3 test_2_1b_smoke_tinyllama.py

Optional:
    python3 test_2_1b_smoke_tinyllama.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --prompts prompts_2_1b.txt --out results_smoke.csv --max-new-tokens 200 \
        --layer-mode last
"""

from __future__ import annotations

import argparse
import re
import sys
from typing import Callable, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ----------------------------------------------------------------------
# Fallback operator parser (PLACEHOLDER — swap for your real metric).
# Counts a few arithmetic/logic operation words/symbols. Meaningless as science;
# only here so the activation pipeline has an n_text column to flow.
# ----------------------------------------------------------------------
_OP_PATTERNS = [
    r"\badd(s|ed|ing)?\b", r"\bsubtract", r"\bmultiply", r"\bdivide",
    r"\bsum\b", r"\bproduct\b", r"\bcompare", r"\bif\b", r"\bthen\b",
    r"\botherwise\b", r"\bcompute", r"\bcount", r"[\+\-\*/=]",
]


def parse_operator_count_fallback(text: str) -> Tuple[int, dict]:
    counts = {}
    total = 0
    for pat in _OP_PATTERNS:
        n = len(re.findall(pat, text, flags=re.IGNORECASE))
        if n:
            counts[pat] = n
        total += n
    return total, counts


OPERATOR_PARSER: Callable[[str], Tuple[int, dict]] = parse_operator_count_fallback


# ----------------------------------------------------------------------
# Default smoke-test prompts (a spread of expected operator counts AND lengths,
# so the length control has variance to work with). Replace with your real set.
# ----------------------------------------------------------------------
DEFAULT_PROMPTS = [
    "Say hello.",
    "What is 2 plus 2?",
    "Add 14 and 27, then subtract 5. Show each step.",
    "Compare 3/4 and 5/8 and explain which is larger and why.",
    "A train travels 60 km in 1.5 hours. Compute its average speed, then say how far it goes in 4 hours.",
    "List three fruits.",
    "If x is 7 and y is 3, compute x times y, then add 10, then divide by 2.",
    "Explain in one sentence what a prime number is.",
    "Sum the numbers 1 through 10 step by step.",
    "Describe the color blue without using the word blue.",
]


def load_prompts(path: str | None) -> List[str]:
    if not path:
        return DEFAULT_PROMPTS
    with open(path, "r", encoding="utf-8") as f:
        prompts = [ln.strip() for ln in f if ln.strip()]
    return prompts or DEFAULT_PROMPTS


# ----------------------------------------------------------------------
# Prompt formatting — chat template if the tokenizer has one (TinyLlama-Chat
# and gemma-2-2b-it both do), else raw. Returns (input_ids, prompt_len).
# prompt_len is the post-template token count — the correct slice boundary.
# ----------------------------------------------------------------------
def format_prompt(tokenizer, prompt: str, device) -> Tuple[torch.Tensor, int]:
    if tokenizer.chat_template:
        chat = [{"role": "user", "content": prompt}]
        result = tokenizer.apply_chat_template(
            chat, return_tensors="pt", add_generation_prompt=True
        )
        input_ids = (result["input_ids"] if hasattr(result, "keys") else result).to(device)
    else:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    return input_ids, int(input_ids.shape[1])


# ----------------------------------------------------------------------
# Hidden-state selection: drop embedding layer (index 0), pick last/midmean/allmean.
# Returns [seq_len, hidden_dim] for batch index 0.
# ----------------------------------------------------------------------
def select_hidden_state_tensor(hidden_states, layer_mode: str = "last",
                               mid_layer_count: int = 4) -> torch.Tensor:
    if not hidden_states:
        raise ValueError("No hidden states. Did you set output_hidden_states=True?")
    layer_tensors = list(hidden_states[1:]) if len(hidden_states) > 1 else list(hidden_states)
    if layer_mode == "last":
        selected = layer_tensors[-1]
    elif layer_mode == "allmean":
        selected = torch.stack(layer_tensors, dim=0).mean(dim=0)
    elif layer_mode == "midmean":
        n = len(layer_tensors)
        center = n // 2
        half = max(mid_layer_count // 2, 1)
        start = max(center - half, 0)
        end = min(start + mid_layer_count, n)
        selected = torch.stack(layer_tensors[start:end], dim=0).mean(dim=0)
    else:
        raise ValueError(f"Unknown layer_mode: {layer_mode}")
    return selected[0].detach().to(torch.float32).cpu()


def participation_ratio(matrix: np.ndarray, eps: float = 1e-12) -> float:
    """
    Effective dimension of an activation trajectory.
    matrix: [num_tokens, hidden_dim]
      1. center across tokens
      2. covariance eigenvalues via singular values
      3. PR = (sum lambda)^2 / sum(lambda^2)
    Scale-invariant: any constant factor on lambda cancels.
    """
    x = matrix.astype(np.float64)
    n_tokens = x.shape[0]              # <-- shape[0], not an external `num_tokens`
    if n_tokens < 2:
        return float("nan")
    x = x - x.mean(axis=0, keepdims=True)
    sv = np.linalg.svd(x, full_matrices=False, compute_uv=False)
    eig = (sv ** 2) / max(n_tokens - 1, 1)
    eig = eig[eig > eps]
    if eig.size == 0:
        return float("nan")
    num = float(np.sum(eig) ** 2)
    den = float(np.sum(eig ** 2) + eps)
    return num / den


def partial_corr_control_length(x, y, control, rank: bool = False) -> Tuple[float, float]:
    """True residual-on-residual partial correlation of x,y controlling for `control`."""
    x = np.asarray(list(x), float); y = np.asarray(list(y), float); c = np.asarray(list(control), float)
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(c)
    x, y, c = x[m], y[m], c[m]
    if len(x) < 4:
        return float("nan"), float("nan")
    if rank:
        x, y, c = stats.rankdata(x), stats.rankdata(y), stats.rankdata(c)
    cm = c.reshape(-1, 1)
    xr = x - LinearRegression().fit(cm, x).predict(cm)
    yr = y - LinearRegression().fit(cm, y).predict(cm)
    if np.std(xr) == 0 or np.std(yr) == 0:
        return float("nan"), float("nan")
    r, p = stats.pearsonr(xr, yr)
    return float(r), float(p)


def main():
    ap = argparse.ArgumentParser(description="Test 2.1b smoke test on TinyLlama")
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--prompts", default=None, help="one prompt per line; omit for built-in set")
    ap.add_argument("--out", default="results_smoke.csv")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--layer-mode", default="all", choices=["last", "midmean", "allmean", "all"],
                    help="'all' runs last+midmean+allmean and reports each")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[setup] device={device}  model={args.model}")
    if "fallback" in OPERATOR_PARSER.__name__:
        print("[setup] *** FALLBACK parser active — results are a pipeline test, NOT science ***")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=(torch.float32 if device == "cpu" else torch.bfloat16),
    ).to(device)
    model.eval()

    prompts = load_prompts(args.prompts)
    print(f"[setup] {len(prompts)} prompts")

    layer_modes = ["last", "midmean", "allmean"] if args.layer_mode == "all" else [args.layer_mode]

    rows = []
    for i, prompt in enumerate(prompts):
        input_ids, prompt_len = format_prompt(tokenizer, prompt, device)

        with torch.no_grad():
            gen = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,                       # greedy → deterministic smoke test
                pad_token_id=tokenizer.eos_token_id,
            )
        full_ids = gen  # [1, prompt_len + generated_len]
        gen_text = tokenizer.decode(full_ids[0, prompt_len:], skip_special_tokens=True)
        n_text, _ = OPERATOR_PARSER(gen_text)
        gen_token_count = int(full_ids.shape[1] - prompt_len)

        # Clean re-forward of the full sequence to get hidden states (NOT gen cache).
        with torch.no_grad():
            out = model(
                input_ids=full_ids,
                output_hidden_states=True,
                use_cache=False,
                return_dict=True,
            )

        row = {"prompt_id": i, "n_text": n_text,
               "token_count": gen_token_count, "prompt_token_count": prompt_len}

        for lm in layer_modes:
            hidden = select_hidden_state_tensor(out.hidden_states, layer_mode=lm)
            gen_hidden = hidden[prompt_len:, :]        # generated tokens only
            # SANITY PRINT on the very first prompt/first mode: see the shape yourself.
            if i == 0 and lm == layer_modes[0]:
                print(f"[shape] full_seq={tuple(hidden.shape)}  "
                      f"prompt_len={prompt_len}  gen_hidden={tuple(gen_hidden.shape)}  "
                      f"hidden_dim={hidden.shape[1]}")
                if gen_hidden.shape[0] < 2:
                    print("[warn] <2 generated tokens — PR undefined; raise --max-new-tokens "
                          "or check the prompt produced output.")
            pr = participation_ratio(gen_hidden.numpy())
            row[f"d_act_{lm}"] = pr

        rows.append(row)
        print(f"  [{i+1}/{len(prompts)}] gen_tokens={gen_token_count:4d}  n_text={n_text:3d}  "
              + "  ".join(f"PR_{lm}={row[f'd_act_{lm}']:.2f}" for lm in layer_modes))

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"\n[out] wrote {args.out}")

    # Length-variance check — the control needs spread to be meaningful.
    tc = df["token_count"]
    print(f"[len] token_count: min={tc.min()} max={tc.max()} "
          f"mean={tc.mean():.0f} std={tc.std():.0f}")
    if tc.std() < 1e-6 or tc.nunique() < 3:
        print("[warn] token_count has little/no variance — partial correlation will be "
              "uninformative. Use prompts with more varied response lengths.")

    # Correlations per layer mode. Headline = PARTIAL (controls length).
    print("\n=== correlations (FALLBACK parser — pipeline check only) ===")
    for lm in layer_modes:
        d = df[f"d_act_{lm}"]
        raw_r, raw_p = (stats.pearsonr(df["n_text"], d)
                        if df["n_text"].std() and d.std() else (float("nan"), float("nan")))
        sr, sp = (stats.spearmanr(df["n_text"], d)
                  if df["n_text"].std() and d.std() else (float("nan"), float("nan")))
        pr_r, pr_p = partial_corr_control_length(df["n_text"], d, df["token_count"], rank=False)
        ps_r, ps_p = partial_corr_control_length(df["n_text"], d, df["token_count"], rank=True)
        print(f"\n[layer={lm}]")
        print(f"  raw     Pearson r={raw_r:+.3f} (p={raw_p:.3f})   Spearman r={sr:+.3f} (p={sp:.3f})")
        print(f"  partial Pearson r={pr_r:+.3f} (p={pr_p:.3f})   Spearman r={ps_r:+.3f} (p={ps_p:.3f})  <-- headline")

    print("\n[done] Pipeline ran end-to-end. If shapes and length-variance look right above,")
    print("       the plumbing is proven. Swap in the real parser and the gemma-2-2b-it model")
    print("       string for the real run (chat template is already handled).")


if __name__ == "__main__":
    main()
