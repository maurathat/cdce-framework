#!/usr/bin/env python3
"""
cdce_routing_objects.py  —  v0.4 content-addressed routing objects.

Reference stub. Canonical-form + κ-addressing + verify-by-recompute are real.
LLM execution (run_trace) and the discovery/soft witnesses (witness_discovery,
witness_energy) are STUBS with fixed interfaces — the architecture test fills
these in; doing so must not change the canonical forms above.

Depends on v0.3: uor-addr (kappa.json_address). Same JSON memory-store style
as the harness.
"""

import argparse, json, hashlib, os, sys
from typing import Optional

# v0.3 dependency. Fallback keeps the stub runnable offline; the fallback's
# label is NOT interchangeable with the real κ-label and is marked as such.
try:
    from uor_addr import kappa
    def kappa_addr(b: bytes) -> str:
        return kappa.json_address(b)
except Exception:
    def kappa_addr(b: bytes) -> str:
        return "sha256-LOCALFALLBACK:" + hashlib.sha256(b).hexdigest()


# ---- canonical form: the single chokepoint all addressing goes through ----

def canonical_bytes(obj: dict, fields: list) -> bytes:
    """Canonical form = sorted-key JSON over EXACTLY `fields`, nothing else.
    Fields absent from `fields` (titles, prose, timestamps) are excluded from
    identity by construction. This is the v0.3 'hash the set, not the text' rule
    generalized to objects."""
    canon = {k: obj[k] for k in fields if k in obj}
    return json.dumps(canon, sort_keys=True, separators=(",", ":")).encode()


TASK_FIELDS     = ["family", "input_schema", "success_predicate", "scorer", "params"]
TRACE_FIELDS    = ["task_kappa", "model", "strategy_kappa", "input_kappa",
                   "output", "score", "success", "graded", "budget"]
DECISION_FIELDS = ["task_kappa", "model", "candidates", "selected_kappa",
                   "policy", "guard_result", "outcome_trace_kappa"]


def address(obj: dict, fields: list) -> str:
    return kappa_addr(canonical_bytes(obj, fields))


# ---- constructors: attach κ as identity; keep non-canonical metadata aside ----

def make_taskspec(family, input_schema, success_predicate, scorer,
                  params=None, **metadata) -> dict:
    body = {"family": family, "input_schema": input_schema,
            "success_predicate": success_predicate, "scorer": scorer,
            "params": params or {}}
    body["task_kappa"] = address(body, TASK_FIELDS)
    body["_metadata"] = metadata          # title/desc/author — NOT addressed
    return body


def make_trace(task_kappa, model, strategy_kappa, input_kappa,
               output, score, success, budget, graded=True) -> dict:
    body = {"task_kappa": task_kappa, "model": model,
            "strategy_kappa": strategy_kappa, "input_kappa": input_kappa,
            "output": output, "score": float(score), "success": bool(success),
            "graded": bool(graded), "budget": int(budget)}
    body["trace_kappa"] = address(body, TRACE_FIELDS)
    return body


def make_decision(task_kappa, model, candidates, selected_kappa, policy,
                  guard_result, outcome_trace_kappa) -> dict:
    assert selected_kappa in candidates, "selected_kappa must be a candidate"
    body = {"task_kappa": task_kappa, "model": model,
            "candidates": sorted(candidates), "selected_kappa": selected_kappa,
            "policy": policy, "guard_result": guard_result,
            "outcome_trace_kappa": outcome_trace_kappa}
    body["decision_kappa"] = address(body, DECISION_FIELDS)
    return body


# ---- verify-by-recompute: the entire security model (blob protocol move 4) ----

def verify(obj: dict, fields: list, kappa_key: str) -> bool:
    """Recompute the κ-label locally; assert it matches the claimed one.
    A mismatch => reject. The store can be adversarial; this still cannot be
    served the wrong object without detection."""
    claimed = obj.get(kappa_key)
    recomputed = address(obj, fields)
    return claimed == recomputed


def walk_lineage(decision: dict, store) -> list:
    """Walk decision_κ -> trace_κ -> strategy_κ -> (v0.3 strategy lineage),
    verifying each link by recompute. Returns the verified chain; raises on a
    broken link (no ancestor silently substituted)."""
    chain = []
    assert verify(decision, DECISION_FIELDS, "decision_kappa"), "decision κ mismatch"
    chain.append(("decision", decision["decision_kappa"]))

    trace = store.get(decision["outcome_trace_kappa"])
    assert trace and verify(trace, TRACE_FIELDS, "trace_kappa"), "trace κ mismatch"
    chain.append(("trace", trace["trace_kappa"]))

    # strategy_κ and its v0.3 lineage are verified by the existing v0.3 machinery;
    # here we just record the handoff point.
    chain.append(("strategy", trace["strategy_kappa"]))
    return chain


# ---- input addressing ----

# NOTE: Input identity is currently prompt-based (hash of the prompt string).
# The harness tasks have no structured input representation (e.g. coordinate
# lists for TSP, sequences for prediction) — only prompt text. This means any
# whitespace or wording change produces a different κ for the same logical
# instance. Revisit when tasks gain structured input fields; canonicalize those
# instead of the prompt string.
INPUT_FIELDS = ["task_kappa", "prompt", "params"]

def make_input_kappa(task_kappa: str, prompt: str, params: dict = None) -> tuple:
    """Canonicalize a concrete task input and return (input_dict, input_kappa).
    Canonical fields: task_kappa, prompt, params. Everything else excluded."""
    body = {"task_kappa": task_kappa, "prompt": prompt, "params": params or {}}
    k = address(body, INPUT_FIELDS)
    body["input_kappa"] = k
    return body, k


# ---- run_trace: wired to the harness ----

def run_trace(task: dict, model: str, strategy_kappa: str,
              input_kappa: str, budget: int, store) -> dict:
    """Execute `model` on the task instance using strategy_κ; return a Trace
    via make_trace(). Wired to harness llm_clients for LLM calls.
    Contract: returns a make_trace(...) dict. Does NOT alter TRACE_FIELDS."""
    from src.llm_clients import call_model
    from src.config import MODELS

    cfg = MODELS.get(model)
    if cfg is None:
        raise ValueError(f"Unknown model key '{model}'. Available: {list(MODELS.keys())}")

    input_obj = store.get(input_kappa)
    if input_obj is None:
        raise ValueError(f"input_kappa not found in store: {input_kappa}")
    task_prompt = input_obj["prompt"]

    strategy_text = None
    if strategy_kappa != "none":
        strat_obj = store.get(strategy_kappa)
        if strat_obj is not None:
            strategy_text = strat_obj.get("strategy_text", "")

    if strategy_text:
        prompt = (f"Here is a compressed strategy for a task:\n---\n"
                  f"{strategy_text}\n---\n"
                  f"Task: {task_prompt}\n"
                  f"Execute the strategy and return your complete answer.")
    else:
        prompt = task_prompt

    resp = call_model(
        provider=cfg["provider"],
        model=cfg["model"],
        prompt=prompt,
        max_tokens=budget,
        task_id=task.get("_metadata", {}).get("task_id", "routing_trace"),
    )
    output = resp.content

    # Score via the scorer registry. Scorable tasks return (score, success,
    # graded=True). Unscoreable tasks return (0.0, False, graded=False).
    # Consumers must filter on graded=True before computing score statistics.
    from src.scorers import score as score_fn
    task_id = task.get("_metadata", {}).get("task_id", "unknown")
    input_obj = store.get(input_kappa)
    score, success, graded = score_fn(task_id, input_obj, output)

    return make_trace(
        task_kappa=task["task_kappa"],
        model=model,
        strategy_kappa=strategy_kappa,
        input_kappa=input_kappa,
        output=output,
        score=score,
        success=success,
        budget=budget,
        graded=graded,
    )


def witness_discovery(task_kappa: str, all_candidates: list, k: int) -> list:
    """WITNESS 1 (foolable). Narrow `all_candidates` to a shortlist of size k by
    embedding/geometric similarity or a greedy step. NEVER binds. STUB returns
    the first k (identity) so the pipeline runs before the real witness exists."""
    return list(all_candidates)[:k]


def witness_energy(task_kappa: str, shortlist: list) -> list:
    """WITNESS 2 (soft). Rank shortlist by EBM energy E(strategy|task), low=better.
    NEVER binds. STUB returns input order. Wired to the EBM-over-corpus work."""
    return list(shortlist)


# ---- minimal JSON store (same spirit as the harness memory store) ----

class JsonStore:
    def __init__(self, path): self.path = path; os.makedirs(path, exist_ok=True)
    def _f(self, k): return os.path.join(self.path, k.replace(":", "_").replace("/", "_") + ".json")
    def put(self, k, obj):
        with open(self._f(k), "w") as f: json.dump(obj, f, sort_keys=True)
    def get(self, k):
        try:
            with open(self._f(k)) as f: return json.load(f)
        except FileNotFoundError:
            return None


def _demo():
    """Runnable demo of the parts that are REAL: canonical form, addressing,
    verify-by-recompute, and tamper detection. No LLM/EBM needed."""
    store = JsonStore("./routing_store")

    task = make_taskspec(
        family="optimization",
        input_schema="TSP:n_cities=10,coords=int[0,100]^2",
        success_predicate="tour_length==optimal",
        scorer="tsp_optimal_v1", params={"n": 10},
        title="10-city TSP (human-readable, not addressed)")
    store.put(task["task_kappa"], task)
    print("task_κ     =", task["task_kappa"])

    trace = make_trace(task["task_kappa"], "claude-sonnet-4",
                       strategy_kappa="sha256:STRATEGYPLACEHOLDER",
                       input_kappa="sha256:INPUTPLACEHOLDER",
                       output="[0,3,5,...]", score=1.0, success=True,
                       budget=125, graded=True)
    store.put(trace["trace_kappa"], trace)
    print("trace_κ    =", trace["trace_kappa"])

    decision = make_decision(
        task["task_kappa"], "claude-sonnet-4",
        candidates=["sha256:STRATEGYPLACEHOLDER", "sha256:OTHER"],
        selected_kappa="sha256:STRATEGYPLACEHOLDER",
        policy="xbosm_adaptive_weighted_k8_guarded_v1",
        guard_result="pass",
        outcome_trace_kappa=trace["trace_kappa"])
    store.put(decision["decision_kappa"], decision)
    print("decision_κ =", decision["decision_kappa"])

    print("\nverify task     :", verify(task, TASK_FIELDS, "task_kappa"))
    print("verify trace    :", verify(trace, TRACE_FIELDS, "trace_kappa"))
    print("verify decision :", verify(decision, DECISION_FIELDS, "decision_kappa"))

    print("\nlineage (verified):")
    for kind, k in walk_lineage(decision, store):
        print(f"  {kind:9s} {k}")

    # tamper: flip the scorer AFTER addressing -> identity must break
    task["scorer"] = "tsp_optimal_v2"
    print("\nafter tampering with scorer, verify task:",
          verify(task, TASK_FIELDS, "task_kappa"), "(False = tamper detected)")


def _run(args):
    """Run one trace through the harness and emit a verified trace_κ."""
    store = JsonStore("./routing_store")

    # resolve or create task
    task_obj = store.get(args.task)
    if task_obj is None:
        print(f"[!] task_kappa '{args.task}' not in store.")
        print("    Use 'demo' first to populate, or store a TaskSpec manually.")
        sys.exit(1)

    # resolve or create input
    if args.input == "from-task":
        # convenience: build input_kappa from the task's prompt
        prompt = task_obj.get("_metadata", {}).get("prompt",
                    task_obj.get("input_schema", ""))
        input_obj, ik = make_input_kappa(task_obj["task_kappa"], prompt)
        store.put(ik, input_obj)
        print(f"input_κ    = {ik}  (derived from task prompt)")
    else:
        ik = args.input
        if store.get(ik) is None:
            print(f"[!] input_kappa '{ik}' not in store.")
            sys.exit(1)

    print(f"task_κ     = {args.task}")
    print(f"model      = {args.model}")
    print(f"strategy_κ = {args.strategy}")
    print(f"budget     = {args.budget}")
    print()

    trace = run_trace(task_obj, args.model, args.strategy, ik, args.budget, store)
    store.put(trace["trace_kappa"], trace)

    ok = verify(trace, TRACE_FIELDS, "trace_kappa")
    print(f"trace_κ    = {trace['trace_kappa']}")
    print(f"verify     = {ok}")
    if not trace['graded']:
        print(f"score      = {trace['score']}  (ungraded — no objective scorer)")
        print(f"success    = {trace['success']}  (ungraded)")
        print(f"graded     = False")
    else:
        print(f"score      = {trace['score']}")
        print(f"success    = {trace['success']}")
        print(f"graded     = True")
    if not ok:
        print("[FAIL] trace did not verify — this is a bug.")
        sys.exit(1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="v0.4 content-addressed routing objects")
    sub = ap.add_subparsers(dest="mode")

    sub.add_parser("demo", help="run the real (non-LLM) parts")

    rp = sub.add_parser("run", help="execute one trace via the harness")
    rp.add_argument("--task", required=True, help="task_kappa from the store")
    rp.add_argument("--model", default="claude", help="config key: claude|haiku|gpt4o")
    rp.add_argument("--strategy", default="none", help="strategy_kappa or 'none'")
    rp.add_argument("--input", default="from-task",
                    help="input_kappa or 'from-task' to derive from task prompt")
    rp.add_argument("--budget", type=int, default=125)

    args = ap.parse_args()
    if args.mode == "demo":
        _demo()
    elif args.mode == "run":
        _run(args)
    else:
        ap.print_help()
