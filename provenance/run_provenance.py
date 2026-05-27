"""
run_provenance.py
=================
Provenance sidecar for Test 2.1b (and reusable for the 6.3/6.4 scripts).

WHAT THIS DOES, AND WHAT IT DELIBERATELY DOES NOT DO
----------------------------------------------------
This writes a small *canonical* JSON sidecar next to a results CSV recording
EXACTLY what the result was computed against: the prompt set, the operator-parser
identity, the model + generation params, the layer mode, the library versions,
and a per-row record of the inputs that fed each correlation.

It is provenance / identity infrastructure ONLY (Wall 1). It does not compute,
alter, or "improve" any statistic. The science is the participation-ratio +
partial-correlation in your main script; this just makes that result reproducible
and verifiable after the fact. Content-addressing (UOR-ADDR / plain sha256) is
OPTIONAL and applied to the canonical bytes AFTER the science is done — the hash
certifies "same inputs/outputs", not "the operators are in the activations".

The sidecar is written in a canonical form (sorted keys, fixed separators, UTF-8)
so that two identical runs produce byte-identical sidecars -> a stable digest.
That is the only reason canonicalization matters here.

USAGE (add ~4 lines to test_2_1b_activation_grounding.py)
---------------------------------------------------------
    from run_provenance import build_provenance, write_sidecar

    # ... after you have `df` (with n_text, d_act, token_count, ...) and have written results_2_1b.csv ...

    prov = build_provenance(
        results_csv_path=args.out,
        df=df,                                  # the results DataFrame
        prompts_path=args.prompts,              # path to the prompt file (or None for smoke test)
        model_name=args.model,
        max_new_tokens=args.max_new_tokens,
        layer_mode=layer_mode,                  # "last" / "midmean" / "allmean"
        operator_parser=OPERATOR_PARSER,        # the actual callable in use (fallback vs real)
        stats_summary={                         # the headline numbers your script already computes
            "raw_pearson_r": raw_pearson_r,     "raw_pearson_p": raw_pearson_p,
            "raw_spearman_r": raw_spearman_r,   "raw_spearman_p": raw_spearman_p,
            "partial_pearson_r": partial_pearson_r, "partial_pearson_p": partial_pearson_p,
            "partial_spearman_r": partial_spearman_r, "partial_spearman_p": partial_spearman_p,
        },
        control_var="generated_token_count",
        extra={"notes": "primary control = generated token_count; PR <= n_tokens ceiling acknowledged"},
    )
    sidecar_path, digest = write_sidecar(prov, results_csv_path=args.out)
    print(f"[provenance] sidecar: {sidecar_path}")
    print(f"[provenance] sha256:  {digest}")
"""

from __future__ import annotations

import hashlib
import inspect
import json
import os
import platform
import sys
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Iterable, Optional


# ----------------------------------------------------------------------
# Parser identity — answers "which operator parser produced n_text?"
# This is the single most important provenance field for this test,
# because a fallback-parser CSV must never be mistaken for a real-parser one.
# ----------------------------------------------------------------------
def _parser_identity(parser: Optional[Callable]) -> Dict[str, Any]:
    if parser is None:
        return {"name": None, "module": None, "qualname": None, "source_sha256": None,
                "is_fallback": None, "warning": "no parser passed"}
    name = getattr(parser, "__name__", repr(parser))
    module = getattr(parser, "__module__", None)
    qualname = getattr(parser, "__qualname__", name)

    # Hash the parser's source so a changed parser yields a changed identity.
    source_sha = None
    try:
        src = inspect.getsource(parser)
        source_sha = hashlib.sha256(src.encode("utf-8")).hexdigest()
    except (OSError, TypeError):
        source_sha = None  # builtins / C / lambdas without retrievable source

    # Heuristic fallback detection: the script's fallback is named with "fallback".
    is_fallback = "fallback" in name.lower() or "fallback" in str(qualname).lower()

    return {
        "name": name,
        "module": module,
        "qualname": qualname,
        "source_sha256": source_sha,
        "is_fallback": is_fallback,
    }


def _file_digest(path: Optional[str]) -> Optional[Dict[str, Any]]:
    """sha256 + size of a file (e.g. the prompt set), or None if absent."""
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
            size += len(chunk)
    return {"path": os.path.abspath(path), "sha256": h.hexdigest(), "size_bytes": size}


def _lib_versions() -> Dict[str, Optional[str]]:
    versions: Dict[str, Optional[str]] = {}
    for mod in ("torch", "transformers", "numpy", "pandas", "scipy", "sklearn", "accelerate"):
        try:
            m = __import__(mod)
            versions[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            versions[mod] = None
    return versions


def _per_row_inputs(df) -> Dict[str, Any]:
    """
    Record the per-row inputs that fed the correlation, so the headline stats
    can be recomputed from the sidecar alone. Pulls only the columns the science
    used; tolerant of missing optional columns.
    """
    if df is None:
        return {"n_rows": 0, "rows": []}
    wanted = ["prompt_id", "n_text", "d_act", "token_count",
              "prompt_token_count", "total_token_count"]
    present = [c for c in wanted if c in getattr(df, "columns", [])]
    rows = []
    try:
        for _, r in df.iterrows():
            rows.append({c: (None if _is_nan(r[c]) else _coerce(r[c])) for c in present})
    except Exception:
        rows = []
    return {"n_rows": len(rows), "columns": present, "rows": rows}


def _is_nan(v: Any) -> bool:
    try:
        return v != v  # NaN is the only value not equal to itself
    except Exception:
        return False


def _coerce(v: Any) -> Any:
    """Make values JSON-clean and deterministic (ints stay ints, floats rounded sanely)."""
    try:
        import numpy as np
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return round(float(v), 10)
    except Exception:
        pass
    if isinstance(v, float):
        return round(v, 10)
    return v


def build_provenance(
    *,
    results_csv_path: str,
    df=None,
    prompts_path: Optional[str] = None,
    model_name: Optional[str] = None,
    max_new_tokens: Optional[int] = None,
    layer_mode: Optional[str] = None,
    operator_parser: Optional[Callable] = None,
    stats_summary: Optional[Dict[str, Any]] = None,
    control_var: str = "generated_token_count",
    seed: Optional[int] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Assemble the provenance record. Pure data-gathering; no side effects.
    """
    prov = {
        "schema": "cdce.test_2_1b.provenance/v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": "Test 2.1b — activation grounding (operator text-count vs activation effective-dim)",
        "scope_note": (
            "Sourcing-half probe only. Does NOT validate the operator metric against human "
            "judgment (validity half, still requires annotation). Does NOT address any JEPA "
            "question (Gemma is an LLM). Provenance only; does not affect the computed statistic."
        ),
        "model": {
            "name": model_name,
            "max_new_tokens": max_new_tokens,
            "layer_mode": layer_mode,
        },
        "operator_parser": _parser_identity(operator_parser),
        "prompt_set": _file_digest(prompts_path),
        "control_variable": control_var,
        "seed": seed,
        "results_csv": _file_digest(results_csv_path),  # hash of the CSV the science wrote
        "stats_summary": stats_summary or {},
        "per_row_inputs": _per_row_inputs(df),
        "environment": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "libraries": _lib_versions(),
            "argv": sys.argv,
        },
    }
    if extra:
        prov["extra"] = extra

    # Loud guard: a fallback-parser run must be unmistakable in the record.
    if prov["operator_parser"].get("is_fallback"):
        prov["WARNING_FALLBACK_PARSER"] = (
            "n_text was produced by the FALLBACK parser, not the real operator parser. "
            "These results are a pipeline smoke test, NOT a real Test 2.1b measurement."
        )
    return prov


def canonical_bytes(prov: Dict[str, Any]) -> bytes:
    """
    Canonical serialization: sorted keys, compact deterministic separators, UTF-8.
    Two identical runs -> byte-identical output -> stable digest. This is the only
    reason canonicalization is used here (so the sidecar is content-addressable later).
    """
    return json.dumps(prov, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def write_sidecar(prov: Dict[str, Any], results_csv_path: str) -> tuple[str, str]:
    """
    Write the sidecar next to the results CSV. Returns (sidecar_path, sha256_digest).
    The digest is over the canonical bytes WITHOUT the self-referential digest field,
    so it is stable and verifiable.
    """
    digest = hashlib.sha256(canonical_bytes(prov)).hexdigest()

    # Pretty (human-readable) copy carries the digest for convenience; the digest itself
    # is computed over the canonical form excluding this field, so it stays verifiable.
    out = dict(prov)
    out["self_sha256"] = digest

    base = os.path.splitext(results_csv_path)[0]
    sidecar_path = base + ".provenance.json"
    with open(sidecar_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False, sort_keys=True)
    return sidecar_path, digest


def verify_sidecar(sidecar_path: str) -> bool:
    """
    Recompute the digest from a sidecar file and check it matches the stored self_sha256.
    Lets a future reader confirm the provenance record is intact.
    """
    with open(sidecar_path, "r", encoding="utf-8") as f:
        out = json.load(f)
    stored = out.pop("self_sha256", None)
    recomputed = hashlib.sha256(canonical_bytes(out)).hexdigest()
    return stored == recomputed


if __name__ == "__main__":
    # Tiny self-test on synthetic data: proves write + verify roundtrip, no model needed.
    import pandas as pd

    def parse_operator_count_fallback(_text):  # mimics the script's fallback name
        return 0, {}

    df = pd.DataFrame({
        "prompt_id": [0, 1, 2],
        "n_text": [3, 5, 2],
        "d_act": [4.1, 6.7, 3.0],
        "token_count": [120, 240, 80],
    })
    prov = build_provenance(
        results_csv_path="results_2_1b.csv",
        df=df,
        prompts_path=None,
        model_name="google/gemma-2-2b",
        max_new_tokens=256,
        layer_mode="last",
        operator_parser=parse_operator_count_fallback,
        stats_summary={"partial_pearson_r": 0.12, "partial_pearson_p": 0.7},
        seed=42,
    )
    path, digest = write_sidecar(prov, "results_2_1b.csv")
    print("wrote:", path)
    print("digest:", digest)
    print("fallback flagged:", "WARNING_FALLBACK_PARSER" in prov)
    print("verify:", verify_sidecar(path))
