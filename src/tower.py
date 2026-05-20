"""
CDCE Compression Harness — R->C->H->O Tower & E8 Root Comparison

Maps compressed operator geometries to the normed division algebra tower
(Cayley-Dickson construction) and compares against E8 root vectors from
the atlas-embeddings crate.

Tower levels (by operator count):
  O  Octonions   dim=8  (>=7 distinct operators)  non-associative
  H  Quaternions dim=4  (4-6 operators)            non-commutative
  C  Complex     dim=2  (2-3 operators)            commutative
  R  Reals       dim=1  (0-1 operators)            trivially compressed

The key CDCE prediction: under compression pressure, strategies descend
the tower O->H->C->R, losing algebraic structure in the same order that
the Cayley-Dickson construction gains it.  Octonion-level non-associativity
collapses first, then quaternion-level non-commutativity, until only a
single real-valued operator remains.

E8 comparison: each compressed strategy's verb set is embedded into R^8
using a fixed vocabulary-to-axis mapping, then compared against the 240
E8 root vectors via inner product.  Strategies that land near E8 roots
occupy algebraically distinguished positions in operator space.
"""
import json
import os
import math
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────
# Tower classification
# ─────────────────────────────────────────────────────────────────

TOWER_LEVELS = {
    'O': {'name': 'Octonions',   'dim': 8, 'min_ops': 7, 'max_ops': 999,
           'property': 'non-associative'},
    'H': {'name': 'Quaternions', 'dim': 4, 'min_ops': 4, 'max_ops': 6,
           'property': 'non-commutative'},
    'C': {'name': 'Complex',     'dim': 2, 'min_ops': 2, 'max_ops': 3,
           'property': 'commutative'},
    'R': {'name': 'Reals',       'dim': 1, 'min_ops': 0, 'max_ops': 1,
           'property': 'trivial'},
}


def classify_tower_level(operator_count: int) -> str:
    """Map an operator count to its tower level (R/C/H/O)."""
    if operator_count >= 7:
        return 'O'
    elif operator_count >= 4:
        return 'H'
    elif operator_count >= 2:
        return 'C'
    else:
        return 'R'


def tower_descent_signature(levels: list[str]) -> str:
    """Characterize the descent pattern through the tower.

    Returns a string like 'O->H->C->R' or 'H->H->C' showing the
    trajectory as budget decreases.
    """
    if not levels:
        return ''
    compressed = [levels[0]]
    for lv in levels[1:]:
        if lv != compressed[-1]:
            compressed.append(lv)
    return '->'.join(compressed)


# ─────────────────────────────────────────────────────────────────
# Verb-to-R8 embedding for E8 comparison
# ─────────────────────────────────────────────────────────────────

# Map the 9 verb categories from metrics.py onto 8 E8 axes.
# Categories: comparison, decomposition, composition, search,
#             selection, transformation, computation, logical, pattern
# We merge logical+pattern into axis 8 (the half-integer axis of E8).

VERB_CATEGORIES = {
    # Axis 0: Comparison / evaluation
    0: {"compare", "evaluate", "assess", "check", "verify", "test",
        "measure", "rank", "score", "weigh"},
    # Axis 1: Decomposition
    1: {"break", "split", "separate", "decompose", "divide", "partition",
        "isolate", "extract"},
    # Axis 2: Composition
    2: {"combine", "merge", "join", "aggregate", "sum", "total",
        "accumulate", "integrate", "unify"},
    # Axis 3: Search
    3: {"search", "find", "look", "scan", "explore", "try", "iterate",
        "enumerate", "traverse"},
    # Axis 4: Selection
    4: {"choose", "select", "pick", "decide", "assign", "allocate",
        "place", "swap", "move"},
    # Axis 5: Transformation
    5: {"convert", "transform", "translate", "map", "encode", "decode",
        "reduce", "simplify", "compress", "abstract"},
    # Axis 6: Computation
    6: {"calculate", "compute", "add", "subtract", "multiply", "divide",
        "average", "minimize", "maximize", "optimize"},
    # Axis 7: Logical + Pattern (merged)
    7: {"if", "then", "else", "because", "therefore", "since", "implies",
        "assume", "given", "conclude", "pattern", "rule", "formula",
        "sequence", "repeat", "recurse", "generalize", "observe",
        "notice", "identify"},
}

# Reverse lookup: verb -> axis
_VERB_TO_AXIS = {}
for axis, verbs in VERB_CATEGORIES.items():
    for v in verbs:
        _VERB_TO_AXIS[v] = axis


def embed_verbs_r8(unique_verbs: list[str]) -> np.ndarray:
    """Embed a verb set into R^8.

    Each axis gets +1 for each verb present in that category.
    The resulting vector is then normalized to unit length (lies on S^7).
    If no verbs, returns the zero vector.
    """
    vec = np.zeros(8, dtype=np.float64)
    for verb in unique_verbs:
        axis = _VERB_TO_AXIS.get(verb)
        if axis is not None:
            vec[axis] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


# ─────────────────────────────────────────────────────────────────
# E8 root system
# ─────────────────────────────────────────────────────────────────

_E8_ROOTS = None

def load_e8_roots(path: str = "data/e8_roots.json") -> np.ndarray:
    """Load the 240 E8 root vectors from the atlas-embeddings export."""
    global _E8_ROOTS
    if _E8_ROOTS is not None:
        return _E8_ROOTS
    root_path = Path(path)
    if not root_path.exists():
        # Try relative to this file's parent
        root_path = Path(__file__).parent.parent / path
    if not root_path.exists():
        raise FileNotFoundError(
            f"E8 roots not found at {path}. "
            f"Run: cd atlas-embeddings && cargo run --release --example export_roots_json > ../cdce-harness/data/e8_roots.json"
        )
    with open(root_path) as f:
        roots = json.load(f)
    _E8_ROOTS = np.array(roots, dtype=np.float64)
    # Normalize roots to unit vectors for cosine comparison
    norms = np.linalg.norm(_E8_ROOTS, axis=1, keepdims=True)
    norms[norms == 0] = 1
    _E8_ROOTS_UNIT = _E8_ROOTS / norms
    _E8_ROOTS = _E8_ROOTS_UNIT
    return _E8_ROOTS


@dataclass
class E8Comparison:
    """Result of comparing a compressed geometry against E8 roots."""
    verb_vector: list[float]          # R^8 embedding of verb set
    tower_level: str                  # R/C/H/O
    nearest_root_idx: int             # Index of closest E8 root
    nearest_root: list[float]         # The root vector itself
    cosine_similarity: float          # cos(angle) to nearest root
    inner_product: float              # Raw inner product
    root_type: str                    # 'integer' or 'half-integer'
    top_5_roots: list[dict] = field(default_factory=list)


def compare_to_e8(unique_verbs: list[str], operator_count: int = 0) -> Optional[E8Comparison]:
    """Compare a compressed geometry's verb set against E8 root vectors.

    Returns None if the verb set is empty (no embedding possible).
    """
    if not unique_verbs:
        return None

    try:
        roots = load_e8_roots()
    except FileNotFoundError:
        return None

    vec = embed_verbs_r8(unique_verbs)
    if np.linalg.norm(vec) == 0:
        return None

    # Cosine similarity with all 240 roots
    sims = roots @ vec  # (240,) since both are unit vectors
    top_indices = np.argsort(-sims)[:5]
    best_idx = top_indices[0]
    best_sim = float(sims[best_idx])

    # Classify root type: integer roots (indices 0-111) vs half-integer (112-239)
    root_type = 'integer' if best_idx < 112 else 'half-integer'

    # Raw inner product (with non-normalized root)
    raw_roots = json.load(open(Path(__file__).parent.parent / "data" / "e8_roots.json"))
    raw_root = np.array(raw_roots[best_idx])
    inner = float(vec @ raw_root)

    top_5 = []
    for idx in top_indices:
        top_5.append({
            'root_idx': int(idx),
            'cosine': float(sims[idx]),
            'type': 'integer' if idx < 112 else 'half-integer',
            'root': [float(x) for x in raw_roots[idx]],
        })

    return E8Comparison(
        verb_vector=[float(x) for x in embed_verbs_r8(unique_verbs) * np.linalg.norm(embed_verbs_r8(unique_verbs))],
        tower_level=classify_tower_level(operator_count or len(unique_verbs)),
        nearest_root_idx=int(best_idx),
        nearest_root=[float(x) for x in raw_roots[best_idx]],
        cosine_similarity=best_sim,
        inner_product=inner,
        root_type=root_type,
        top_5_roots=top_5,
    )


# ─────────────────────────────────────────────────────────────────
# Analysis: run tower + E8 analysis on experiment data
# ─────────────────────────────────────────────────────────────────

def analyze_tower(results_path: str) -> dict:
    """Run full tower + E8 analysis on an experiment results file."""
    with open(results_path) as f:
        results = json.load(f)

    runs = results.get('runs', [])
    budgets = sorted(set(r['budget'] for r in runs), reverse=True)
    models = sorted(set(r['model'] for r in runs))

    report = {
        'tower_distribution': {},
        'per_model_descent': {},
        'e8_comparisons': [],
        'summary': {},
    }

    # ── Tower distribution by budget ──
    for budget in budgets:
        level_counts = {'R': 0, 'C': 0, 'H': 0, 'O': 0}
        budget_runs = [r for r in runs if r['budget'] == budget]
        for r in budget_runs:
            lv = classify_tower_level(r['operator_count'])
            level_counts[lv] += 1
        total = len(budget_runs) or 1
        report['tower_distribution'][budget] = {
            level: {'count': count, 'pct': round(count / total * 100, 1)}
            for level, count in level_counts.items()
        }

    # ── Per-model descent signatures ──
    for model in models:
        model_levels = []
        for budget in budgets:
            model_budget_runs = [r for r in runs
                                 if r['model'] == model and r['budget'] == budget]
            if model_budget_runs:
                avg_ops = sum(r['operator_count'] for r in model_budget_runs) / len(model_budget_runs)
                model_levels.append(classify_tower_level(round(avg_ops)))
        report['per_model_descent'][model] = {
            'levels': model_levels,
            'signature': tower_descent_signature(model_levels),
        }

    # ── E8 comparisons ──
    e8_available = Path('data/e8_roots.json').exists()
    if e8_available:
        for r in runs:
            verbs = r.get('unique_verbs', [])
            if not verbs:
                continue
            comp = compare_to_e8(verbs, r['operator_count'])
            if comp:
                report['e8_comparisons'].append({
                    'model': r['model'],
                    'task_id': r['task_id'],
                    'budget': r['budget'],
                    'operator_count': r['operator_count'],
                    'tower_level': comp.tower_level,
                    'cosine_to_nearest_e8': comp.cosine_similarity,
                    'nearest_root_idx': comp.nearest_root_idx,
                    'root_type': comp.root_type,
                    'verbs': verbs,
                })

    # ── Summary statistics ──
    if report['e8_comparisons']:
        cosines = [c['cosine_to_nearest_e8'] for c in report['e8_comparisons']]
        by_level = {}
        for c in report['e8_comparisons']:
            by_level.setdefault(c['tower_level'], []).append(c['cosine_to_nearest_e8'])

        report['summary'] = {
            'total_comparisons': len(cosines),
            'mean_cosine': round(float(np.mean(cosines)), 4),
            'max_cosine': round(float(np.max(cosines)), 4),
            'by_tower_level': {
                level: {
                    'count': len(sims),
                    'mean_cosine': round(float(np.mean(sims)), 4),
                    'max_cosine': round(float(np.max(sims)), 4),
                }
                for level, sims in sorted(by_level.items())
            },
            'integer_root_pct': round(
                sum(1 for c in report['e8_comparisons'] if c['root_type'] == 'integer')
                / len(report['e8_comparisons']) * 100, 1
            ),
        }

    return report


def print_tower_report(report: dict):
    """Print a formatted tower + E8 analysis report."""
    print("\n" + "=" * 64)
    print("R -> C -> H -> O   TOWER ANALYSIS")
    print("Cayley-Dickson normed division algebra classification")
    print("=" * 64)

    # Tower distribution
    print(f"\n  {'Budget':>7} | {'R (dim 1)':>10} | {'C (dim 2)':>10} | "
          f"{'H (dim 4)':>10} | {'O (dim 8)':>10}")
    print(f"  {'─' * 56}")
    for budget, dist in sorted(report['tower_distribution'].items(), reverse=True):
        r_pct = dist['R']['pct']
        c_pct = dist['C']['pct']
        h_pct = dist['H']['pct']
        o_pct = dist['O']['pct']
        print(f"  {budget:>7} | {r_pct:>8.1f}% | {c_pct:>8.1f}% | "
              f"{h_pct:>8.1f}% | {o_pct:>8.1f}%")

    # Descent signatures
    print(f"\n  Model Descent Signatures:")
    print(f"  {'─' * 50}")
    for model, data in sorted(report['per_model_descent'].items()):
        sig = data['signature']
        levels = ' '.join(data['levels'])
        print(f"    {model:<16} {levels:<20} {sig}")

    # E8 comparison
    if report.get('summary'):
        s = report['summary']
        print(f"\n  E8 Root Vector Comparison ({s['total_comparisons']} geometries):")
        print(f"  {'─' * 50}")
        print(f"    Mean cosine to nearest E8 root: {s['mean_cosine']:.4f}")
        print(f"    Max cosine to nearest E8 root:  {s['max_cosine']:.4f}")
        print(f"    Integer root affinity:           {s['integer_root_pct']:.1f}%")

        print(f"\n    {'Level':>6} | {'Count':>6} | {'Mean cos':>9} | {'Max cos':>9}")
        print(f"    {'─' * 38}")
        for level, stats in sorted(s['by_tower_level'].items()):
            print(f"    {level:>6} | {stats['count']:>6} | "
                  f"{stats['mean_cosine']:>9.4f} | {stats['max_cosine']:>9.4f}")

    print("\n" + "=" * 64)


def run_tower_analysis():
    """Run tower + E8 analysis on the latest experiment."""
    results_dir = Path('results')
    if not results_dir.exists():
        print("No results directory found.")
        return

    # Find latest experiment file
    exp_files = sorted(results_dir.glob('experiment_*.json'))
    if not exp_files:
        print("No experiment results found. Run the main experiment first.")
        return

    latest = str(exp_files[-1])
    print(f"Analyzing: {latest}")

    report = analyze_tower(latest)
    print_tower_report(report)

    # Save
    out_path = results_dir / f"tower_analysis_{Path(latest).stem.split('_', 1)[1]}.json"
    with open(out_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  Saved: {out_path}")

    return report


if __name__ == '__main__':
    run_tower_analysis()
