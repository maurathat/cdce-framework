#!/usr/bin/env python3
"""
Temporal Straightening Analysis for CDCE Compression Lineages
=============================================================

Tests whether compressed-reasoning lineages straighten across generations,
mirroring the emergent temporal-straightening result in LeWorldModel
(Maes et al. 2026, App. H): latent trajectories become increasingly straight
over training, measured as mean cosine similarity between consecutive
velocity vectors.

Here the "trajectory" is a lineage chain of compressed strategies across
compression generations. The "position" at each generation is the strategy's
operator vector; the "velocity" is the generation-to-generation delta; and
"straightness" is the mean cosine similarity between consecutive velocities.

  position   p_g   = operator_vector(strategy at generation g)
  velocity   v_g   = p_{g+1} - p_g
  straightness S    = mean_g  cos(v_g, v_{g+1})

  S -> 1  : straight trajectory (consecutive steps collinear) -> attractor pull
  S ~ 0   : random-walk-like (no directional persistence)
  S < 0   : oscillating / overshooting

This is the CDCE attractor claim in its cleanest, least-foolable form:
a single scalar, on existing data, directly comparable to a published
external finding. No new training, no E8, no geometry hypothesis.

USAGE
-----
    python temporal_straightening.py --memory /path/to/memory_store.json

Expects a memory store that is either:
  (a) a JSON list of MemoryEntry dicts, or
  (b) a JSON dict mapping uor_addr -> MemoryEntry dict, or
  (c) a directory of per-entry JSON files.

Each MemoryEntry should have (per Protocol v0.3 schema):
    uor_addr (or content_hash), generation,
    lineage (list of prior uor_addrs/content_hashes),
    operator_set (or unique_verbs) (list[str]) OR operator_count (int),
    model, task_id, task_family
Extra fields are ignored. Missing optional fields degrade gracefully.

The operator VECTOR is built from operator_set against a fixed global
vocabulary (so vectors are comparable across entries). If only
operator_count is present, a 1-D fallback trajectory is used and a warning
is printed (count-only straightening is weaker evidence — it cannot see
WHICH operators changed, only how many).
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np


# --------------------------------------------------------------------------
# Field normalization — adapt harness field names to script expectations
# --------------------------------------------------------------------------

def normalize_entry(e):
    """Map harness field names to the canonical names the analysis expects.

    content_hash  -> uor_addr
    unique_verbs  -> operator_set
    """
    if "uor_addr" not in e and "content_hash" in e:
        e["uor_addr"] = e["content_hash"]
    if "operator_set" not in e and "unique_verbs" in e:
        e["operator_set"] = e["unique_verbs"]
    return e


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_entries(path):
    """Load MemoryEntry records from a json list, json dict, or directory."""
    entries = []
    if os.path.isdir(path):
        for fn in sorted(os.listdir(path)):
            if fn.endswith(".json"):
                with open(os.path.join(path, fn)) as f:
                    obj = json.load(f)
                    entries.extend(obj if isinstance(obj, list) else [obj])
    else:
        with open(path) as f:
            obj = json.load(f)
        if isinstance(obj, list):
            entries = obj
        elif isinstance(obj, dict):
            # dict keyed by addr, OR a wrapper like {"entries": [...]}
            if "entries" in obj and isinstance(obj["entries"], list):
                entries = obj["entries"]
            else:
                entries = list(obj.values())
        else:
            raise ValueError("Unrecognized memory store format")
    # normalize field names, then keep only dict-shaped records with an addr
    return [normalize_entry(e) for e in entries
            if isinstance(e, dict) and ("uor_addr" in e or "content_hash" in e)]


# --------------------------------------------------------------------------
# Vocabulary + vectorization
# --------------------------------------------------------------------------

def build_vocabulary(entries):
    """Fixed global operator vocabulary, sorted for determinism."""
    vocab = set()
    for e in entries:
        for op in e.get("operator_set", []) or []:
            vocab.add(op)
        for op in e.get("operator_multiset", []) or []:
            vocab.add(op)
    return sorted(vocab)


def operator_vector(entry, vocab_index):
    """Operator vector over the global vocabulary.

    Prefers a COUNT-weighted vector when operator multiplicity is available
    (`operator_multiset`, or reconstructable from reuse_ratio + operator_count),
    because count-weighting is what makes directed reinforcement visible: an
    attractor-march that keeps emphasizing the same operators shows up as
    growing magnitude along a stable direction. Falls back to binary presence
    (the operator *set*) otherwise.

    NOTE: binary presence cannot distinguish a directed march from a random
    walk when every step touches fresh operators (one-hot velocities are
    orthogonal). The regime diagnostic in the report flags when this matters.
    """
    v = np.zeros(len(vocab_index), dtype=float)
    multiset = entry.get("operator_multiset")
    if multiset:
        for op in multiset:
            idx = vocab_index.get(op)
            if idx is not None:
                v[idx] += 1.0
        return v
    # fallback: binary presence over operator_set
    for op in (entry.get("operator_set", []) or []):
        idx = vocab_index.get(op)
        if idx is not None:
            v[idx] = 1.0
    return v


def operator_vector_binary(entry, vocab_index):
    """Binary presence vector (operator_set only, ignoring multiplicity)."""
    v = np.zeros(len(vocab_index), dtype=float)
    for op in (entry.get("operator_set", []) or []):
        idx = vocab_index.get(op)
        if idx is not None:
            v[idx] = 1.0
    return v


def operator_vector_count(entry, vocab_index):
    """Count-weighted vector from operator_multiset. Returns None if absent."""
    v = np.zeros(len(vocab_index), dtype=float)
    multiset = entry.get("operator_multiset")
    if not multiset:
        return None
    for op in multiset:
        idx = vocab_index.get(op)
        if idx is not None:
            v[idx] += 1.0
    return v


# --------------------------------------------------------------------------
# Lineage reconstruction
# --------------------------------------------------------------------------

def reconstruct_chains(entries):
    """Group entries into ordered lineage chains.

    Strategy: index by uor_addr; a chain is a maximal path following the
    `lineage` parent pointers. We emit one trajectory per leaf (an entry
    that is no one's parent), walking back through parents, then ordering
    by generation ascending.
    """
    by_addr = {e["uor_addr"]: e for e in entries}
    # immediate parent = last element of lineage list, if present
    def parent_of(e):
        lin = e.get("lineage") or []
        if lin:
            p = lin[-1]
            return p if p in by_addr else None
        return None

    children = defaultdict(list)
    has_parent = set()
    for e in entries:
        p = parent_of(e)
        if p is not None:
            children[p].append(e["uor_addr"])
            has_parent.add(e["uor_addr"])

    # leaves = entries that are not a parent of anything
    all_parents = set(children.keys())
    leaves = [a for a in by_addr if a not in all_parents]

    chains = []
    for leaf in leaves:
        chain = []
        cur = by_addr[leaf]
        seen = set()
        while cur is not None and cur["uor_addr"] not in seen:
            seen.add(cur["uor_addr"])
            chain.append(cur)
            p = parent_of(cur)
            cur = by_addr.get(p) if p else None
        chain.reverse()  # oldest -> newest
        if len(chain) >= 3:  # need >=3 points for >=2 velocities -> >=1 cosine
            chains.append(chain)

    # Fallback: if lineage pointers are absent/empty, group by
    # (model, task_id) and order by generation. This recovers trajectories
    # when the store didn't persist explicit parent chains.
    if not chains:
        groups = defaultdict(list)
        for e in entries:
            key = (e.get("model", "?"), e.get("task_id", e.get("task_family", "?")))
            groups[key].append(e)
        for key, grp in groups.items():
            grp = [g for g in grp if "generation" in g]
            grp.sort(key=lambda g: g["generation"])
            if len(grp) >= 3:
                chains.append(grp)

    return chains


# --------------------------------------------------------------------------
# Straightening metric
# --------------------------------------------------------------------------

def cos(a, b, eps=1e-12):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < eps or nb < eps:
        return None  # undefined velocity (no change between generations)
    return float(np.dot(a, b) / (na * nb))


def chain_straightness(chain, vocab_index):
    """Velocity-collinearity straightness: mean cos between consecutive
    velocity vectors. This is the literal LeWM metric.

    CAVEAT (important for binary/sparse operator vectors): if each generation
    changes a DIFFERENT operator, consecutive single-step velocities are
    orthogonal by construction (they touch disjoint coordinates), so this
    metric reads ~0 even for a perfectly directed march toward an attractor.
    Use `chain_progress_straightness` as the primary metric for set-valued
    trajectories; keep this one for comparability with LeWM's formulation.

    Returns (mean_cos, n_valid_pairs, per_step_coses).
    """
    positions = [operator_vector(e, vocab_index) for e in chain]
    velocities = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
    coses = []
    for i in range(len(velocities) - 1):
        c = cos(velocities[i], velocities[i + 1])
        if c is not None:
            coses.append(c)
    if not coses:
        return None, 0, []
    return float(np.mean(coses)), len(coses), coses


def chain_progress_straightness(chain, vocab_index):
    """Progress-ratio straightness, robust to single-coordinate moves.

    Straightness = net displacement / total path length:

        R = || p_end - p_start ||  /  sum_g || p_{g+1} - p_g ||

    R -> 1 : every step advances the same net direction (straight march to
             an attractor) -- INCLUDING the case where each step flips a new
             bit toward a fixed target set, which the velocity-cosine metric
             misses.
    R ~ 1/sqrt(n) : random-walk-like (displacement grows ~sqrt of path).
    R -> 0 : returns near start / oscillates (no net progress).

    This is the better primary metric for set-valued (binary) operator
    trajectories. Returns (R, path_length, net_displacement).
    """
    positions = [operator_vector(e, vocab_index) for e in chain]
    steps = [np.linalg.norm(positions[i + 1] - positions[i])
             for i in range(len(positions) - 1)]
    total_path = float(np.sum(steps))
    net = float(np.linalg.norm(positions[-1] - positions[0]))
    if total_path < 1e-12:
        return None, 0.0, 0.0  # no motion at all (already at fixed point)
    return net / total_path, total_path, net


# --------------------------------------------------------------------------
# Baselines (the anti-fooling controls)
# --------------------------------------------------------------------------

def shuffled_baseline(chains, vocab_index, n_shuffle=200, seed=0):
    """Null model: shuffle the generation ORDER within each chain.

    If straightness is a real temporal phenomenon, real-order straightness
    should exceed shuffled-order straightness. If they match, the apparent
    straightening is an artifact of the vectors' static geometry, not of
    the compression dynamics. This is the same discipline as the E8 random
    baselines: a result only counts if it beats its order-destroying null.
    """
    rng = np.random.default_rng(seed)
    real_vals, shuf_means = [], []
    for chain in chains:
        s, n, _ = chain_straightness(chain, vocab_index)
        if s is None:
            continue
        real_vals.append(s)
        positions = [operator_vector(e, vocab_index) for e in chain]
        shuffs = []
        for _ in range(n_shuffle):
            order = rng.permutation(len(positions))
            shuf_pos = [positions[i] for i in order]
            vels = [shuf_pos[i + 1] - shuf_pos[i] for i in range(len(shuf_pos) - 1)]
            cs = [cos(vels[i], vels[i + 1]) for i in range(len(vels) - 1)]
            cs = [c for c in cs if c is not None]
            if cs:
                shuffs.append(np.mean(cs))
        if shuffs:
            shuf_means.append(np.mean(shuffs))
    return np.array(real_vals), np.array(shuf_means)


# --------------------------------------------------------------------------
# Centroid-distance analysis
# --------------------------------------------------------------------------

def analyze_centroid_distances(chains, vocab_index, vec_fn, label):
    """For each chain, compute distance-to-centroid at each generation.

    Classify each chain's trend via linear regression slope on
    (generation_index, distance_to_centroid):
      slope < -0.005 : shrinking
      slope >  0.005 : growing
      else           : flat/bounded
    """
    print(f"\n{'='*60}")
    print(f"  CENTROID-DISTANCE ANALYSIS — {label}")
    print(f"{'='*60}")

    trends = {"shrinking": 0, "flat": 0, "growing": 0}
    all_slopes = []
    all_early_dist = []
    all_late_dist = []
    skipped = 0

    for ci, chain in enumerate(chains):
        vecs = [vec_fn(e, vocab_index) for e in chain]
        if any(v is None for v in vecs):
            skipped += 1
            continue

        positions = np.array(vecs)
        centroid = positions.mean(axis=0)
        dists = np.array([np.linalg.norm(p - centroid) for p in positions])

        if dists.max() < 1e-12:
            trends["flat"] += 1
            all_slopes.append(0.0)
            continue

        x = np.arange(len(dists), dtype=float)
        x_centered = x - x.mean()
        slope = np.dot(x_centered, dists) / np.dot(x_centered, x_centered)
        all_slopes.append(slope)

        half = len(dists) // 2
        all_early_dist.append(dists[:half].mean())
        all_late_dist.append(dists[half:].mean())

        if slope < -0.005:
            trends["shrinking"] += 1
        elif slope > 0.005:
            trends["growing"] += 1
        else:
            trends["flat"] += 1

    n_measured = len(all_slopes)
    print(f"  Chains measured: {n_measured}  (skipped {skipped})")
    if n_measured == 0:
        print("  No chains to analyze.")
        return

    slopes = np.array(all_slopes)
    print(f"\n  Per-chain trend classification (slope threshold ±0.005):")
    print(f"    Shrinking : {trends['shrinking']:3d} / {n_measured}"
          f"  ({100*trends['shrinking']/n_measured:.1f}%)")
    print(f"    Flat      : {trends['flat']:3d} / {n_measured}"
          f"  ({100*trends['flat']/n_measured:.1f}%)")
    print(f"    Growing   : {trends['growing']:3d} / {n_measured}"
          f"  ({100*trends['growing']/n_measured:.1f}%)")

    print(f"\n  Aggregate slope (dist-to-centroid vs generation index):")
    print(f"    mean slope  : {slopes.mean():+.5f}")
    print(f"    median slope: {np.median(slopes):+.5f}")
    print(f"    std          : {slopes.std():.5f}")

    if all_early_dist:
        early = np.array(all_early_dist)
        late = np.array(all_late_dist)
        print(f"\n  Early-half vs late-half mean distance to centroid:")
        print(f"    early mean dist : {early.mean():.4f}")
        print(f"    late  mean dist : {late.mean():.4f}")
        print(f"    late - early    : {(late.mean() - early.mean()):+.4f}")
        if late.mean() - early.mean() < -0.05:
            print("    -> Distances SHRINK: lineages converge toward centroid")
            print("       over generations (contraction toward attractor).")
        elif late.mean() - early.mean() > 0.05:
            print("    -> Distances GROW: lineages diverge from centroid")
            print("       (expansion, not contraction).")
        else:
            print("    -> Distances roughly STABLE across generations.")

    print(f"\n  INTERPRETATION:")
    dominant = max(trends, key=trends.get)
    if dominant == "shrinking":
        print("  Majority of chains contract toward their centroid over")
        print("  generations — consistent with convergence to a basin.")
    elif dominant == "growing":
        print("  Majority of chains move AWAY from their centroid over")
        print("  generations — exploration or divergence, not contraction.")
    else:
        print("  Most chains stay at roughly constant distance from centroid —")
        print("  bounded wandering, neither contracting nor diverging.")


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="CDCE temporal straightening analysis")
    ap.add_argument("--memory", required=True,
                    help="Path to memory store (json file or directory)")
    ap.add_argument("--centroid", action="store_true",
                    help="Also run centroid-distance contraction analysis")
    ap.add_argument("--shuffles", type=int, default=200,
                    help="Permutations for the shuffled-order null (default 200)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    entries = load_entries(args.memory)
    print(f"Loaded {len(entries)} memory entries.")

    has_sets = sum(1 for e in entries if e.get("operator_set"))
    if has_sets == 0:
        print("\nWARNING: no entries have an `operator_set` field.")
        print("Cannot build operator vectors. If only `operator_count` exists,")
        print("count-only straightening is much weaker evidence (it sees how")
        print("MANY operators changed, not WHICH). Aborting; re-export the store")
        print("with operator_set to run the real analysis.")
        sys.exit(1)

    vocab = build_vocabulary(entries)
    vocab_index = {op: i for i, op in enumerate(vocab)}
    print(f"Operator vocabulary size: {len(vocab)}")

    chains = reconstruct_chains(entries)
    print(f"Reconstructed {len(chains)} lineage chains of length >= 3.")
    if not chains:
        print("No chains long enough to measure straightening (need >= 3 generations).")
        sys.exit(1)

    lengths = [len(c) for c in chains]
    print(f"Chain lengths: min {min(lengths)}, "
          f"median {int(np.median(lengths))}, max {max(lengths)}")

    # ---- REGIME DIAGNOSTIC ----
    has_multiset = sum(1 for e in entries if e.get("operator_multiset"))
    print("\n=== REPRESENTATION REGIME ===")
    if has_multiset:
        print(f"  Count-weighted vectors available ({has_multiset} entries with"
              f" operator_multiset). Directed reinforcement IS measurable.")
    else:
        print("  Binary presence vectors only (operator_set, no multiplicity).")
        print("  CAVEAT: if lineages touch fresh operators each generation,")
        print("  progress ratio sits near the random-walk line BY CONSTRUCTION")
        print("  and a null result is uninformative. Re-export with operator")
        print("  multiplicity (reuse_ratio implies it) to make the test sharp.")
    # revisit rate: how often do consecutive generations touch shared operators?
    revisit = []
    for chain in chains:
        sets = [set(e.get("operator_set", []) or []) for e in chain]
        for i in range(len(sets) - 1):
            union = sets[i] | sets[i + 1]
            inter = sets[i] & sets[i + 1]
            if union:
                revisit.append(len(inter) / len(union))
    if revisit:
        rr = float(np.mean(revisit))
        print(f"  Mean consecutive-generation operator overlap (Jaccard): {rr:.3f}")
        if rr < 0.2:
            print("  -> LOW overlap: lineages churn operators; straightness in")
            print("     SET space will be near-null regardless. Trust count")
            print("     vectors or treat a null as 'not measurable here'.")
        else:
            print("  -> Adequate overlap: set-space straightness is meaningful.")

    # Per-chain straightness — BOTH metrics
    per_chain = []          # velocity-cosine (LeWM-comparable)
    per_chain_progress = [] # progress-ratio (primary for binary trajectories)
    for chain in chains:
        s, n, coses = chain_straightness(chain, vocab_index)
        R, path, net = chain_progress_straightness(chain, vocab_index)
        if R is not None:
            per_chain_progress.append((chain, R, path, net))
        if s is not None:
            per_chain.append((chain, s, n))

    if not per_chain_progress:
        print("\nAll trajectories have zero path length (no operator-set "
              "changes across generations). That is itself a finding: the "
              "lineages are *already* at a fixed point — convergence by stasis.")
        sys.exit(0)

    # ---- PRIMARY METRIC: progress ratio ----
    Rvals = np.array([R for _, R, _, _ in per_chain_progress])
    # random-walk expectation for each chain: ~ 1/sqrt(n_steps)
    rw_expect = np.array([1.0 / np.sqrt(max(path, 1.0))
                          for _, R, path, _ in per_chain_progress])
    print("\n=== PROGRESS-RATIO STRAIGHTNESS (primary; robust to bit-flips) ===")
    print("  R = net displacement / total path length")
    print(f"  chains measured       : {len(Rvals)}")
    print(f"  mean R                : {Rvals.mean():.4f}")
    print(f"  median R              : {np.median(Rvals):.4f}")
    print(f"  random-walk expect.   : {rw_expect.mean():.4f}  (mean 1/sqrt path)")
    adv = Rvals.mean() - rw_expect.mean()
    print(f"  advantage over RW     : {adv:+.4f}")
    n_beat = int(np.sum(Rvals > rw_expect))
    print(f"  chains R > RW-expect  : {n_beat}/{len(Rvals)}")
    print("\n  INTERPRETATION:")
    if adv > 0.10 and n_beat > 0.6 * len(Rvals):
        print("  Lineages make directed net progress well above random-walk")
        print("  expectation — strategies march toward an attractor rather than")
        print("  wandering. This is the cross-substrate analogue of LeWM's")
        print("  emergent straightening and a clean instance of the CDCE")
        print("  stable-attractor claim, with no geometry/E8 commitment.")
    elif adv < -0.05:
        print("  Lineages make LESS net progress than a random walk — they")
        print("  oscillate or return toward earlier states. Convergence, if any,")
        print("  is not via a straight march. Informative boundary.")
    else:
        print("  Progress is ~random-walk: no directed-march signal. The")
        print("  attractor claim is not supported by trajectory straightness")
        print("  on this data (look to the EBM basin analysis instead).")

    # ---- SECONDARY METRIC: velocity cosine (LeWM-literal, caveated) ----
    svals = np.array([s for _, s, _ in per_chain])
    print("\n=== VELOCITY-COSINE STRAIGHTNESS (LeWM-literal; caveated) ===")
    print("  NOTE: for set-valued/binary operator vectors this reads ~0 when")
    print("  each step changes a different operator (orthogonal velocities),")
    print("  even under directed progress. Trust the progress ratio above.")
    print(f"  mean cos              : {svals.mean():+.4f}")
    print(f"  median                : {np.median(svals):+.4f}")

    # ---- Direction consistency: do steps share a net direction? ----
    # Cosine of each step against the chain's overall displacement vector.
    # High mean => steps consistently advance the net direction (a march).
    print("\n=== STEP/NET DIRECTION CONSISTENCY ===")
    dir_means = []
    for chain, R, path, net in per_chain_progress:
        positions = [operator_vector(e, vocab_index) for e in chain]
        netvec = positions[-1] - positions[0]
        if np.linalg.norm(netvec) < 1e-12:
            continue
        step_dirs = []
        for i in range(len(positions) - 1):
            c = cos(positions[i + 1] - positions[i], netvec)
            if c is not None:
                step_dirs.append(c)
        if step_dirs:
            dir_means.append(np.mean(step_dirs))
    if dir_means:
        dm = np.array(dir_means)
        print(f"  chains measured        : {len(dm)}")
        print(f"  mean step·net cosine   : {dm.mean():+.4f}")
        print("  (high => steps consistently advance the net direction;")
        print("   ~0 => steps scatter around the displacement; this is the")
        print("   bit-flip-robust read on 'directed march toward attractor')")

    # Trend check: does straightness INCREASE along generations?
    # (LeWM's claim is straightening *over training*, i.e. later steps straighter.)
    print("\n=== WITHIN-CHAIN TREND (does it straighten over generations?) ===")
    early_late = []
    for chain, s, n in per_chain:
        positions = [operator_vector(e, vocab_index) for e in chain]
        vels = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
        step_cos = [cos(vels[i], vels[i + 1]) for i in range(len(vels) - 1)]
        step_cos = [c for c in step_cos if c is not None]
        if len(step_cos) >= 2:
            half = len(step_cos) // 2
            early = np.mean(step_cos[:half]) if half > 0 else np.nan
            late = np.mean(step_cos[half:])
            if not np.isnan(early):
                early_late.append((early, late))
    if early_late:
        ea = np.array([e for e, _ in early_late])
        la = np.array([l for _, l in early_late])
        print(f"  chains with >=2 cosines : {len(early_late)}")
        print(f"  early-half mean cos     : {ea.mean():+.4f}")
        print(f"  late-half  mean cos     : {la.mean():+.4f}")
        print(f"  late - early            : {(la.mean()-ea.mean()):+.4f}")
        if la.mean() - ea.mean() > 0.05:
            print("  -> trajectories straighten over generations (LeWM-like).")
        elif la.mean() - ea.mean() < -0.05:
            print("  -> trajectories DE-straighten over generations.")
        else:
            print("  -> no clear within-chain trend.")
    else:
        print("  Chains too short for an early/late split (need >=4 generations).")

    # ---- CENTROID-DISTANCE CONTRACTION (optional) ----
    if args.centroid:
        analyze_centroid_distances(chains, vocab_index,
                                  operator_vector_binary,
                                  "BINARY (operator_set)")
        has_multiset = sum(1 for e in entries if e.get("operator_multiset"))
        if has_multiset > 0:
            analyze_centroid_distances(chains, vocab_index,
                                      operator_vector_count,
                                      "COUNT-WEIGHTED (operator_multiset)")
        else:
            print("\n  No operator_multiset data — skipping count-weighted"
                  " centroid analysis.")
            print("  Run add_multiset.py first to extract from strategy_text.")

    print("\nDone. (Report is descriptive; pair with the EBM basin analysis for")
    print("the energy-side view of the same attractor claim.)")


if __name__ == "__main__":
    main()
