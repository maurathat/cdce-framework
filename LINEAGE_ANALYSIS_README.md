# Lineage Trajectory Analysis — CDCE

Measures the trajectory geometry of compression lineages: do recursively-compressed
reasoning strategies *march* toward an attractor, *wander*, or *spiral in*?

Motivated by the emergent temporal-straightening result in LeWorldModel
(Maes et al. 2026, arXiv:2603.19312, App. H), where latent trajectories straighten
over training. This asks the analogous question of CDCE compression lineages.

## Result (992 geometries, 42 lineage chains ≥ 3 generations, May 2026)

**Damped oscillatory convergence — the lineages spiral inward.**

| Measurement | Value | Reading |
|---|---|---|
| Progress ratio R (net displacement / path) | 0.060 | below random-walk |
| Random-walk expectation | 0.195 | — |
| Chains beating expectation | 0 / 42 | not a straight march |
| Velocity cosine (consecutive steps) | −0.48 | steps reverse (oscillation) |
| Distance-to-centroid, binary | 24/42 shrinking, slope −0.010 | **contraction** |
| Distance-to-centroid, count-weighted | 9/12 shrinking, slope −0.019 | **stronger contraction** |

Contraction-toward-centroid **with** negative velocity cosine **with** below-random
net progress is a single geometry — an inward **spiral** — and the only one of four
candidates consistent with all three signals:

- random walk → distance-to-centroid grows (it shrinks) ✗
- straight march → velocity cosine positive (it's negative) ✗
- set-membership churn artifact → distance flat, weakens under count-weighting
  (it contracts and *strengthens*) ✗
- **damped spiral (basin residence) → all three signals match** ✓

The count-weighting *strengthening* the contraction (rather than washing it out) is
the test that distinguishes real basin confinement from binary set-membership noise.

**Interpretation.** LeWM sees straightening because its system is still *approaching*
an attractor (transient). CDCE lineages, compressed 38 generations deep, are already
*within* the basin and show the steady-state signature: confined oscillation. The two
are complementary phases of one attractor phenomenon.

**Confidence.** The contraction is robust on the 42-chain binary measurement and
strengthens under count-weighting; the count-weighted figure rests on 12 fully-covered
chains (3 still grow), so it corroborates rather than carries. The claim stands on the
binary result.

**Open follow-up.** A damped oscillator has a characteristic decay rate, so the
contraction slope should scale with compression pressure — tighter token budgets →
faster inward spiral. Testable on the same corpus.

## Usage

```bash
# straightening + direction metrics
python3 temporal_straightening.py --memory ./memory

# add the decisive basin-confinement test
python3 temporal_straightening.py --memory ./memory --centroid
```

**Note:** The public repo includes a 30-entry sample under `memory/samples/`. The full
992-entry corpus that produces the reported figures is archived separately.

The memory store may be a JSON list, a JSON dict keyed by `uor_addr`, or a directory
of per-entry JSON files. Each entry uses the Protocol v0.3 MemoryEntry schema:
`uor_addr`, `generation`, `lineage`, `operator_set`, `operator_count`, `model`,
`task_id`, `task_family`, and optionally `operator_multiset` (count-weighted).

### Building count-weighted vectors

If the store lacks `operator_multiset`, re-extract it from `strategy_text` using the
harness's verb vocabulary, counting every occurrence (not just presence). The CDCE
finding requires this for the count-weighted comparison; the binary result needs only
`operator_set`.

## Method notes

- **Operator vector**: count-weighted (`operator_multiset`) when available, else binary
  presence over `operator_set`. Binary presence cannot distinguish a directed march
  from a random walk when every step touches fresh operators (one-hot velocities are
  orthogonal) — the script's regime diagnostic flags this via consecutive-generation
  operator overlap (Jaccard).
- **Progress ratio** is the primary straightness metric (robust to single-coordinate
  moves); velocity cosine is reported for LeWM comparability but is caveated.
- **Centroid distance** is the decisive measurement; run it (`--centroid`).
- **Baselines are built in**: random-walk expectation (1/√path) for the progress ratio;
  the binary-vs-count comparison for the churn-artifact control.

## Files

- `temporal_straightening.py` — straightening + direction + centroid analysis (this is
  the canonical analysis script)
- `add_multiset.py` — (Mac Mini, local) re-extracts `operator_multiset` from
  `strategy_text` using the harness's 70-verb vocabulary

Companion to AGI Architecture Predictions §4.7 (where this result is written up) and
the EBM-over-corpus analysis (the energy-landscape view of the same attractor).
