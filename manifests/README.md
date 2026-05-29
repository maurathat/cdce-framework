# §6.4 v0.4.1 Corpus Manifests

## Canonical (these three define the published corpus)

- `section_6_4_v04_20260528_152116.json` — Phase 1 opt_routing, 54 cells, 1 rep each
- `section_6_4_v04_20260528_152507.json` — Phase 1 trans_nl_code, 54 cells, 1 rep each
- `section_6_4_v04_20260528_155940.json` — Phase 2 targeted replication, 18 cells, 10 reps each

Combined: 288 cell executions, 286 unique trace_κs after dedup, 0 verification failures.

To reproduce all numerical claims:

```bash
python3 analyze_section_6_4_corpus.py \
    manifests/section_6_4_v04_20260528_152116.json \
    manifests/section_6_4_v04_20260528_152507.json \
    manifests/section_6_4_v04_20260528_155940.json
```

## Non-canonical (provenance)

Other manifests in this directory are dry-runs, schema v0.4.0 antecedents, or
intermediate runs preserved for history. They are NOT part of the published
corpus and should not be passed to `analyze_section_6_4_corpus.py`.

- `section_6_4_v04_20260528_135548.json` — dry-run, pre-v0.4.1 schema, full grid (108 cells). First dry-run of the driver.
- `section_6_4_v04_20260528_135759.json` — dry-run, pre-v0.4.1 schema, full grid (108 cells). Second dry-run after adding source-model homogeneity note.
- `section_6_4_v04_20260528_140122.json` — live, pre-v0.4.1, 6 cells. First `--limit 6` scorer verification run (scorer had first-list bug).
- `section_6_4_v04_20260528_140555.json` — live, pre-v0.4.1, 1 cell. Single-cell diagnostic run with scorer print instrumentation.
- `section_6_4_v04_20260528_140812.json` — live, pre-v0.4.1, 6 cells. Second `--limit 6` run after `_numbers_from_text` last-list fix.
- `section_6_4_v04_20260528_141209.json` — live, pre-v0.4.1, 6 cells. Third `--limit 6` run after `expected_len` scorer tightening.
- `section_6_4_v04_20260528_141341.json` — live, pre-v0.4.1, 108 cells. Full v0.4.0 corpus attempt; 54 trans_nl_code OK, 54 opt_routing failed with `uor-addr invalid-input` (NaN-in-JSON bug, fixed by v0.4.1 `graded` field).
- `section_6_4_v04_20260528_151519.json` — dry-run, pre-v0.4.1, opt_routing only (54 cells). Cost estimate check before Phase 1.
- `section_6_4_v04_20260528_151524.json` — live, pre-v0.4.1, 3 cells. `--limit 3` opt_routing verification after `graded` field fix.
- `section_6_4_v04_20260528_155408.json` — dry-run, v0.4.1, Phase 2 full grid (180 cells). Cost estimate for Phase 2.
- `section_6_4_v04_20260528_155500.json` — dry-run, v0.4.1, Phase 2 `--limit 3` (30 cells). Verification slice plan.
- `section_6_4_v04_20260528_155515.json` — live, v0.4.1, 10 cells. Phase 2 `--limit 1` sanity check (Gemini × opt_routing × no_strategy × 2000, 10 reps).
