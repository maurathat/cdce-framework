"""
CDCE Compression Harness — Decompression Round-Trip Test

Tests whether compressed strategies are stable attractors:
1. Self-decompression: Model A compresses -> Model A decompresses
2. Cross-model decompression: Model A compresses -> Model B decompresses
3. Round-trip: Compress -> Decompress -> Recompress. Hash match = fixed point.

Structural fidelity is measured via verb-set Jaccard similarity between
the original compressed strategy and the recompressed output.  A Jaccard
of 1.0 means the same reasoning operators survived the round-trip even
if the surface text changed.
"""
import json
import os
import time
from datetime import datetime

from src.config import RESULTS_DIR, get_active_models
from src.llm_clients import call_model, init_keys
from src.metrics import extract_metrics, metrics_to_dict
from src.memory import MemoryStore, compute_hash
from src.energy import compute_energy, energy_to_dict


DECOMPRESS_BUDGET = 2000
RECOMPRESS_BUDGET = 125


def _verb_jaccard(verbs_a: list, verbs_b: list) -> float:
    """Jaccard similarity between two verb lists."""
    a, b = set(verbs_a), set(verbs_b)
    if not a and not b:
        return 1.0
    union = a | b
    return len(a & b) / len(union) if union else 0.0


def build_decompress_prompt(compressed_strategy, task_id):
    return (
        f"Below is a highly compressed strategy for solving a task. "
        f"Expand it fully. Elaborate every step, explain the reasoning, "
        f"restore any details that were compressed away. "
        f"Reconstruct the complete approach.\n\n"
        f"COMPRESSED STRATEGY:\n---\n{compressed_strategy}\n---\n\n"
        f"Expand this into a full, detailed solution strategy."
    )


def build_recompress_prompt(decompressed_strategy, task_id):
    return (
        f"Below is a detailed strategy. Compress it to its absolute essence. "
        f"Use the fewest possible distinct operations and steps. "
        f"Every word must earn its place.\n\n"
        f"STRATEGY:\n---\n{decompressed_strategy}\n---\n\n"
        f"[CONSTRAINT: Your COMPLETE response must be under {RECOMPRESS_BUDGET} tokens. "
        f"Be maximally compressed.]"
    )


def run_decompression_test():
    print("=" * 60)
    print("CDCE DECOMPRESSION ROUND-TRIP TEST")
    print("Testing attractor stability of compressed geometries")
    print("=" * 60)

    init_keys()
    active_models = get_active_models()
    memory = MemoryStore()

    if not active_models:
        print("[ERROR] No API keys configured!")
        return

    print(f"\nActive models: {', '.join(active_models.keys())}")

    # ── Collect most compressed strategies (lowest budget) per model/task ──
    compressed = {}
    for model_name in active_models:
        for task_file in os.listdir(memory.memory_dir):
            if not task_file.endswith('.json') or task_file == 'memory_index.json':
                continue
            with open(os.path.join(memory.memory_dir, task_file)) as f:
                try:
                    entry = json.load(f)
                except Exception:
                    continue
            if entry.get('model') == model_name and entry.get('budget_at_creation') == RECOMPRESS_BUDGET:
                key = (model_name, entry['task_id'])
                existing = compressed.get(key)
                if not existing or entry.get('generation', 0) > existing.get('generation', 0):
                    compressed[key] = entry

    if not compressed:
        print("\nNo compressed strategies found at budget=125.")
        print("Run the main experiment first with full budget levels.")
        return

    print(f"\nFound {len(compressed)} compressed strategies at budget={RECOMPRESS_BUDGET}")

    results = {
        "test": "decompression_round_trip",
        "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "models": list(active_models.keys()),
        "self_decompress": [],
        "cross_decompress": [],
        "round_trips": [],
        "summary": {},
    }

    # ─────────────────────────────────────────────────────────────────
    # PHASE 1: Self-Decompression
    # ─────────────────────────────────────────────────────────────────
    print(f"\n{'─' * 50}")
    print("PHASE 1: Self-Decompression")
    print(f"{'─' * 50}")

    decompressed_cache = {}

    for (model_name, task_id), entry in sorted(compressed.items()):
        cfg = active_models.get(model_name)
        if not cfg:
            continue

        compressed_text = entry['strategy_text']
        compressed_hash = entry['content_hash']
        original_ops = entry.get('operator_count', 0)
        original_verbs = entry.get('unique_verbs', [])

        print(f"\n  {model_name}/{task_id} (gen {entry.get('generation')}, {original_ops} ops)")
        print(f"    Compressed addr: {compressed_hash[:40]}...")

        prompt = build_decompress_prompt(compressed_text, task_id)
        resp = call_model(
            provider=cfg["provider"],
            model=cfg["model"],
            prompt=prompt,
            max_tokens=DECOMPRESS_BUDGET,
            task_id=task_id,
        )

        if resp.content.startswith("[ERROR"):
            print(f"    [SKIP] {resp.content[:80]}")
            continue

        decomp_metrics = extract_metrics(
            resp.content, task_id, entry.get('task_family', ''),
            model_name, DECOMPRESS_BUDGET, resp.output_tokens,
        )

        expansion = decomp_metrics.word_count / max(len(compressed_text.split()), 1)
        print(f"    Decompressed: {decomp_metrics.word_count} words, "
              f"{decomp_metrics.operator_count} ops, "
              f"expansion={expansion:.1f}x")

        decompressed_cache[(model_name, task_id)] = {
            "text": resp.content,
            "hash": compute_hash(resp.content),
            "ops": decomp_metrics.operator_count,
            "words": decomp_metrics.word_count,
            "verbs": decomp_metrics.unique_verbs,
        }

        results["self_decompress"].append({
            "model": model_name,
            "task_id": task_id,
            "task_family": entry.get('task_family', ''),
            "generation": entry.get('generation', 0),
            "compressed_hash": compressed_hash,
            "compressed_ops": original_ops,
            "compressed_verbs": original_verbs,
            "decompressed_hash": compute_hash(resp.content),
            "decompressed_ops": decomp_metrics.operator_count,
            "decompressed_words": decomp_metrics.word_count,
            "decompressed_verbs": decomp_metrics.unique_verbs,
            "expansion_ratio": round(expansion, 2),
            "input_tokens": resp.input_tokens,
            "output_tokens": resp.output_tokens,
            "latency_ms": resp.latency_ms,
        })

    # ─────────────────────────────────────────────────────────────────
    # PHASE 2: Cross-Model Decompression
    # ─────────────────────────────────────────────────────────────────
    print(f"\n{'─' * 50}")
    print("PHASE 2: Cross-Model Decompression")
    print(f"{'─' * 50}")

    model_names = list(active_models.keys())
    for (source_model, task_id), entry in sorted(compressed.items()):
        compressed_text = entry['strategy_text']
        compressed_hash = entry['content_hash']

        for target_model in model_names:
            if target_model == source_model:
                continue
            cfg = active_models.get(target_model)
            if not cfg:
                continue

            print(f"  {source_model}/{task_id} → {target_model}...", end=" ", flush=True)

            prompt = build_decompress_prompt(compressed_text, task_id)
            resp = call_model(
                provider=cfg["provider"],
                model=cfg["model"],
                prompt=prompt,
                max_tokens=DECOMPRESS_BUDGET,
                task_id=task_id,
            )

            if resp.content.startswith("[ERROR"):
                print("[SKIP]")
                continue

            decomp_metrics = extract_metrics(
                resp.content, task_id, entry.get('task_family', ''),
                target_model, DECOMPRESS_BUDGET, resp.output_tokens,
            )

            # Compare with self-decompression
            self_decomp = decompressed_cache.get((source_model, task_id))
            hash_match_self = False
            verb_fidelity = 0.0
            if self_decomp:
                hash_match_self = compute_hash(resp.content) == self_decomp["hash"]
                verb_fidelity = _verb_jaccard(self_decomp["verbs"], decomp_metrics.unique_verbs)

            print(f"{decomp_metrics.operator_count} ops, "
                  f"verb_fidelity={verb_fidelity:.2f}, "
                  f"hash_match={hash_match_self}")

            results["cross_decompress"].append({
                "source_model": source_model,
                "target_model": target_model,
                "task_id": task_id,
                "task_family": entry.get('task_family', ''),
                "compressed_hash": compressed_hash,
                "decompressed_hash": compute_hash(resp.content),
                "decompressed_ops": decomp_metrics.operator_count,
                "decompressed_words": decomp_metrics.word_count,
                "decompressed_verbs": decomp_metrics.unique_verbs,
                "matches_self_decompress": hash_match_self,
                "verb_fidelity_vs_self": round(verb_fidelity, 4),
                "input_tokens": resp.input_tokens,
                "output_tokens": resp.output_tokens,
                "latency_ms": resp.latency_ms,
            })

    # ─────────────────────────────────────────────────────────────────
    # PHASE 3: Round-Trip Recompression
    # ─────────────────────────────────────────────────────────────────
    print(f"\n{'─' * 50}")
    print("PHASE 3: Round-Trip Recompression")
    print(f"{'─' * 50}")

    for (model_name, task_id), decomp in sorted(decompressed_cache.items()):
        cfg = active_models.get(model_name)
        if not cfg:
            continue

        original_entry = compressed.get((model_name, task_id))
        if not original_entry:
            continue

        original_hash = original_entry['content_hash']
        original_ops = original_entry.get('operator_count', 0)
        original_verbs = original_entry.get('unique_verbs', [])

        print(f"\n  {model_name}/{task_id}: ", end="", flush=True)

        prompt = build_recompress_prompt(decomp["text"], task_id)
        resp = call_model(
            provider=cfg["provider"],
            model=cfg["model"],
            prompt=prompt,
            max_tokens=RECOMPRESS_BUDGET,
            task_id=task_id,
        )

        if resp.content.startswith("[ERROR"):
            print("[SKIP]")
            continue

        recomp_hash = compute_hash(resp.content)
        recomp_metrics = extract_metrics(
            resp.content, task_id, original_entry.get('task_family', ''),
            model_name, RECOMPRESS_BUDGET, resp.output_tokens,
        )

        hash_match = recomp_hash == original_hash
        ops_delta = abs(recomp_metrics.operator_count - original_ops)
        verb_fidelity = _verb_jaccard(original_verbs, recomp_metrics.unique_verbs)

        if hash_match:
            status = "★ FIXED POINT"
        else:
            status = f"ops_delta={ops_delta}, verb_fidelity={verb_fidelity:.2f}"
        print(f"{recomp_metrics.operator_count} ops (was {original_ops}), {status}")

        results["round_trips"].append({
            "model": model_name,
            "task_id": task_id,
            "task_family": original_entry.get('task_family', ''),
            "generation": original_entry.get('generation', 0),
            "original_hash": original_hash,
            "original_ops": original_ops,
            "original_verbs": original_verbs,
            "recompressed_hash": recomp_hash,
            "recompressed_ops": recomp_metrics.operator_count,
            "recompressed_verbs": recomp_metrics.unique_verbs,
            "hash_match": hash_match,
            "ops_delta": ops_delta,
            "verb_fidelity": round(verb_fidelity, 4),
            "input_tokens": resp.input_tokens,
            "output_tokens": resp.output_tokens,
            "latency_ms": resp.latency_ms,
        })

    # ─────────────────────────────────────────────────────────────────
    # SUMMARY
    # ─────────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("ROUND-TRIP SUMMARY")
    print(f"{'=' * 60}")

    total_self = len(results["self_decompress"])
    total_cross = len(results["cross_decompress"])
    total_rt = len(results["round_trips"])
    fixed_points = sum(1 for r in results["round_trips"] if r["hash_match"])

    print(f"\n  Phase 1 — Self-Decompressions:   {total_self}")
    print(f"  Phase 2 — Cross-Model Transfers: {total_cross}")
    print(f"  Phase 3 — Round-Trips Completed: {total_rt}")

    # ── Per-model round-trip breakdown ──
    if total_rt:
        print(f"\n  {'Model':<16} {'RT':>4} {'Fixed':>6} {'Rate':>7} "
              f"{'Avg Δops':>9} {'Avg Fidelity':>13}")
        print(f"  {'─' * 58}")

        by_model = {}
        for r in results["round_trips"]:
            by_model.setdefault(r["model"], []).append(r)

        model_summaries = {}
        for model, rts in sorted(by_model.items()):
            n = len(rts)
            fp = sum(1 for r in rts if r["hash_match"])
            avg_delta = sum(r["ops_delta"] for r in rts) / n
            avg_fidelity = sum(r["verb_fidelity"] for r in rts) / n
            rate = fp / n if n else 0

            print(f"  {model:<16} {n:>4} {fp:>6} {rate:>6.0%} "
                  f"{avg_delta:>9.1f} {avg_fidelity:>13.3f}")

            model_summaries[model] = {
                "round_trips": n,
                "fixed_points": fp,
                "fixed_point_rate": round(rate, 4),
                "mean_ops_delta": round(avg_delta, 2),
                "mean_verb_fidelity": round(avg_fidelity, 4),
            }

        # ── Aggregate ──
        overall_rate = fixed_points / total_rt if total_rt else 0
        overall_delta = sum(r["ops_delta"] for r in results["round_trips"]) / total_rt
        overall_fidelity = sum(r["verb_fidelity"] for r in results["round_trips"]) / total_rt

        print(f"  {'─' * 58}")
        print(f"  {'TOTAL':<16} {total_rt:>4} {fixed_points:>6} {overall_rate:>6.0%} "
              f"{overall_delta:>9.1f} {overall_fidelity:>13.3f}")

        results["summary"]["overall"] = {
            "total_self_decompress": total_self,
            "total_cross_decompress": total_cross,
            "total_round_trips": total_rt,
            "fixed_points": fixed_points,
            "fixed_point_rate": round(overall_rate, 4),
            "mean_ops_delta": round(overall_delta, 2),
            "mean_verb_fidelity": round(overall_fidelity, 4),
        }
        results["summary"]["by_model"] = model_summaries

    # ── Cross-model fidelity matrix ──
    if total_cross:
        print(f"\n  Cross-Model Verb Fidelity (vs self-decompression):")

        # Build matrix: source_model → target_model → avg fidelity
        matrix = {}
        counts = {}
        for r in results["cross_decompress"]:
            src, tgt = r["source_model"], r["target_model"]
            key = (src, tgt)
            matrix.setdefault(key, 0.0)
            counts.setdefault(key, 0)
            matrix[key] += r["verb_fidelity_vs_self"]
            counts[key] += 1

        src_models = sorted(set(r["source_model"] for r in results["cross_decompress"]))
        tgt_models = sorted(set(r["target_model"] for r in results["cross_decompress"]))

        # Header
        header = f"  {'src \\ tgt':<14}"
        for tgt in tgt_models:
            header += f" {tgt:>12}"
        print(header)
        print(f"  {'─' * (14 + 13 * len(tgt_models))}")

        fidelity_matrix = {}
        for src in src_models:
            row = f"  {src:<14}"
            row_data = {}
            for tgt in tgt_models:
                key = (src, tgt)
                if key in matrix and counts[key] > 0:
                    avg = matrix[key] / counts[key]
                    row += f" {avg:>12.3f}"
                    row_data[tgt] = round(avg, 4)
                else:
                    row += f" {'—':>12}"
            print(row)
            fidelity_matrix[src] = row_data

        results["summary"]["cross_model_fidelity"] = fidelity_matrix

    # ── Expansion ratio stats ──
    if total_self:
        expansions = [r["expansion_ratio"] for r in results["self_decompress"]]
        avg_exp = sum(expansions) / len(expansions)
        print(f"\n  Avg expansion ratio (compress→decompress): {avg_exp:.1f}x")
        results["summary"]["mean_expansion_ratio"] = round(avg_exp, 2)

    print(f"\n{'=' * 60}")

    # ── Save results ──
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results_path = os.path.join(
        RESULTS_DIR, f"decompress_{results['timestamp']}.json"
    )
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {results_path}")

    return results
