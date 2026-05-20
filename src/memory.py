"""
CDCE Compression Harness — Memory Persistence Layer

Stores compressed strategies (geometric structures) between experiment runs.
This is what makes the compression recursive across sessions, not just
within a single run.

The key insight: we don't store answers. We store compressed operational
geometry — the strategy itself. Over multiple runs, these structures
get compressed further, and we track whether they converge toward
exceptional algebraic signatures.

Storage hierarchy:
1. Local JSON store (always available, zero dependencies)
2. UOR canonical content-addressing via uor-addr κ-labels

Each stored memory has:
- A UOR κ-label (canonical identity — same strategy = same address)
- The compressed strategy text
- Metrics at time of storage
- A generation counter (how many compression cycles produced this)
- Lineage (chain of addresses showing compression history)

Address format:
  sha256:<64-hex-digits>  (71 bytes, deterministic via JCS+NFC+SHA-256)
  Produced by uor-addr's JSON realization when available, with local
  fallback for inputs that exceed typed-input bounds (>3968 bytes).
"""
import os
import json
import hashlib
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# UOR canonical addressing via uor-addr
try:
    from uor_addr import kappa, AddressError
    _UOR_ADDR_AVAILABLE = True
except ImportError:
    _UOR_ADDR_AVAILABLE = False


MEMORY_DIR = "memory"
MEMORY_INDEX = "memory_index.json"


@dataclass
class MemoryEntry:
    """A single stored compressed geometry."""
    content_hash: str                      # canonical identity
    model: str                             # which model produced this
    task_id: str                           # which task family/type
    task_family: str
    generation: int                        # compression cycle count
    budget_at_creation: int                # token budget when stored
    strategy_text: str                     # the compressed geometry itself
    timestamp: float = field(default_factory=time.time)

    # Compression metrics at time of storage
    operator_count: int = 0
    reuse_ratio: float = 0.0
    unique_verbs: list = field(default_factory=list)

    # Lineage — chain of prior hashes this compressed from
    lineage: list = field(default_factory=list)

    # Stability tracking
    times_retrieved: int = 0
    last_retrieved: float = 0.0


def _local_fallback_hash(text: str) -> str:
    """Local SHA-256 fallback when uor-addr can't handle the input.

    Produces the same sha256:<64-hex> format as uor-addr κ-labels
    so all downstream code can treat addresses uniformly.
    """
    normalized = " ".join(text.split())
    digest = hashlib.sha256(normalized.encode()).hexdigest()
    return f"sha256:{digest}"


def compute_hash(text: str) -> str:
    """Content-derived canonical address via UOR κ-label.

    Wraps the strategy text in a canonical JSON envelope and passes it
    through uor-addr's JSON realization (JCS + NFC + SHA-256).  Falls
    back to local SHA-256 if uor-addr is unavailable or the input
    exceeds typed-input bounds (>3968 bytes post-canonicalization).

    Returns a 71-byte string: ``sha256:<64-hex-digits>``.
    """
    if not _UOR_ADDR_AVAILABLE:
        return _local_fallback_hash(text)

    # Canonical JSON envelope for the strategy text
    normalized = " ".join(text.split())
    envelope = json.dumps(
        {"@type": "cdce:CompressedGeometry", "content": normalized},
        sort_keys=True,
        separators=(",", ":"),
    )
    try:
        return kappa.json_address(envelope.encode("utf-8"))
    except AddressError:
        # Input too large for typed-input bounds — fall back
        return _local_fallback_hash(text)


def _addr_to_filename(addr: str) -> str:
    """Convert a κ-label to a safe filename.

    ``sha256:abcdef...`` → ``sha256_abcdef...`` (colon is not
    filesystem-safe on Windows).  Also handles legacy 20-char
    hex hashes that predate the UOR integration.
    """
    return addr.replace(":", "_")


def _filename_to_addr(name: str) -> str:
    """Reverse of _addr_to_filename.

    ``sha256_abcdef...`` → ``sha256:abcdef...``
    Legacy 20-char hex filenames are returned as-is.
    """
    if name.startswith("sha256_"):
        return "sha256:" + name[7:]
    return name


class MemoryStore:
    """
    Persistent memory for compressed geometric structures.

    Over multiple experiment cycles, this store accumulates
    increasingly compressed strategies. The convergence of
    these structures IS the CDCE signal — if different models
    on different tasks produce strategies with the same hash,
    that's compression convergence in action.
    """

    def __init__(self, memory_dir: str = MEMORY_DIR):
        self.memory_dir = Path(memory_dir)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.memory_dir / MEMORY_INDEX
        self.index = self._load_index()

    def _load_index(self) -> dict:
        """Load or create the memory index."""
        if self.index_path.exists():
            with open(self.index_path) as f:
                return json.load(f)
        return {
            "created": time.time(),
            "total_entries": 0,
            "total_generations": 0,
            "entries": {},           # hash -> metadata
            "by_model": {},          # model -> [hashes]
            "by_task": {},           # task_id -> [hashes]
            "convergence_log": [],   # cross-model hash collisions over time
        }

    def _save_index(self):
        """Persist the index to disk."""
        with open(self.index_path, "w") as f:
            json.dump(self.index, f, indent=2)

    def store(self, entry: MemoryEntry) -> str:
        """
        Store a compressed strategy. Returns the content hash.

        If the exact same strategy (by content hash) already exists,
        we note the convergence but don't duplicate.
        """
        h = entry.content_hash

        # Store the full strategy text
        entry_path = self.memory_dir / f"{_addr_to_filename(h)}.json"
        entry_dict = asdict(entry)
        with open(entry_path, "w") as f:
            json.dump(entry_dict, f, indent=2)

        # Check for convergence — did a different model or task
        # arrive at the same hash?
        is_collision = False
        if h in self.index["entries"]:
            existing = self.index["entries"][h]
            if (existing.get("model") != entry.model or
                    existing.get("task_id") != entry.task_id):
                is_collision = True
                self.index["convergence_log"].append({
                    "timestamp": time.time(),
                    "hash": h,
                    "models": [existing.get("model"), entry.model],
                    "tasks": [existing.get("task_id"), entry.task_id],
                    "generation": entry.generation,
                    "type": "cross_convergence",
                })
                print(f"    ★ CONVERGENCE: {entry.model}/{entry.task_id} "
                      f"→ same hash as {existing.get('model')}/{existing.get('task_id')}")

        # Update index
        self.index["entries"][h] = {
            "model": entry.model,
            "task_id": entry.task_id,
            "task_family": entry.task_family,
            "generation": entry.generation,
            "budget": entry.budget_at_creation,
            "operator_count": entry.operator_count,
            "timestamp": entry.timestamp,
            "lineage_length": len(entry.lineage),
        }

        # Track by model
        self.index["by_model"].setdefault(entry.model, [])
        if h not in self.index["by_model"][entry.model]:
            self.index["by_model"][entry.model].append(h)

        # Track by task
        self.index["by_task"].setdefault(entry.task_id, [])
        if h not in self.index["by_task"][entry.task_id]:
            self.index["by_task"][entry.task_id].append(h)

        self.index["total_entries"] = len(self.index["entries"])
        self._save_index()

        return h

    def retrieve(self, model: str, task_id: str) -> Optional[MemoryEntry]:
        """
        Retrieve the most recent compressed strategy for a model/task pair.

        This is what gets fed back to the agent as prior geometry.
        """
        task_hashes = self.index.get("by_task", {}).get(task_id, [])
        if not task_hashes:
            return None

        # Find entries for this model and task, sorted by generation
        candidates = []
        for h in task_hashes:
            entry_path = self.memory_dir / f"{_addr_to_filename(h)}.json"
            if entry_path.exists():
                with open(entry_path) as f:
                    data = json.load(f)
                if data.get("model") == model:
                    candidates.append(data)

        if not candidates:
            # Fall back: return ANY model's strategy for this task
            # This enables cross-model memory transfer
            for h in task_hashes:
                entry_path = self.memory_dir / f"{_addr_to_filename(h)}.json"
                if entry_path.exists():
                    with open(entry_path) as f:
                        data = json.load(f)
                    candidates.append(data)

        if not candidates:
            return None

        # Return highest generation (most compressed)
        best = max(candidates, key=lambda x: x.get("generation", 0))

        # Update retrieval stats
        h = best["content_hash"]
        if h in self.index["entries"]:
            self.index["entries"][h]["times_retrieved"] = (
                self.index["entries"][h].get("times_retrieved", 0) + 1
            )
        self._save_index()

        return MemoryEntry(**{
            k: best[k] for k in MemoryEntry.__dataclass_fields__
            if k in best
        })

    def retrieve_cross_model(self, task_id: str) -> list[MemoryEntry]:
        """
        Retrieve ALL models' strategies for a task.
        Used for convergence analysis.
        """
        task_hashes = self.index.get("by_task", {}).get(task_id, [])
        entries = []
        for h in task_hashes:
            entry_path = self.memory_dir / f"{_addr_to_filename(h)}.json"
            if entry_path.exists():
                with open(entry_path) as f:
                    data = json.load(f)
                entries.append(MemoryEntry(**{
                    k: data[k] for k in MemoryEntry.__dataclass_fields__
                    if k in data
                }))
        return entries

    def get_generation(self, model: str, task_id: str) -> int:
        """Get the current generation count for a model/task pair."""
        prior = self.retrieve(model, task_id)
        if prior:
            return prior.generation + 1
        return 1

    def get_lineage(self, content_hash: str) -> list[str]:
        """
        Trace the full compression lineage of a strategy.
        Returns chain of hashes from oldest ancestor to current.
        """
        entry_path = self.memory_dir / f"{_addr_to_filename(content_hash)}.json"
        if not entry_path.exists():
            return [content_hash]

        with open(entry_path) as f:
            data = json.load(f)

        lineage = data.get("lineage", [])
        return lineage + [content_hash]

    def convergence_report(self) -> dict:
        """
        Generate a report on convergence patterns in memory.

        Key signals:
        - Hash collisions across models (same geometry, different substrate)
        - Hash collisions across tasks (same geometry, different domain)
        - Lineage depth (how many generations of compression)
        - Operator count trends across generations
        """
        report = {
            "total_stored": self.index["total_entries"],
            "unique_hashes": len(self.index["entries"]),
            "models": list(self.index.get("by_model", {}).keys()),
            "tasks": list(self.index.get("by_task", {}).keys()),
            "convergence_events": len(self.index.get("convergence_log", [])),
            "convergence_log": self.index.get("convergence_log", []),
        }

        # Compute cross-model hash overlap
        model_hashes = self.index.get("by_model", {})
        models = list(model_hashes.keys())
        cross_model = {}
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                m1, m2 = models[i], models[j]
                set1 = set(model_hashes[m1])
                set2 = set(model_hashes[m2])
                shared = set1 & set2
                if shared:
                    cross_model[f"{m1}_x_{m2}"] = {
                        "shared_hashes": list(shared),
                        "count": len(shared),
                    }
        report["cross_model_convergence"] = cross_model

        # Operator count trajectory across generations
        gen_ops = {}
        for h, meta in self.index.get("entries", {}).items():
            gen = meta.get("generation", 0)
            ops = meta.get("operator_count", 0)
            gen_ops.setdefault(gen, []).append(ops)

        report["operator_by_generation"] = {
            gen: {
                "mean": sum(ops) / len(ops),
                "min": min(ops),
                "max": max(ops),
                "count": len(ops),
            }
            for gen, ops in sorted(gen_ops.items())
        }

        return report

    def print_report(self):
        """Print a human-readable convergence report."""
        r = self.convergence_report()
        print("\n" + "=" * 60)
        print("MEMORY CONVERGENCE REPORT")
        print("=" * 60)
        print(f"  Total stored geometries: {r['total_stored']}")
        print(f"  Unique hashes: {r['unique_hashes']}")
        print(f"  Models: {', '.join(r['models'])}")
        print(f"  Convergence events: {r['convergence_events']}")

        if r["cross_model_convergence"]:
            print(f"\n  Cross-Model Hash Collisions (same geometry, different substrate):")
            for pair, data in r["cross_model_convergence"].items():
                print(f"    {pair}: {data['count']} shared structures")

        if r["operator_by_generation"]:
            print(f"\n  Operator Count by Generation:")
            print(f"  {'Gen':>5} | {'Mean Ops':>9} | {'Min':>5} | {'Max':>5} | {'N':>4}")
            print(f"  {'─' * 40}")
            for gen, stats in r["operator_by_generation"].items():
                print(f"  {gen:>5} | {stats['mean']:>9.1f} | "
                      f"{stats['min']:>5} | {stats['max']:>5} | {stats['count']:>4}")

        if r["convergence_events"] > 0:
            print(f"\n  ★ CONVERGENCE EVENTS:")
            for event in r["convergence_log"][-10:]:
                print(f"    Gen {event.get('generation')}: "
                      f"{event['models'][0]} ↔ {event['models'][1]} "
                      f"on {event['tasks'][0]} / {event['tasks'][1]}")

        print("=" * 60)


# ─────────────────────────────────────────────
# UOR Integration via uor-addr
# ─────────────────────────────────────────────

class UORBridge:
    """
    Bridge to UOR canonical content-addressing via the uor-addr crate.

    When uor-addr is installed, provides:
    - Deterministic κ-label generation (sha256:<64-hex>) via JCS+NFC+SHA-256
    - TC-05 replay verification (verify without re-hashing)
    - Content fingerprinting (raw 32-byte SHA-256 digest)

    Falls back gracefully if uor-addr is not installed.
    """

    def __init__(self):
        self.available = _UOR_ADDR_AVAILABLE
        if self.available:
            print("  UOR addr: uor-addr library loaded (canonical κ-labels active)")
        else:
            print("  UOR addr: uor-addr not installed (using local SHA-256 fallback)")

    def address(self, content: str, metadata: dict) -> Optional[str]:
        """Compute a canonical κ-label for a compressed geometry.

        Returns the 71-byte ``sha256:...`` label, or None if uor-addr
        is unavailable.
        """
        if not self.available:
            return None

        normalized = " ".join(content.split())
        envelope = json.dumps(
            {
                "@type": "cdce:CompressedGeometry",
                "content": normalized,
                "metadata": metadata,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        try:
            return kappa.json_address(envelope.encode("utf-8"))
        except AddressError:
            return None

    def verify(self, content: str, metadata: dict) -> Optional[bool]:
        """Verify a geometry's address via TC-05 replay.

        Mints a witness, then replays the derivation without
        re-invoking SHA-256.  Returns True if the replayed label
        matches the forward label, None if uor-addr is unavailable.
        """
        if not self.available:
            return None

        normalized = " ".join(content.split())
        envelope = json.dumps(
            {
                "@type": "cdce:CompressedGeometry",
                "content": normalized,
                "metadata": metadata,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        try:
            with kappa.json_address_with_witness(envelope.encode("utf-8")) as grounded:
                forward = grounded.kappa_label()
                replayed = grounded.verify()
                return forward == replayed
        except (AddressError, Exception):
            return None
