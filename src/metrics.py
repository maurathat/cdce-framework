"""
CDCE Compression Harness — Metrics Extraction

The detection loop. Extracts compression signatures from LLM responses:
- Operator count (distinct reasoning operations)
- Vocabulary convergence (cross-family strategy similarity)
- Reuse ratio (repeated patterns vs novel steps)
- Composition order sensitivity (non-commutativity signal)
- Solution quality (correctness check)
"""
import re
import hashlib
import json
from collections import Counter
from dataclasses import dataclass, field


# ─────────────────────────────────────────────
# Reasoning operation vocabulary
# These are the "operators" we count in the agent's reasoning
# ─────────────────────────────────────────────

OPERATION_VERBS = {
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


@dataclass
class CompressionMetrics:
    """Metrics extracted from a single LLM response."""
    task_id: str
    task_family: str
    model: str
    budget: int

    # Core CDCE metrics
    operator_count: int = 0                # distinct reasoning operations used
    total_operations: int = 0              # total operation instances
    reuse_ratio: float = 0.0              # repeated / total operations
    unique_verbs: list = field(default_factory=list)  # which operations
    verb_distribution: dict = field(default_factory=dict)

    # Text metrics
    word_count: int = 0
    sentence_count: int = 0
    step_count: int = 0                   # explicit reasoning steps
    compression_ratio: float = 0.0        # output_tokens / budget

    # Content hash for canonical dedup
    content_hash: str = ""

    # Raw content for later analysis
    raw_content: str = ""


def extract_metrics(
    content: str,
    task_id: str,
    task_family: str,
    model: str,
    budget: int,
    output_tokens: int = 0,
) -> CompressionMetrics:
    """
    Extract compression metrics from an LLM response.
    This is the core of the detection loop.
    """
    metrics = CompressionMetrics(
        task_id=task_id,
        task_family=task_family,
        model=model,
        budget=budget,
        raw_content=content,
    )

    if content.startswith("[ERROR"):
        return metrics

    # Content hash (UOR-compatible canonical identity)
    metrics.content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]

    # Basic text metrics
    cleaned = content.lower()
    for ch in ['*', '#', '`', '>', '|', '-', '_', ':', '"', "'", '(', ')', '[', ']', '{', '}', ',', '.', '!', '?', ';', '\n', '\t']:
        cleaned = cleaned.replace(ch, ' ')
    words = cleaned.split()
    metrics.word_count = len(words)
    metrics.sentence_count = len(re.findall(r'[.!?]+', content))
    metrics.compression_ratio = output_tokens / budget if budget > 0 else 0

    # Count explicit reasoning steps (numbered lists, "Step N:", bullets)
    step_patterns = [
        r'(?:step\s+\d)',
        r'(?:^\s*\d+[\.\)]\s)',
        r'(?:^\s*[-•]\s)',
        r'(?:first|second|third|finally|next|then)',
    ]
    step_matches = 0
    for pattern in step_patterns:
        step_matches += len(re.findall(pattern, content, re.IGNORECASE | re.MULTILINE))
    metrics.step_count = max(1, step_matches)

    # ── OPERATOR COUNT (the key CDCE metric) ──
    # Find all reasoning operation verbs in the response
    word_set = set(words)
    # Also check for verb forms (computing, computed, computes, etc.)
    found_verbs = []
    verb_counts = Counter()

    for verb in OPERATION_VERBS:
        # Check exact match and common inflections
        forms = {verb, verb + "s", verb + "ed", verb + "ing", verb + "e",
                 verb + "es", verb + "d"}
        for form in forms:
            count = words.count(form)
            if count > 0:
                found_verbs.append(verb)
                verb_counts[verb] += count
                break

    metrics.unique_verbs = sorted(set(found_verbs))
    metrics.operator_count = len(metrics.unique_verbs)
    metrics.verb_distribution = dict(verb_counts)
    metrics.total_operations = sum(verb_counts.values())
    if metrics.total_operations >= 2:
        metrics.reuse_ratio = (
            1 - (metrics.operator_count / metrics.total_operations)
        )
    else:
        metrics.reuse_ratio = 0.0

    return metrics


def compute_convergence(
    metrics_list: list[CompressionMetrics],
) -> dict:
    """
    Compute cross-family convergence at a given compression level.
    
    If strategies for optimization, prediction, and translation
    are converging, their verb sets should increasingly overlap.
    """
    by_family = {}
    for m in metrics_list:
        by_family.setdefault(m.task_family, []).append(m)

    if len(by_family) < 2:
        return {"jaccard_mean": 0, "jaccard_pairs": {}}

    # Compute Jaccard similarity between verb sets of each family pair
    families = list(by_family.keys())
    jaccard_pairs = {}
    jaccard_values = []

    for i in range(len(families)):
        for j in range(i + 1, len(families)):
            f1, f2 = families[i], families[j]
            verbs_1 = set()
            verbs_2 = set()
            for m in by_family[f1]:
                verbs_1.update(m.unique_verbs)
            for m in by_family[f2]:
                verbs_2.update(m.unique_verbs)

            if not verbs_1 and not verbs_2:
                jacc = 0
            else:
                intersection = verbs_1 & verbs_2
                union = verbs_1 | verbs_2
                jacc = len(intersection) / len(union) if union else 0

            pair_key = f"{f1}_x_{f2}"
            jaccard_pairs[pair_key] = round(jacc, 4)
            jaccard_values.append(jacc)

    mean_jaccard = sum(jaccard_values) / len(jaccard_values) if jaccard_values else 0

    return {
        "jaccard_mean": round(mean_jaccard, 4),
        "jaccard_pairs": jaccard_pairs,
    }


def metrics_to_dict(m: CompressionMetrics) -> dict:
    """Serialize metrics for JSON storage."""
    return {
        "task_id": m.task_id,
        "task_family": m.task_family,
        "model": m.model,
        "budget": m.budget,
        "operator_count": m.operator_count,
        "total_operations": m.total_operations,
        "reuse_ratio": round(m.reuse_ratio, 4),
        "unique_verbs": m.unique_verbs,
        "word_count": m.word_count,
        "sentence_count": m.sentence_count,
        "step_count": m.step_count,
        "compression_ratio": round(m.compression_ratio, 4),
        "content_hash": m.content_hash,
    }
