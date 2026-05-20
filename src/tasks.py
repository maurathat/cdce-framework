"""
CDCE Compression Harness — Task Generator

Three task families designed to share hidden structure:
- Optimization: minimize cost over constrained space
- Prediction: extend structure under constraint  
- Translation: structure-preserving maps between representations

Under sufficient compression, strategies for all three should converge.
"""
import random
import hashlib
import json

try:
    from uor_addr import kappa, AddressError
    _UOR_ADDR_AVAILABLE = True
except ImportError:
    _UOR_ADDR_AVAILABLE = False


def content_hash(obj: dict) -> str:
    """UOR canonical κ-label for task identity.

    Returns ``sha256:<64-hex>`` via uor-addr when available,
    otherwise falls back to local SHA-256.
    """
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    if _UOR_ADDR_AVAILABLE:
        try:
            return kappa.json_address(canonical.encode("utf-8"))
        except AddressError:
            pass
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    return f"sha256:{digest}"


# ─────────────────────────────────────────────
# FAMILY 1: OPTIMIZATION
# All are: minimize cost over combinatorial space
# ─────────────────────────────────────────────

OPTIMIZATION_TASKS = [
    {
        "id": "opt_routing",
        "name": "Route Planning",
        "prompt": (
            "You are a delivery planner. Find the shortest route visiting all 6 stops "
            "exactly once, starting and ending at the depot.\n\n"
            "Distances (miles):\n"
            "Depot→A:10, Depot→B:15, Depot→C:20, A→B:12, A→C:8, A→D:15,\n"
            "B→C:10, B→D:14, B→E:9, C→D:7, C→E:11, D→E:6,\n"
            "D→F:13, E→F:8, A→E:18, A→F:22, B→F:16, C→F:14\n\n"
            "Show your reasoning step by step. State your final route and total distance."
        ),
    },
    {
        "id": "opt_portfolio",
        "name": "Resource Allocation",
        "prompt": (
            "You have $100,000 to allocate across 5 projects. Each has an expected return "
            "and risk score (1-10). Minimize risk while achieving at least 12% total return.\n\n"
            "Project A: 15% return, risk 8\n"
            "Project B: 10% return, risk 3\n"
            "Project C: 20% return, risk 9\n"
            "Project D: 8% return, risk 2\n"
            "Project E: 12% return, risk 5\n\n"
            "Show your reasoning step by step. State your final allocation and explain why."
        ),
    },
    {
        "id": "opt_schedule",
        "name": "Scheduling",
        "prompt": (
            "Schedule 6 meetings into 4 time slots. Some people attend multiple meetings, "
            "creating conflicts. Minimize the number of conflicts.\n\n"
            "Meetings: M1(Alice,Bob), M2(Bob,Carol), M3(Carol,Dave), "
            "M4(Dave,Eve), M5(Eve,Alice), M6(Alice,Carol)\n"
            "Slots: 9am, 10am, 11am, 1pm\n\n"
            "Show your reasoning step by step. State your final schedule and conflict count."
        ),
    },
]


# ─────────────────────────────────────────────
# FAMILY 2: PREDICTION
# All are: extend structured pattern under constraint
# ─────────────────────────────────────────────

PREDICTION_TASKS = [
    {
        "id": "pred_math",
        "name": "Sequence Prediction",
        "prompt": (
            "Find the pattern and predict the next 3 values:\n\n"
            "2, 6, 14, 30, 62, ?\n\n"
            "Show your reasoning step by step. Explain the rule and give the next 3 values."
        ),
    },
    {
        "id": "pred_music",
        "name": "Musical Pattern",
        "prompt": (
            "A melody follows this pattern of intervals (in semitones):\n"
            "+4, -2, +5, -3, +4, -2, +5, -3, +4, -2, ?\n\n"
            "The starting note is C4 (middle C = MIDI 60).\n\n"
            "Show your reasoning step by step. Predict the next 4 intervals and "
            "the resulting note sequence."
        ),
    },
    {
        "id": "pred_code",
        "name": "Code Completion",
        "prompt": (
            "Complete this function. Identify the pattern and fill in the body:\n\n"
            "```\n"
            "def transform(items):\n"
            "    # Given: [1,2,3,4,5] -> [1,3,6,10,15]\n"
            "    # Given: [2,4,6] -> [2,6,12]\n"
            "    # Given: [10,20] -> [10,30]\n"
            "    # Implement the general pattern:\n"
            "    pass\n"
            "```\n\n"
            "Show your reasoning step by step. Explain the pattern and write the function."
        ),
    },
]


# ─────────────────────────────────────────────
# FAMILY 3: TRANSLATION
# All are: structure-preserving maps between representations
# ─────────────────────────────────────────────

TRANSLATION_TASKS = [
    {
        "id": "trans_nl_code",
        "name": "Natural Language → Code",
        "prompt": (
            "Convert this description to a Python function:\n\n"
            "'Given a list of numbers, return a new list where each element is the "
            "average of itself and all elements that came before it in the original list.'\n\n"
            "Show your reasoning step by step. Write the function and test it with [4, 8, 6, 2, 10]."
        ),
    },
    {
        "id": "trans_lang_lang",
        "name": "Python → JavaScript",
        "prompt": (
            "Convert this Python to equivalent JavaScript:\n\n"
            "```python\n"
            "def flatten_nested(data, prefix=''):\n"
            "    result = {}\n"
            "    for key, value in data.items():\n"
            "        new_key = f'{prefix}.{key}' if prefix else key\n"
            "        if isinstance(value, dict):\n"
            "            result.update(flatten_nested(value, new_key))\n"
            "        else:\n"
            "            result[new_key] = value\n"
            "    return result\n"
            "```\n\n"
            "Show your reasoning step by step. Preserve all behavior exactly."
        ),
    },
    {
        "id": "trans_math_nl",
        "name": "Math → Natural Language",
        "prompt": (
            "Explain this equation in plain English that a high school student would understand:\n\n"
            "∇·F = ∂Fx/∂x + ∂Fy/∂y + ∂Fz/∂z\n\n"
            "Show your reasoning step by step. Use at least one concrete physical analogy. "
            "Then verify your explanation captures the full meaning."
        ),
    },
]


ALL_TASKS = {
    "optimization": OPTIMIZATION_TASKS,
    "prediction": PREDICTION_TASKS,
    "translation": TRANSLATION_TASKS,
}


def get_tasks(family: str = None) -> list[dict]:
    """Get tasks, optionally filtered by family."""
    if family:
        return ALL_TASKS.get(family, [])
    tasks = []
    for family_tasks in ALL_TASKS.values():
        tasks.extend(family_tasks)
    return tasks


def build_compression_prompt(task: dict, budget: int, prior_strategy: str = None) -> str:
    """
    Build the prompt with compression constraints.
    
    If prior_strategy is provided, the agent must compress its own
    prior reasoning — this is what makes it recursive.
    """
    parts = []

    if prior_strategy:
        parts.append(
            f"PREVIOUS STRATEGY (from a higher budget run):\n"
            f"---\n{prior_strategy}\n---\n\n"
            f"You must solve the same type of problem below, but with a MUCH tighter "
            f"response budget. Compress your approach — find the essential operations, "
            f"drop everything redundant. Use fewer distinct steps and operations.\n\n"
        )

    parts.append(task["prompt"])

    parts.append(
        f"\n\n[CONSTRAINT: Your COMPLETE response must be under {budget} tokens. "
        f"Be maximally compressed. Every word must earn its place.]"
    )

    return "".join(parts)
