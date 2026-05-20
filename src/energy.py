"""
CDCE Compression Harness — Energy Metrics

Thermodynamic layer for tracking compression energetics.
Maps token consumption to free-energy framework:

  Work (W)        = total tokens consumed
  Quality (Q)     = solution quality score [0-1]  
  Efficiency (η)  = Q / W — useful work per token
  Dissipation (D) = W * (1 - η) — wasted tokens
  Free Energy (F) = Q - λW — what the system minimizes

The phase transition shows up as an efficiency peak:
the system gets MORE efficient under pressure until
a critical threshold, then collapses.

This maps directly to:
  - HJB: minimizing cumulative cost (W) subject to value (Q)
  - Navier-Stokes: minimizing energy dissipation (D)
  - Jarzynski: non-equilibrium work relates to free energy
"""
import math
from dataclasses import dataclass, field


# Cost per token (approximate, USD) for energy accounting
TOKEN_COSTS = {
    "claude": {"input": 3.0 / 1_000_000, "output": 15.0 / 1_000_000},
    "haiku": {"input": 0.80 / 1_000_000, "output": 4.0 / 1_000_000},
    "gemini_flash": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
    "gemini_pro": {"input": 1.25 / 1_000_000, "output": 10.0 / 1_000_000},
    "gpt4": {"input": 2.50 / 1_000_000, "output": 10.0 / 1_000_000},
}

# Default cost if model not in table
DEFAULT_COST = {"input": 1.0 / 1_000_000, "output": 5.0 / 1_000_000}


@dataclass
class EnergyMetrics:
    """Thermodynamic metrics for a single compression run."""
    model: str
    task_id: str
    budget: int

    # Raw measurements
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    dollar_cost: float = 0.0

    # Quality score [0-1]
    quality: float = 0.0

    # Derived thermodynamic quantities
    work: float = 0.0              # total token expenditure (normalized)
    useful_work: float = 0.0       # quality * budget (value extracted)
    dissipation: float = 0.0       # work - useful_work
    efficiency: float = 0.0        # useful_work / work
    free_energy: float = 0.0       # quality - lambda * work
    tokens_per_quality: float = 0.0  # cost of each unit of quality

    # Compression-specific
    compression_ratio: float = 0.0   # output_tokens / budget
    info_density: float = 0.0        # quality / output_tokens


def estimate_quality(operator_count, reuse_ratio, word_count, budget,
                     step_count=1, has_answer=True):
    """
    Estimate solution quality [0-1] from compression metrics.
    
    Quality = did the agent produce a coherent, structured response
    that actually addresses the task?
    
    Components:
    - Produced output at all (0 or 1)
    - Structural coherence (has steps, has operators)
    - Compression efficiency (not just truncated)
    - Reuse (higher reuse = more compressed = better)
    """
    if word_count == 0:
        return 0.0

    scores = []

    # 1. Output exists and has substance
    substance = min(1.0, word_count / (budget * 0.3))
    scores.append(substance)

    # 2. Structural coherence — uses reasoning operations
    if operator_count > 0:
        coherence = min(1.0, operator_count / 5.0)
    else:
        coherence = 0.2  # produced words but no detected reasoning
    scores.append(coherence)

    # 3. Step structure
    step_score = min(1.0, step_count / 3.0)
    scores.append(step_score)

    # 4. Reuse bonus — higher reuse means more compressed
    reuse_score = 0.5 + (reuse_ratio * 0.5)
    scores.append(reuse_score)

    return sum(scores) / len(scores)


def compute_energy(
    model: str,
    task_id: str,
    budget: int,
    input_tokens: int,
    output_tokens: int,
    latency_ms: float,
    operator_count: int,
    reuse_ratio: float,
    word_count: int,
    step_count: int = 1,
    lambda_cost: float = 0.001,
) -> EnergyMetrics:
    """
    Compute thermodynamic energy metrics for a compression run.
    
    lambda_cost: trade-off parameter between quality and work.
                 Higher = penalize token consumption more.
    """
    em = EnergyMetrics(
        model=model,
        task_id=task_id,
        budget=budget,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        latency_ms=latency_ms,
    )

    # Dollar cost
    costs = TOKEN_COSTS.get(model, DEFAULT_COST)
    em.dollar_cost = (
        input_tokens * costs["input"]
        + output_tokens * costs["output"]
    )

    # Quality estimate
    em.quality = estimate_quality(
        operator_count, reuse_ratio, word_count, budget, step_count
    )

    # Thermodynamic quantities
    em.work = em.total_tokens / 1000.0  # normalize to kilotokens
    em.useful_work = em.quality * (budget / 1000.0)
    em.dissipation = max(0, em.work - em.useful_work)
    em.efficiency = em.useful_work / em.work if em.work > 0 else 0
    em.free_energy = em.quality - lambda_cost * em.total_tokens

    # Info density
    em.compression_ratio = output_tokens / budget if budget > 0 else 0
    em.tokens_per_quality = (
        em.total_tokens / em.quality if em.quality > 0 else float('inf')
    )
    em.info_density = (
        em.quality / (output_tokens / 1000.0)
        if output_tokens > 0 else 0
    )

    return em


def energy_to_dict(em: EnergyMetrics) -> dict:
    """Serialize for JSON storage."""
    return {
        "model": em.model,
        "task_id": em.task_id,
        "budget": em.budget,
        "input_tokens": em.input_tokens,
        "output_tokens": em.output_tokens,
        "total_tokens": em.total_tokens,
        "latency_ms": round(em.latency_ms, 1),
        "dollar_cost": round(em.dollar_cost, 6),
        "quality": round(em.quality, 4),
        "work": round(em.work, 4),
        "useful_work": round(em.useful_work, 4),
        "dissipation": round(em.dissipation, 4),
        "efficiency": round(em.efficiency, 4),
        "free_energy": round(em.free_energy, 4),
        "compression_ratio": round(em.compression_ratio, 4),
        "info_density": round(em.info_density, 4),
        "tokens_per_quality": round(em.tokens_per_quality, 1),
    }


def compute_phase_transition(energy_by_budget: dict) -> dict:
    """
    Detect the compression phase transition.
    
    Look for the budget level where efficiency peaks
    before collapsing. That's the critical threshold —
    the Reynolds number analogue.
    """
    budgets = sorted(energy_by_budget.keys(), reverse=True)
    if len(budgets) < 3:
        return {"detected": False}

    efficiencies = [energy_by_budget[b]["mean_efficiency"] for b in budgets]

    # Find peak efficiency
    peak_idx = efficiencies.index(max(efficiencies))
    peak_budget = budgets[peak_idx]

    # Check if there's a drop after the peak
    if peak_idx < len(efficiencies) - 1:
        post_peak = efficiencies[peak_idx + 1:]
        drop = efficiencies[peak_idx] - min(post_peak)
        has_transition = drop > 0.05
    else:
        has_transition = False
        drop = 0

    return {
        "detected": has_transition,
        "critical_budget": peak_budget,
        "peak_efficiency": round(efficiencies[peak_idx], 4),
        "efficiency_drop": round(drop, 4),
        "efficiency_curve": {
            b: round(e, 4) for b, e in zip(budgets, efficiencies)
        },
    }
