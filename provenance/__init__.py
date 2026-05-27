"""
provenance/ — experiment-adjacent provenance tooling for CDCE.

NOT part of the harness core (src/). This package records what a result was
computed against (prompt set, parser identity, model + params, library versions,
per-row inputs) as a canonical, content-addressable sidecar. It is identity /
reproducibility infrastructure only — it never computes or alters any statistic.

Dependency direction: experiment scripts import FROM provenance. provenance
imports nothing from src/ (kept self-contained on purpose: stdlib + lazy
numpy/pandas only), so it can be shared by both the transformers-world
activation scripts and the API-world 6.3/6.4 scripts without coupling them.

Usage:
    from provenance import build_provenance, write_sidecar
"""

from .run_provenance import (
    build_provenance,
    write_sidecar,
    verify_sidecar,
    canonical_bytes,
)

__all__ = [
    "build_provenance",
    "write_sidecar",
    "verify_sidecar",
    "canonical_bytes",
]
