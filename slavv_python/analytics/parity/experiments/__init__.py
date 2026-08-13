"""Parity Experiment: cheap-first same-class compare, not Certification.

Exact Proof Coordinator owns prove/capture. Parity Preflight owns launch
safety. This package owns artifact class, proof-record pairing, and the
cheap-loop gate.
"""

from __future__ import annotations

from slavv_python.analytics.parity.experiments.artifact_class import (
    ArtifactClass,
    ArtifactClassError,
    CoverageCompare,
    PairSetCompare,
    classify_edge_artifact,
    compare_same_class_pair_sets,
    coverage_of_finals_by_raw,
)
from slavv_python.analytics.parity.experiments.cost import (
    CheapLoopError,
    ExperimentCost,
    HypothesisKind,
    require_cheap_loop,
)
from slavv_python.analytics.parity.experiments.load import EdgeArtifact, load_edge_artifact
from slavv_python.analytics.parity.experiments.proof_record import (
    ProofRecord,
    ProofRecordError,
    load_proof_record,
    require_evaluated_adr0012,
)

__all__ = [
    "ArtifactClass",
    "ArtifactClassError",
    "CheapLoopError",
    "CoverageCompare",
    "EdgeArtifact",
    "ExperimentCost",
    "HypothesisKind",
    "PairSetCompare",
    "ProofRecord",
    "ProofRecordError",
    "classify_edge_artifact",
    "compare_same_class_pair_sets",
    "coverage_of_finals_by_raw",
    "load_edge_artifact",
    "load_proof_record",
    "require_cheap_loop",
    "require_evaluated_adr0012",
]
