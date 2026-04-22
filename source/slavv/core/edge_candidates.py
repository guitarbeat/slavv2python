"""Edge candidate generation helpers for SLAVV."""

from __future__ import annotations

from ._edge_candidates.audit import (
    _build_edge_candidate_audit,
    _normalize_candidate_connection_sources,
    _normalize_candidate_origin_counts,
)
from ._edge_candidates.candidate_manifest import _append_candidate_unit
from ._edge_candidates.common import (
    BoolArray,
    Float32Array,
    Float64Array,
    Int16Array,
    Int32Array,
    Int64Array,
)
from ._edge_candidates.generate import _generate_edge_candidates
from .edge_primitives import estimate_vessel_directions, generate_edge_directions, trace_edge

__all__ = [
    "BoolArray",
    "Float32Array",
    "Float64Array",
    "Int16Array",
    "Int32Array",
    "Int64Array",
    "_append_candidate_unit",
    "_build_edge_candidate_audit",
    "_generate_edge_candidates",
    "_normalize_candidate_connection_sources",
    "_normalize_candidate_origin_counts",
    "estimate_vessel_directions",
    "generate_edge_directions",
    "trace_edge",
]
