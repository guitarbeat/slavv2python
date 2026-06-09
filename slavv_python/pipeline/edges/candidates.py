"""Edge candidate generation helpers for SLAVV."""

from __future__ import annotations

from slavv_python.pipeline.edges.frontier_events import (
    _build_frontier_candidate_lifecycle,
)
from slavv_python.pipeline.edges.audit import (
    _build_edge_candidate_audit,
    _normalize_candidate_connection_sources,
    _normalize_candidate_origin_counts,
)
from slavv_python.pipeline.edges.candidate_generation import (
    _finalize_matlab_parity_candidates,
    _generate_edge_candidates,
    _generate_edge_candidates_matlab_frontier,
)
from slavv_python.pipeline.edges.candidate_manifest import (
    _append_candidate_unit,
)
from slavv_python.pipeline.edges.discovery import _use_matlab_frontier_tracer
from slavv_python.pipeline.edges.matlab_indexing import (
    _coord_to_matlab_linear_index,
    _matlab_linear_index_to_coord,
    _path_coords_from_linear_indices,
    _path_max_energy_from_linear_indices,
    _trace_local_geodesic_between_vertices,
    _vertex_center_linear_lookup,
)
from slavv_python.pipeline.edges.trace_directions import (
    estimate_vessel_directions,
    generate_edge_directions,
)
from slavv_python.pipeline.edges.tracing import trace_edge

__all__ = [
    "_append_candidate_unit",
    "_build_edge_candidate_audit",
    "_build_frontier_candidate_lifecycle",
    "_coord_to_matlab_linear_index",
    "_finalize_matlab_parity_candidates",
    "_generate_edge_candidates",
    "_generate_edge_candidates_matlab_frontier",
    "_matlab_linear_index_to_coord",
    "_normalize_candidate_connection_sources",
    "_normalize_candidate_origin_counts",
    "_path_coords_from_linear_indices",
    "_path_max_energy_from_linear_indices",
    "_trace_local_geodesic_between_vertices",
    "_use_matlab_frontier_tracer",
    "_vertex_center_linear_lookup",
    "estimate_vessel_directions",
    "generate_edge_directions",
    "trace_edge",
]
