"""Edge candidate generation helpers for SLAVV."""

from __future__ import annotations

from .edge_candidates_internal.audit import (
    _build_edge_candidate_audit,
    _normalize_candidate_connection_sources,
    _normalize_candidate_origin_counts,
)
from .edge_candidates_internal.candidate_manifest import (
    _append_candidate_unit,
)
from .edge_candidates_internal.common import (
    _coord_to_matlab_linear_index,
    _matlab_linear_index_to_coord,
    _path_coords_from_linear_indices,
    _path_max_energy_from_linear_indices,
    _trace_local_geodesic_between_vertices,
    _use_matlab_frontier_tracer,
    _vertex_center_linear_lookup,
)
from .edge_candidates_internal.generate import (
    _finalize_matlab_parity_candidates,
    _generate_edge_candidates,
    _generate_edge_candidates_matlab_frontier,
)
from .edge_candidates_internal.lifecycle import (
    _build_frontier_candidate_lifecycle,
)
from .edges_internal.edge_tracing import trace_edge
from .edges_internal.trace_directions import (
    estimate_vessel_directions,
    generate_edge_directions,
)

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
