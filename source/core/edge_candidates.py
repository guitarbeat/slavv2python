"""Edge candidate generation helpers for source."""

from __future__ import annotations

from ._edge_candidates.audit import (
    _build_edge_candidate_audit,
    _normalize_candidate_connection_sources,
    _normalize_candidate_count_map,
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
    _candidate_endpoint_pair_set,
    _candidate_incident_pair_counts,
    _coord_to_matlab_linear_index,
    _matlab_frontier_edge_budget,
    _matlab_frontier_offsets,
    _matlab_linear_index_to_coord,
    _path_coords_from_linear_indices,
    _path_max_energy_from_linear_indices,
    _trace_local_geodesic_between_vertices,
    _use_matlab_frontier_tracer,
    _vertex_center_linear_lookup,
)
from ._edge_candidates.generate import (
    _finalize_matlab_parity_candidates,
    _generate_edge_candidates,
    _generate_edge_candidates_matlab_frontier,
)
from ._edge_candidates.global_watershed import _generate_edge_candidates_matlab_global_watershed
from ._edge_candidates.lifecycle import _build_frontier_candidate_lifecycle
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
    "_build_frontier_candidate_lifecycle",
    "_candidate_endpoint_pair_set",
    "_candidate_incident_pair_counts",
    "_coord_to_matlab_linear_index",
    "_finalize_matlab_parity_candidates",
    "_generate_edge_candidates",
    "_generate_edge_candidates_matlab_frontier",
    "_generate_edge_candidates_matlab_global_watershed",
    "_matlab_frontier_edge_budget",
    "_matlab_frontier_offsets",
    "_matlab_linear_index_to_coord",
    "_normalize_candidate_connection_sources",
    "_normalize_candidate_count_map",
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
