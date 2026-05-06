"""Preferred internal name for MATLAB-style frontier tracing."""

from __future__ import annotations

from .._edge_candidates_internal_internal.common import (
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
from .._edge_candidates_internal_internal.global_watershed import _generate_edge_candidates_matlab_global_watershed
from .._edge_candidates_internal_internal.lifecycle import _build_frontier_candidate_lifecycle

__all__ = [
    "BoolArray",
    "Float32Array",
    "Float64Array",
    "Int16Array",
    "Int32Array",
    "Int64Array",
    "_build_frontier_candidate_lifecycle",
    "_candidate_endpoint_pair_set",
    "_candidate_incident_pair_counts",
    "_coord_to_matlab_linear_index",
    "_generate_edge_candidates_matlab_global_watershed",
    "_matlab_frontier_edge_budget",
    "_matlab_frontier_offsets",
    "_matlab_linear_index_to_coord",
    "_path_coords_from_linear_indices",
    "_path_max_energy_from_linear_indices",
    "_trace_local_geodesic_between_vertices",
    "_use_matlab_frontier_tracer",
    "_vertex_center_linear_lookup",
]


