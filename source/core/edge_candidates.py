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
from ._edge_candidates.frontier_resolution import (
    _build_frontier_lifecycle_event,
    _frontier_bifurcation_choice_from_reason,
    _frontier_claim_reassignment_from_reason,
    _frontier_parent_child_outcome_from_reason,
    _normalize_frontier_resolution_result,
    _prune_frontier_indices_beyond_found_vertices,
    _resolve_frontier_edge_connection,
    _resolve_frontier_edge_connection_details,
)
from ._edge_candidates.frontier_trace import _trace_origin_edges_matlab_frontier
from ._edge_candidates.generate import (
    _finalize_matlab_parity_candidates,
    _generate_edge_candidates,
    _generate_edge_candidates_matlab_frontier,
)
from ._edge_candidates.geodesic import _salvage_matlab_parity_candidates_with_local_geodesics
from ._edge_candidates.global_watershed import _generate_edge_candidates_matlab_global_watershed
from ._edge_candidates.lifecycle import _build_frontier_candidate_lifecycle
from ._edge_candidates.watershed import (
    _augment_matlab_frontier_candidates_with_watershed_contacts,
    _supplement_matlab_frontier_candidates_with_watershed_joins,
)
from ._edge_candidates.watershed_support import (
    _best_watershed_contact_coords,
    _build_watershed_join_trace,
    _rasterize_trace_segment,
)
from .edge_primitives import estimate_vessel_directions, generate_edge_directions, trace_edge

__all__ = [
    "BoolArray",
    "Float32Array",
    "Float64Array",
    "Int16Array",
    "Int32Array",
    "Int64Array",
    "_append_candidate_unit",
    "_augment_matlab_frontier_candidates_with_watershed_contacts",
    "_best_watershed_contact_coords",
    "_build_edge_candidate_audit",
    "_build_frontier_candidate_lifecycle",
    "_build_frontier_lifecycle_event",
    "_build_watershed_join_trace",
    "_candidate_endpoint_pair_set",
    "_candidate_incident_pair_counts",
    "_coord_to_matlab_linear_index",
    "_finalize_matlab_parity_candidates",
    "_frontier_bifurcation_choice_from_reason",
    "_frontier_claim_reassignment_from_reason",
    "_frontier_parent_child_outcome_from_reason",
    "_generate_edge_candidates",
    "_generate_edge_candidates_matlab_frontier",
    "_generate_edge_candidates_matlab_global_watershed",
    "_matlab_frontier_edge_budget",
    "_matlab_frontier_offsets",
    "_matlab_linear_index_to_coord",
    "_normalize_candidate_connection_sources",
    "_normalize_candidate_count_map",
    "_normalize_candidate_origin_counts",
    "_normalize_frontier_resolution_result",
    "_path_coords_from_linear_indices",
    "_path_max_energy_from_linear_indices",
    "_prune_frontier_indices_beyond_found_vertices",
    "_rasterize_trace_segment",
    "_resolve_frontier_edge_connection",
    "_resolve_frontier_edge_connection_details",
    "_salvage_matlab_parity_candidates_with_local_geodesics",
    "_supplement_matlab_frontier_candidates_with_watershed_joins",
    "_trace_local_geodesic_between_vertices",
    "_trace_origin_edges_matlab_frontier",
    "_use_matlab_frontier_tracer",
    "_vertex_center_linear_lookup",
    "estimate_vessel_directions",
    "generate_edge_directions",
    "trace_edge",
]

