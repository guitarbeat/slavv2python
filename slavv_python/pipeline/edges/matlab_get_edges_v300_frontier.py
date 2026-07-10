"""MATLAB port barrel: ``get_edges_V300.m`` helpers for Watershed Discovery.

Re-exports geometry/indexing helpers used by Exact Route Watershed Discovery
(``WatershedDiscovery`` → ``generate_watershed_candidates``).

MATLAB source: ``external/Vectorization-Public/source/get_edges_V300.m``
"""

from __future__ import annotations

from slavv_python.pipeline.edges.candidate_manifest import (
    endpoint_pairs_from_connections as _candidate_endpoint_pair_set,
    incident_pair_counts as _candidate_incident_pair_counts,
)
from slavv_python.pipeline.edges.discovery import (
    _use_matlab_frontier_tracer,
    _use_watershed_discovery,
)
from slavv_python.pipeline.edges.edge_types import (
    BoolArray,
    Float64Array,
    Int16Array,
    Int32Array,
    Int64Array,
)
from slavv_python.pipeline.edges.frontier_events import _build_frontier_candidate_lifecycle
from slavv_python.pipeline.edges.matlab_get_edges_by_watershed import (
    _generate_edge_candidates_matlab_global_watershed,
)
from slavv_python.pipeline.edges.matlab_get_edges_v300_geometry import (
    _matlab_frontier_edge_budget,
    _matlab_frontier_offsets,
)
from slavv_python.pipeline.edges.matlab_indexing import (
    _coord_to_matlab_linear_index,
    _matlab_linear_index_to_coord,
    _path_coords_from_linear_indices,
    _path_max_energy_from_linear_indices,
    _trace_local_geodesic_between_vertices,
    _vertex_center_linear_lookup,
)

__all__ = [
    "BoolArray",
    "Float64Array",
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
    "_use_watershed_discovery",
    "_vertex_center_linear_lookup",
]
