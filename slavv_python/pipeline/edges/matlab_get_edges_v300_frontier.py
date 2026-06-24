"""MATLAB port barrel: ``get_edges_V300.m`` frontier tracing entry surface.

Role: re-exports watershed discovery and frontier geometry helpers used when the
exact-route selects the MATLAB frontier tracer.

MATLAB source: ``external/Vectorization-Public/source/get_edges_V300.m``
"""

from __future__ import annotations

from slavv_python.pipeline.edges.candidate_payload import (
    _candidate_endpoint_pair_set,
    _candidate_incident_pair_counts,
)
from slavv_python.pipeline.edges.discovery import _use_matlab_frontier_tracer
from slavv_python.pipeline.edges.edge_types import (
    BoolArray,
    Float64Array,
    Int16Array,
    Int32Array,
    Int64Array,
)
from slavv_python.pipeline.edges.frontier_events import _build_frontier_candidate_lifecycle
from slavv_python.pipeline.edges.matlab_get_edges_v300_geometry import (
    _matlab_frontier_edge_budget,
    _matlab_frontier_offsets,
)
from slavv_python.pipeline.edges.matlab_get_edges_by_watershed import (
    _generate_edge_candidates_matlab_global_watershed,
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
    "_vertex_center_linear_lookup",
]
