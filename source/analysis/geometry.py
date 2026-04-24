"""
Geometric operations and network statistics for SLAVV.
Handles registration, spatial metrics, and statistical analysis of the vascular network.
"""

from __future__ import annotations

from ._geometry import (
    calculate_branching_angles,
    calculate_image_stats,
    calculate_network_statistics,
    calculate_surface_area,
    calculate_vessel_volume,
    crop_edges,
    crop_vertices,
    crop_vertices_by_mask,
    evaluate_registration,
    get_edge_metric,
    get_edges_for_vertex,
    icp_register_rigid,
    register_strands,
    register_vector_sets,
    resample_vectors,
    smooth_edge_traces,
    subsample_vectors,
    transform_vector_set,
)

__all__ = [
    "calculate_branching_angles",
    "calculate_image_stats",
    "calculate_network_statistics",
    "calculate_surface_area",
    "calculate_vessel_volume",
    "crop_edges",
    "crop_vertices",
    "crop_vertices_by_mask",
    "evaluate_registration",
    "get_edge_metric",
    "get_edges_for_vertex",
    "icp_register_rigid",
    "register_strands",
    "register_vector_sets",
    "resample_vectors",
    "smooth_edge_traces",
    "subsample_vectors",
    "transform_vector_set",
]
