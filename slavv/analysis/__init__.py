"""
Analysis modules for SLAVV.

This subpackage contains:
- geometry: Geometric operations and network statistics
- ml_curator: Machine learning-based curation
"""
from .geometry import (
    calculate_branching_angles,
    calculate_network_statistics,
    calculate_surface_area,
    calculate_vessel_volume,
    crop_vertices,
    crop_edges,
    crop_vertices_by_mask,
    get_edges_for_vertex,
    get_edge_metric,
    resample_vectors,
    smooth_edge_traces,
    transform_vector_set,
    subsample_vectors,
    register_vector_sets,
    register_strands,
)
from .ml_curator import MLCurator, AutomaticCurator

__all__ = [
    "calculate_branching_angles",
    "calculate_network_statistics",
    "calculate_surface_area",
    "calculate_vessel_volume",
    "crop_vertices",
    "crop_edges",
    "crop_vertices_by_mask",
    "get_edges_for_vertex",
    "get_edge_metric",
    "resample_vectors",
    "smooth_edge_traces",
    "transform_vector_set",
    "subsample_vectors",
    "register_vector_sets",
    "register_strands",
    "MLCurator",
    "AutomaticCurator",
]

