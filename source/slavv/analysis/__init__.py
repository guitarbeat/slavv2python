"""
Analysis modules for SLAVV.

This subpackage contains:
- geometry: Geometric operations and network statistics
- ml_curator: Machine learning-based curation
"""

from __future__ import annotations

from .geometry import (
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
    register_strands,
    register_vector_sets,
    resample_vectors,
    smooth_edge_traces,
    subsample_vectors,
    transform_vector_set,
)
from .ml_curator import (
    AutomaticCurator,
    MLCurator,
    choose_edges,
    choose_vertices,
    extract_uncurated_info,
)

# InteractiveCurator requires PyQt5/PyVista/VTK — import directly when needed:
#   from slavv.visualization.interactive_curator import InteractiveCurator, run_curator

__all__ = [
    "AutomaticCurator",
    "MLCurator",
    "calculate_branching_angles",
    "calculate_image_stats",
    "calculate_network_statistics",
    "calculate_surface_area",
    "calculate_vessel_volume",
    "choose_edges",
    "choose_vertices",
    "crop_edges",
    "crop_vertices",
    "crop_vertices_by_mask",
    "evaluate_registration",
    "extract_uncurated_info",
    "get_edge_metric",
    "get_edges_for_vertex",
    "register_strands",
    "register_vector_sets",
    "resample_vectors",
    "smooth_edge_traces",
    "subsample_vectors",
    "transform_vector_set",
]
