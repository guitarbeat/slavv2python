"""
Analysis modules for source.

This subpackage contains:
- geometry: Geometric operations and network statistics
- ml_curator: Machine learning-based curation
"""

from __future__ import annotations

from .automatic_curator import AutomaticCurator
from .curation_heuristics import choose_edges, choose_vertices, extract_uncurated_info
from .drews_curator import DrewsCurator
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
    MLCurator,
)

# Interactive curation is optional GUI code â€” import directly when needed:
#   from source.visualization.interactive_curator import InteractiveCurator, run_curator
#   from source.visualization.napari_curator import run_curator_napari

__all__ = [
    "AutomaticCurator",
    "DrewsCurator",
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



