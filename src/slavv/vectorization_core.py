
"""
Deprecated: Use src.slavv.pipeline, src.slavv.geometry, src.slavv.tracing, or src.slavv.energy instead.
This module is kept for backward compatibility for scripts or tests that import directly from here.
"""
import warnings

# Use absolute imports to be robust
from src.slavv.pipeline import SLAVVProcessor
from src.slavv.utils import (
    preprocess_image,
    validate_parameters,
    get_chunking_lattice,
)
from src.slavv.geometry import (
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

warnings.warn(
    "src.slavv.vectorization_core is deprecated. Import from src.slavv.pipeline, src.slavv.geometry, etc. instead.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = [
    "SLAVVProcessor",
    "preprocess_image",
    "validate_parameters",
    "get_chunking_lattice",
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
]
