"""
SLAVV - Segmentation-Less, Automated, Vascular Vectorization

A Python implementation of the SLAVV algorithm for extracting and analyzing
vascular networks from 3D microscopy images.
"""

from __future__ import annotations

from .analysis import (
    AutomaticCurator,
    MLCurator,
    calculate_branching_angles,
    calculate_network_statistics,
    calculate_surface_area,
    calculate_vessel_volume,
    crop_edges,
    crop_vertices,
    crop_vertices_by_mask,
    get_edge_metric,
    get_edges_for_vertex,
    register_strands,
    register_vector_sets,
    resample_vectors,
    smooth_edge_traces,
    subsample_vectors,
    transform_vector_set,
)
from .core import SlavvPipeline, SLAVVProcessor
from .utils import (
    get_chunking_lattice,
    preprocess_image,
    validate_parameters,
)

try:
    from .visualization import NetworkVisualizer
except ImportError:
    import logging

    logging.getLogger(__name__).warning(
        "Visualization module unavailable (missing dependencies, likely plotly)."
    )
    NetworkVisualizer = None

from .io import (
    dicom_to_tiff,
    load_network_from_casx,
    load_network_from_csv,
    load_network_from_json,
    load_network_from_mat,
    load_network_from_vmv,
    load_tiff_volume,
    save_network_to_csv,
    save_network_to_json,
)

__version__ = "0.1.0"
__all__ = [
    "AutomaticCurator",
    "MLCurator",
    "NetworkVisualizer",
    "SLAVVProcessor",
    "SlavvPipeline",
    "calculate_branching_angles",
    "calculate_network_statistics",
    "calculate_surface_area",
    "calculate_vessel_volume",
    "crop_edges",
    "crop_vertices",
    "crop_vertices_by_mask",
    "dicom_to_tiff",
    "get_chunking_lattice",
    "get_edge_metric",
    "get_edges_for_vertex",
    "load_network_from_casx",
    "load_network_from_csv",
    "load_network_from_json",
    "load_network_from_mat",
    "load_network_from_vmv",
    "load_tiff_volume",
    "preprocess_image",
    "register_strands",
    "register_vector_sets",
    "resample_vectors",
    "save_network_to_csv",
    "save_network_to_json",
    "smooth_edge_traces",
    "subsample_vectors",
    "transform_vector_set",
    "validate_parameters",
]
