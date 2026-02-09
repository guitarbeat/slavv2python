"""
SLAVV - Segmentation-Less, Automated, Vascular Vectorization

A Python implementation of the SLAVV algorithm for extracting and analyzing
vascular networks from 3D microscopy images.
"""

from .core import SLAVVProcessor
from .utils import (
    preprocess_image,
    validate_parameters,
    get_chunking_lattice,
)
from .analysis import (
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

from .analysis import MLCurator, AutomaticCurator
try:
    from .visualization import NetworkVisualizer
except ImportError:
    import logging
    logging.getLogger(__name__).warning("Visualization module unavailable (missing dependencies, likely plotly).")
    NetworkVisualizer = None

from .io import (
    load_tiff_volume,
    load_network_from_mat,
    load_network_from_casx,
    load_network_from_vmv,
    load_network_from_csv,
    load_network_from_json,
    save_network_to_csv,
    save_network_to_json,
    dicom_to_tiff,
)

__version__ = "0.1.0"
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
    "MLCurator",
    "AutomaticCurator",
    "NetworkVisualizer",
    "load_tiff_volume",
    "load_network_from_mat",
    "load_network_from_casx",
    "load_network_from_vmv",
    "load_network_from_csv",
    "load_network_from_json",
    "save_network_to_csv",
    "save_network_to_json",
    "dicom_to_tiff",
]
