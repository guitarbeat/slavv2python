"""File I/O operations for SLAVV.

Submodules
----------
tiff        — TIFF / DICOM volume loading
network_io  — Network load/save (MAT, CASX, VMV, CSV, JSON)
exporters   — Pipeline result export and network partitioning
matlab_parser — MATLAB batch-folder parsing
matlab_bridge — Convert MATLAB batch output to Python checkpoints
"""
from .tiff import load_tiff_volume, dicom_to_tiff
from .network_io import (
    Network,
    MatNetwork,
    load_network_from_mat,
    load_network_from_casx,
    load_network_from_vmv,
    load_network_from_csv,
    load_network_from_json,
    save_network_to_csv,
    save_network_to_json,
)
from .exporters import export_pipeline_results, partition_network, parse_registration_file

__all__ = [
    # image loading
    "load_tiff_volume",
    "dicom_to_tiff",
    # network
    "Network",
    "MatNetwork",
    "load_network_from_mat",
    "load_network_from_casx",
    "load_network_from_vmv",
    "load_network_from_csv",
    "load_network_from_json",
    "save_network_to_csv",
    "save_network_to_json",
    # pipeline export
    "export_pipeline_results",
    "partition_network",
    "parse_registration_file",
]
