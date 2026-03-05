"""File I/O operations for SLAVV.

Submodules
----------
tiff        — TIFF / DICOM volume loading
network_io  — Network load/save (MAT, CASX, VMV, CSV, JSON)
exporters   — Pipeline result export and network partitioning
matlab_parser — MATLAB batch-folder parsing
matlab_bridge — Convert MATLAB batch output to Python checkpoints
"""

from __future__ import annotations

from .exporters import export_pipeline_results, parse_registration_file, partition_network
from .network_io import (
    MatNetwork,
    Network,
    convert_casx_to_vmv,
    load_network_from_casx,
    load_network_from_csv,
    load_network_from_json,
    load_network_from_mat,
    load_network_from_vmv,
    save_network_to_casx,
    save_network_to_csv,
    save_network_to_json,
    save_network_to_vmv,
)
from .tiff import dicom_to_tiff, load_tiff_volume

__all__ = [
    "MatNetwork",
    # network
    "Network",
    "convert_casx_to_vmv",
    "dicom_to_tiff",
    # pipeline export
    "export_pipeline_results",
    "load_network_from_casx",
    "load_network_from_csv",
    "load_network_from_json",
    "load_network_from_mat",
    "load_network_from_vmv",
    # image loading
    "load_tiff_volume",
    "parse_registration_file",
    "partition_network",
    "save_network_to_casx",
    "save_network_to_csv",
    "save_network_to_json",
    "save_network_to_vmv",
]
