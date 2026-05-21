"""
SLAVV - Segmentation-Less, Automated, Vascular Vectorization

A Python implementation of the SLAVV algorithm for extracting and analyzing
vascular networks from 3D microscopy images.
"""

from __future__ import annotations

__version__ = "0.1.0"

# 1. ENGINE (Pipeline Control)
from .engine import SlavvPipeline, find_repo_root

# 2. STORAGE (Data I/O)
from .storage.loaders.tiff import load_tiff_volume
from .storage.loaders.network import load_network

# 3. UTILS (Shared Helpers)
from slavv_python.utils.validation import validate_parameters

__all__ = [
    "SlavvPipeline",
    "find_repo_root",
    "load_tiff_volume",
    "load_network",
    "validate_parameters",
]
