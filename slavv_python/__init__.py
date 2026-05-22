"""
SLAVV - Segmentation-Less, Automated, Vascular Vectorization

A Python implementation of the SLAVV algorithm for extracting and analyzing
vascular networks from 3D microscopy images.
"""

from __future__ import annotations

__version__ = "0.1.0"

# 1. ENGINE (Pipeline Control)
# 3. UTILS (Shared Helpers)
from slavv_python.utils.validation import validate_parameters

from .engine import SlavvPipeline, find_repo_root
from .storage.loaders.network import load_network

# 2. STORAGE (Data I/O)
from .storage.loaders.tiff import load_tiff_volume

__all__ = [
    "SlavvPipeline",
    "find_repo_root",
    "load_network",
    "load_tiff_volume",
    "validate_parameters",
]
