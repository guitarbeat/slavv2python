"""
Utility modules for SLAVV.

This subpackage contains shared utilities:
- preprocessing: Image preprocessing functions
- validation: Parameter validation
- chunking: Memory-efficient processing utilities
- math: Mathematical helper functions
"""

from __future__ import annotations

from .chunking import get_chunking_lattice
from .formatting import format_size, format_time
from .math import calculate_path_length, fourier_transform_even, weighted_ks_test
from .preprocessing import preprocess_image
from .profiling import profile_process_image
from .synthetic import generate_synthetic_vessel_volume
from .system_info import get_matlab_info, get_system_info
from .validation import validate_parameters

__all__ = [
    "calculate_path_length",
    "format_size",
    "format_time",
    "fourier_transform_even",
    "generate_synthetic_vessel_volume",
    "get_chunking_lattice",
    "get_matlab_info",
    "get_system_info",
    "preprocess_image",
    "profile_process_image",
    "validate_parameters",
    "weighted_ks_test",
]
