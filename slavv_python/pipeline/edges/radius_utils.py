"""Shared helpers for scalar radius normalization."""

from __future__ import annotations

import numpy as np


def _scalar_radius(radius_value: np.ndarray | float | int) -> float:
    """Convert isotropic or axis-aware radii into a single tracing radius."""
    radius_array = np.asarray(radius_value, dtype=np.float32).reshape(-1)
    if radius_array.size == 0:
        return 0.0
    if radius_array.size == 1:
        return float(radius_array[0])
    return float(np.cbrt(np.prod(radius_array)))
