"""Preferred internal name for energy-gradient helpers."""

from __future__ import annotations

from .._energy.gradients import (
    compute_gradient_fast,
    compute_gradient_impl,
    is_numba_acceleration_enabled,
    spherical_structuring_element,
)

__all__ = [
    "compute_gradient_fast",
    "compute_gradient_impl",
    "is_numba_acceleration_enabled",
    "spherical_structuring_element",
]
