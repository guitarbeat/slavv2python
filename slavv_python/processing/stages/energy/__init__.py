"""Energy calculation package."""

from __future__ import annotations

from .energy import (
    calculate_energy_field,
    calculate_energy_field_resumable,
    compute_gradient_fast,
    compute_gradient_impl,
    is_numba_acceleration_enabled,
    spherical_structuring_element,
)

# Alias for top-level orchestrator and legacy tests
compute_energy = calculate_energy_field

__all__ = [
    "calculate_energy_field",
    "calculate_energy_field_resumable",
    "compute_energy",
    "compute_gradient_fast",
    "compute_gradient_impl",
    "is_numba_acceleration_enabled",
    "spherical_structuring_element",
]
