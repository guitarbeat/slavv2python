"""
Energy field calculations for source.
Includes Hessian-based vessel enhancement (Frangi/Sato) and Numba-accelerated gradient computation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from slavv_python.processing.stages.energy.gradients import (
    compute_gradient_fast,
    compute_gradient_impl,
    is_numba_acceleration_enabled,
    spherical_structuring_element,
)
from slavv_python.processing.stages.energy.manager import EnergyManager

if TYPE_CHECKING:
    import numpy as np

    from slavv_python.engine.state import StageController
    from slavv_python.schema.results import EnergyResult


def calculate_energy_field(
    image: np.ndarray, params: dict[str, Any], get_chunking_lattice_func=None
) -> EnergyResult:
    """
    Calculate multi-scale energy field using MATLAB-faithful matched filtering.

    The default ``energy_method='hessian'`` path implements the released
    ``get_energy_V202`` / ``energy_filter_V200`` style matched-filter energy
    stage, including MATLAB-style octave downsampling and a configurable
    projection mode over the scale dimension. Set ``energy_method='frangi'``,
    ``'sato'``, ``'simpleitk_objectness'``, or ``'cupy_hessian'`` in
    ``params`` to use explicit non-parity backends.
    """
    return EnergyManager.run(image, params, get_chunking_lattice_func)


def calculate_energy_field_resumable(
    image: np.ndarray,
    params: dict[str, Any],
    stage_controller: StageController,
    get_chunking_lattice_func=None,
) -> EnergyResult:
    """Compute energy with resumable chunk/scale units backed by memmaps."""
    return EnergyManager.run_resumable(
        image,
        params,
        stage_controller,
        get_chunking_lattice_func=get_chunking_lattice_func,
    )


__all__ = [
    "calculate_energy_field",
    "calculate_energy_field_resumable",
    "compute_gradient_fast",
    "compute_gradient_impl",
    "is_numba_acceleration_enabled",
    "spherical_structuring_element",
]
