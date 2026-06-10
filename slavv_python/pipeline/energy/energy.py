"""
Energy field calculations for source.
Includes Hessian-based vessel enhancement (Frangi/Sato) and Numba-accelerated gradient computation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from slavv_python.pipeline.energy.gradients import (
    compute_gradient_fast,
    compute_gradient_impl,
    is_numba_acceleration_enabled,
    spherical_structuring_element,
)
from slavv_python.pipeline.energy.manager import EnergyManager

if TYPE_CHECKING:
    import numpy as np

    from slavv_python.engine.state import StageController
    from slavv_python.schema.results import EnergyResult


def calculate_energy_field(
    image: np.ndarray, params: dict[str, Any], get_chunking_lattice_func=None
) -> EnergyResult:
    """Calculate the multi-scale energy field for vessel enhancement.

    This function implements MATLAB-faithful matched filtering. The default
    `energy_method='hessian'` path mirrors the `get_energy_V202` logic,
    including octave downsampling and projection.

    Args:
        image: The 3D input volume (TIFF data).
        params: Authoritative configuration dictionary.
        get_chunking_lattice_func: Optional override for chunk lattice calculation.

    Returns:
        An EnergyResult object containing the 3D energy map and scale indices.
    """
    return EnergyManager.run(image, params, get_chunking_lattice_func)


def calculate_energy_field_resumable(
    image: np.ndarray,
    params: dict[str, Any],
    stage_controller: StageController,
    get_chunking_lattice_func=None,
) -> EnergyResult:
    """Compute the multi-scale energy field with resumable checkpointing.

    This version persists intermediate chunk/scale units to disk (typically
    using memmaps) to allow recovery after interruption.

    Args:
        image: The 3D input volume.
        params: Authoritative configuration dictionary.
        stage_controller: Controller for resumable checkpointing.
        get_chunking_lattice_func: Optional override for chunk lattice calculation.

    Returns:
        An EnergyResult object containing the 3D energy map and scale indices.
    """
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
