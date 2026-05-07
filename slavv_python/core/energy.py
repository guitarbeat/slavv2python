"""
Energy field calculations for source.
Includes Hessian-based vessel enhancement (Frangi/Sato) and Numba-accelerated gradient computation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from .energy_internal.energy_chunking import (
    _calculate_energy_field_chunked,
    _compute_direct_energy_outputs,
    _compute_energy_scale,
    _energy_lattice,
    _energy_result_payload,
    _open_energy_storage_array,
    _project_scale_stack,
    _remove_storage_path,
    _select_energy_storage_format,
)
from .energy_internal.energy_config import _prepare_energy_config
from .energy_internal.energy_gradients import (
    compute_gradient_fast,
    compute_gradient_impl,
    is_numba_acceleration_enabled,
    spherical_structuring_element,
)
from .energy_internal.resumable_energy import (
    calculate_energy_field_resumable as _calculate_energy_field_resumable,
)

if TYPE_CHECKING:
    from slavv_python.runtime import StageController


def calculate_energy_field(
    image: np.ndarray, params: dict[str, Any], get_chunking_lattice_func=None
) -> dict[str, Any]:
    """
    Calculate multi-scale energy field using MATLAB-faithful matched filtering.

    The default ``energy_method='hessian'`` path implements the released
    ``get_energy_V202`` / ``energy_filter_V200`` style matched-filter energy
    stage, including MATLAB-style octave downsampling and a configurable
    projection mode over the scale dimension. Set ``energy_method='frangi'``,
    ``'sato'``, ``'simpleitk_objectness'``, or ``'cupy_hessian'`` in
    ``params`` to use explicit non-parity backends.
    """
    image = image.astype(np.float32, copy=False)
    config = _prepare_energy_config(image, params)
    lattice = _energy_lattice(
        image.shape,
        int(config["max_voxels"]),
        int(config["margin"]),
        get_chunking_lattice_func,
    )
    if len(lattice) > 1:
        return cast(
            "dict[str, Any]",
            _calculate_energy_field_chunked(
                image,
                params,
                config,
                lattice,
                get_chunking_lattice_func,
                calculate_energy_field,
            ),
        )
    energy_3d, scale_indices, energy_4d = _compute_direct_energy_outputs(image, config)
    return cast(
        "dict[str, Any]",
        _energy_result_payload(config, image.shape, energy_3d, scale_indices, energy_4d),
    )


def calculate_energy_field_resumable(
    image: np.ndarray,
    params: dict[str, Any],
    stage_controller: StageController,
    get_chunking_lattice_func=None,
) -> dict[str, Any]:
    """Compute energy with resumable chunk/scale units backed by memmaps."""
    return cast(
        "dict[str, Any]",
        _calculate_energy_field_resumable(
            image,
            params,
            stage_controller,
            get_chunking_lattice_func=get_chunking_lattice_func,
            prepare_energy_config=_prepare_energy_config,
            select_energy_storage_format=_select_energy_storage_format,
            energy_lattice=_energy_lattice,
            remove_storage_path=_remove_storage_path,
            open_energy_storage_array=_open_energy_storage_array,
            compute_energy_scale=_compute_energy_scale,
            project_scale_stack=_project_scale_stack,
        ),
    )


__all__ = [
    "calculate_energy_field",
    "calculate_energy_field_resumable",
    "compute_gradient_fast",
    "compute_gradient_impl",
    "is_numba_acceleration_enabled",
    "spherical_structuring_element",
]
