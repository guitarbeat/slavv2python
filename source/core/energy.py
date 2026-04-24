"""
Energy field calculations for source.
Includes Hessian-based vessel enhancement (Frangi/Sato) and Numba-accelerated gradient computation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from ._energy.chunking import (
    _calculate_energy_field_chunked,
    _compute_direct_energy_outputs,
    _compute_energy_scale,
    _energy_lattice,
    _energy_result_payload,
    _open_energy_storage_array,
    _remove_storage_path,
    _select_energy_storage_format,
)
from ._energy.config import _prepare_energy_config
from ._energy.gradients import (
    compute_gradient_fast,
    compute_gradient_impl,
    is_numba_acceleration_enabled,
    spherical_structuring_element,
)
from ._energy.resumable import calculate_energy_field_resumable as _calculate_energy_field_resumable

if TYPE_CHECKING:
    from source.runtime import StageController


def calculate_energy_field(
    image: np.ndarray, params: dict[str, Any], get_chunking_lattice_func=None
) -> dict[str, Any]:
    """
    Calculate multi-scale energy field using Hessian-based filtering.

    This implements the energy calculation from ``get_energy_V202`` in
    MATLAB, including PSF prefiltering and configurable Gaussian/annular
    ratios. Set ``energy_method='frangi'`` or ``'sato'`` in ``params`` to use
    scikit-image's :func:`~skimage.filters.frangi` or
    :func:`~skimage.filters.sato` vesselness filters as alternative backends.
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



