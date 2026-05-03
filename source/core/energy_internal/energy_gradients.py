"""Energy-gradient helpers."""

from __future__ import annotations

import logging
from typing import Any, cast

import numpy as np

try:
    from numba import njit
except ImportError:
    njit = None

_NUMBA_AVAILABLE = njit is not None
_NUMBA_FAILURE_MESSAGE = (
    "Numba gradient acceleration is unavailable in this environment; "
    "falling back to the pure-Python helpers."
)

logger = logging.getLogger(__name__)


def _compute_gradient_impl_python(energy, pos_int, microns_per_voxel):
    """Compute local energy gradient via central differences."""
    grad = np.zeros(3)

    pos_y, pos_x, pos_z = pos_int
    shape_y, shape_x, shape_z = energy.shape

    if shape_y < 3 or shape_x < 3 or shape_z < 3:
        return np.zeros(3)
    if pos_y < 1:
        pos_y = 1
    elif pos_y > shape_y - 2:
        pos_y = shape_y - 2

    if pos_x < 1:
        pos_x = 1
    elif pos_x > shape_x - 2:
        pos_x = shape_x - 2

    if pos_z < 1:
        pos_z = 1
    elif pos_z > shape_z - 2:
        pos_z = shape_z - 2

    grad[0] = (energy[pos_y + 1, pos_x, pos_z] - energy[pos_y - 1, pos_x, pos_z]) / (
            2.0 * microns_per_voxel[0]
    )
    grad[1] = (energy[pos_y, pos_x + 1, pos_z] - energy[pos_y, pos_x - 1, pos_z]) / (
            2.0 * microns_per_voxel[1]
    )
    grad[2] = (energy[pos_y, pos_x, pos_z + 1] - energy[pos_y, pos_x, pos_z - 1]) / (
            2.0 * microns_per_voxel[2]
    )

    return grad


def _compute_gradient_fast_python(energy, p0, p1, p2, inv_mpv_2x):
    """Optimized gradient computation avoiding position-array allocations."""
    s0, s1, s2 = energy.shape

    if s0 < 3 or s1 < 3 or s2 < 3:
        return np.zeros(3)

    if p0 < 1:
        p0 = 1
    elif p0 > s0 - 2:
        p0 = s0 - 2

    if p1 < 1:
        p1 = 1
    elif p1 > s1 - 2:
        p1 = s1 - 2

    if p2 < 1:
        p2 = 1
    elif p2 > s2 - 2:
        p2 = s2 - 2

    grad = np.empty(3)
    grad[0] = (energy[p0 + 1, p1, p2] - energy[p0 - 1, p1, p2]) * inv_mpv_2x[0]
    grad[1] = (energy[p0, p1 + 1, p2] - energy[p0, p1 - 1, p2]) * inv_mpv_2x[1]
    grad[2] = (energy[p0, p1, p2 + 1] - energy[p0, p1, p2 - 1]) * inv_mpv_2x[2]

    return grad


if _NUMBA_AVAILABLE:
    _compute_gradient_impl_numba = cast("Any", njit(cache=False)(_compute_gradient_impl_python))
    _compute_gradient_fast_numba = cast("Any", njit(cache=False)(_compute_gradient_fast_python))
else:
    _compute_gradient_impl_numba = None
    _compute_gradient_fast_numba = None

_NUMBA_ACCELERATION_ENABLED = _NUMBA_AVAILABLE


def compute_gradient_impl(
        energy: np.ndarray,
        pos_int: np.ndarray | tuple[int, int, int],
        microns_per_voxel: np.ndarray,
) -> np.ndarray:
    """Compute a local energy gradient with optional Numba acceleration."""
    global _NUMBA_ACCELERATION_ENABLED

    if _NUMBA_ACCELERATION_ENABLED and _compute_gradient_impl_numba is not None:
        try:
            return cast(
                "np.ndarray", _compute_gradient_impl_numba(energy, pos_int, microns_per_voxel)
            )
        except Exception as exc:  # pragma: no cover - depends on local numba build
            logger.warning("%s Detail: %s", _NUMBA_FAILURE_MESSAGE, exc)
            _NUMBA_ACCELERATION_ENABLED = False

    return cast("np.ndarray", _compute_gradient_impl_python(energy, pos_int, microns_per_voxel))


def compute_gradient_fast(
        energy: np.ndarray,
        p0: int,
        p1: int,
        p2: int,
        inv_mpv_2x: np.ndarray,
) -> np.ndarray:
    """Compute a local energy gradient without allocating a position array."""
    global _NUMBA_ACCELERATION_ENABLED

    if _NUMBA_ACCELERATION_ENABLED and _compute_gradient_fast_numba is not None:
        try:
            return cast("np.ndarray", _compute_gradient_fast_numba(energy, p0, p1, p2, inv_mpv_2x))
        except Exception as exc:  # pragma: no cover - depends on local numba build
            logger.warning("%s Detail: %s", _NUMBA_FAILURE_MESSAGE, exc)
            _NUMBA_ACCELERATION_ENABLED = False

    return cast("np.ndarray", _compute_gradient_fast_python(energy, p0, p1, p2, inv_mpv_2x))


def is_numba_acceleration_enabled() -> bool:
    """Return whether gradient helpers are currently using Numba-compiled paths."""
    return _NUMBA_ACCELERATION_ENABLED


def spherical_structuring_element(radius: int, microns_per_voxel: np.ndarray) -> np.ndarray:
    """Create a 3D spherical structuring element accounting for voxel spacing."""
    microns_per_voxel = np.asarray(microns_per_voxel, dtype=float)
    r_phys = float(radius) * microns_per_voxel.min()
    ranges = [
        np.arange(-int(np.ceil(r_phys / spacing)), int(np.ceil(r_phys / spacing)) + 1)
        for spacing in microns_per_voxel
    ]
    yy, xx, zz = np.meshgrid(*ranges, indexing="ij")
    dist2 = (
            (yy * microns_per_voxel[0]) ** 2
            + (xx * microns_per_voxel[1]) ** 2
            + (zz * microns_per_voxel[2]) ** 2
    )
    return (dist2 <= r_phys ** 2).astype(bool)  # type: ignore[no-any-return]


__all__ = [
    "compute_gradient_fast",
    "compute_gradient_impl",
    "is_numba_acceleration_enabled",
    "spherical_structuring_element",
]
