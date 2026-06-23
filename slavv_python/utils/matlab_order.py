"""MATLAB-compatible linear indexing and coordinate transformations."""

from __future__ import annotations

from typing import cast

import numpy as np


def zyx_to_matlab_linear_indices(
    coords_zyx: np.ndarray,
    shape_zyx: tuple[int, int, int],
) -> np.ndarray:
    """Convert physical [Z, Y, X] coordinates to MATLAB [Y, X, Z] linear indices.

    In MATLAB, the first dimension (Y) is the fastest-varying (stride 1),
    followed by the second dimension (X), and then the third (Z).

    Formula: y + x*ny + z*ny*nx
    """
    coords = np.asarray(coords_zyx, dtype=np.int64)
    _nz, ny, nx = shape_zyx
    # coords[:, 0] is Z, coords[:, 1] is Y, coords[:, 2] is X
    linear_indices = coords[:, 1] + coords[:, 2] * ny + coords[:, 0] * ny * nx
    return cast("np.ndarray", linear_indices)


def yxz_to_matlab_linear_indices(
    coords_yxz: np.ndarray,
    shape_yxz: tuple[int, int, int],
) -> np.ndarray:
    """Convert internal [Y, X, Z] coordinates to MATLAB [Y, X, Z] linear indices.

    Formula: y + x*ny + z*ny*nx
    """
    coords = np.asarray(coords_yxz, dtype=np.int64)
    ny, nx, _nz = shape_yxz
    linear_indices = coords[:, 0] + coords[:, 1] * ny + coords[:, 2] * ny * nx
    return cast("np.ndarray", linear_indices)


def matlab_linear_index_to_yxz(
    index: int | np.ndarray,
    shape_yxz: tuple[int, int, int],
) -> np.ndarray:
    """Convert MATLAB linear index to [Y, X, Z] coordinates."""
    ny, nx, _nz = shape_yxz
    xy_plane = ny * nx

    z = index // xy_plane
    remainder = index % xy_plane
    x = remainder // ny
    y = remainder % ny

    if isinstance(index, np.ndarray):
        return np.stack([y, x, z], axis=-1).astype(np.int32)
    return np.array([y, x, z], dtype=np.int32)


def matlab_linear_index_to_zyx(
    index: int | np.ndarray,
    shape_zyx: tuple[int, int, int],
) -> np.ndarray:
    """Convert MATLAB linear index to physical [Z, Y, X] coordinates."""
    _nz, ny, nx = shape_zyx
    xy_plane = ny * nx

    z = index // xy_plane
    remainder = index % xy_plane
    x = remainder // ny
    y = remainder % ny

    if isinstance(index, np.ndarray):
        return np.stack([z, y, x], axis=-1).astype(np.int32)
    return np.array([z, y, x], dtype=np.int32)
