"""Shared payload helpers for vertex extraction."""

from __future__ import annotations

from typing import cast

import numpy as np
from typing_extensions import TypeAlias

from slavv_python.schema.results import VertexSet

Int64Array: TypeAlias = "np.ndarray"
Float32Array: TypeAlias = "np.ndarray"


def empty_vertices_result() -> VertexSet:
    """Return the canonical empty vertex payload."""
    return VertexSet.create(
        vertex_positions=np.empty((0, 3), dtype=np.float32),
        vertex_scales=np.empty((0,), dtype=np.int16),
        vertex_energies=np.empty((0,), dtype=np.float32),
        lumen_radius_pixels=np.empty((0,), dtype=np.float32),
        lumen_radius_microns=np.empty((0,), dtype=np.float32),
    )


def build_vertices_result(
    vertex_positions: np.ndarray,
    vertex_scales: np.ndarray,
    vertex_energies: np.ndarray,
    lumen_radius_pixels: np.ndarray,
    lumen_radius_microns: np.ndarray,
) -> VertexSet:
    """Build the canonical vertex payload using the deep VertexSet schema."""
    return VertexSet.create(
        vertex_positions=vertex_positions,
        vertex_scales=vertex_scales,
        vertex_energies=vertex_energies,
        lumen_radius_pixels=lumen_radius_pixels,
        lumen_radius_microns=lumen_radius_microns,
    )


def coerce_radius_axes(
    lumen_radius_pixels: np.ndarray,
    lumen_radius_pixels_axes: np.ndarray | None,
) -> np.ndarray:
    """Normalize scale radii into a ``(num_scales, 3)`` axis-aware array."""
    if lumen_radius_pixels_axes is not None:
        axes: Float32Array = np.asarray(lumen_radius_pixels_axes, dtype=np.float32)
        if axes.ndim == 2 and axes.shape[1] == 3:
            return cast("np.ndarray", axes)

    radii = np.asarray(lumen_radius_pixels, dtype=np.float32).reshape(-1, 1)
    repeated_radii: Float32Array = np.repeat(radii, 3, axis=1)
    return cast("np.ndarray", repeated_radii)


def sort_vertex_order(
    vertex_positions: np.ndarray,
    vertex_energies: np.ndarray,
    image_shape: tuple[int, int, int],
    energy_sign: float,
) -> np.ndarray:
    """Sort vertices like MATLAB: by energy, then by column-major linear index for ties."""
    if len(vertex_positions) == 0:
        empty_order: Int64Array = np.array([], dtype=np.int64)
        return cast("np.ndarray", empty_order)

    linear_indices = matlab_linear_indices(vertex_positions, image_shape)
    if energy_sign < 0:
        sort_order: Int64Array = np.asarray(
            np.lexsort((linear_indices, vertex_energies)),
            dtype=np.int64,
        )
        return cast("np.ndarray", sort_order)
    sort_order = np.asarray(
        np.lexsort((linear_indices, -vertex_energies)),
        dtype=np.int64,
    )
    return cast("np.ndarray", sort_order)


from slavv_python.utils.matlab_order import zyx_to_matlab_linear_indices

def matlab_linear_indices(coords: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    """Return MATLAB-style column-major linear indices for 0-based coordinates.
    
    Coordinates are assumed to be in physical [Z, Y, X] order (index 0, 1, 2).
    MATLAB column-major order prioritizes Y, then X, then Z.
    """
    return zyx_to_matlab_linear_indices(coords, shape)


__all__ = [
    "build_vertices_result",
    "coerce_radius_axes",
    "empty_vertices_result",
    "matlab_linear_indices",
    "sort_vertex_order",
]
