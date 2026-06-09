"""MATLAB-order linear indexing and path helpers for Edge Discovery."""

from __future__ import annotations

import math
from typing import cast

import numpy as np
from skimage.graph import route_through_array

from slavv_python.pipeline.edges.edge_types import Float32Array, Int32Array
from slavv_python.pipeline.vertices.results import (
    matlab_linear_indices as _matlab_linear_indices,
)


def _coord_to_matlab_linear_index(coord: np.ndarray, shape: tuple[int, int, int]) -> int:
    """Convert a 0-based ``(y, x, z)`` coordinate into MATLAB linear order."""
    y, x, z = (int(value) for value in coord[:3])
    return int(y + x * shape[0] + z * shape[0] * shape[1])


def _matlab_linear_index_to_coord(index: int, shape: tuple[int, int, int]) -> np.ndarray:
    """Convert a 0-based MATLAB linear index into a ``(y, x, z)`` coordinate."""
    xy_plane = shape[0] * shape[1]
    z = index // xy_plane
    pos_xy = index - z * xy_plane
    x = pos_xy // shape[0]
    y = pos_xy - x * shape[0]
    coord: Int32Array = np.array([y, x, z], dtype=np.int32)
    return cast("np.ndarray", coord)


def _argmin_with_linear_index_tiebreak(
    energies: np.ndarray,
    linear_indices: np.ndarray,
) -> int:
    """Return strel index with minimum energy; ties break on lowest Fortran linear index."""
    energy_values = np.asarray(energies, dtype=np.float64).reshape(-1)
    linear_values = np.asarray(linear_indices, dtype=np.int64).reshape(-1)
    if energy_values.size == 0:
        raise ValueError("energies must be non-empty")
    min_energy = float(np.min(energy_values))
    tied = np.flatnonzero(energy_values == min_energy)
    if tied.size == 1:
        return int(tied[0])
    return int(tied[np.argmin(linear_values[tied])])


def _path_coords_from_linear_indices(
    path_linear: list[int],
    shape: tuple[int, int, int],
) -> np.ndarray:
    """Convert a linear-index path into origin-to-terminal spatial coordinates."""
    coords = [_matlab_linear_index_to_coord(index, shape) for index in reversed(path_linear)]
    coord_array: Float32Array = np.asarray(coords, dtype=np.float32)
    return cast("np.ndarray", coord_array)


def _path_max_energy_from_linear_indices(
    path_linear: list[int],
    energy: np.ndarray,
    shape: tuple[int, int, int],
) -> float:
    """Return the maximum sampled energy along a linear-index path."""
    if not path_linear:
        return float("-inf")
    samples = []
    for index in path_linear:
        coord = _matlab_linear_index_to_coord(index, shape)
        samples.append(float(energy[coord[0], coord[1], coord[2]]))
    return max(samples, default=float("-inf"))


def _vertex_center_linear_lookup(
    vertex_positions: np.ndarray,
    image_shape: tuple[int, int, int],
) -> dict[int, int]:
    """Map rounded vertex centers to their vertex indices."""
    if len(vertex_positions) == 0:
        return {}
    coords = np.rint(np.asarray(vertex_positions, dtype=np.float32)).astype(np.int32, copy=False)
    max_coord: Int32Array = np.asarray(image_shape, dtype=np.int32) - 1
    coords = np.clip(coords, 0, max_coord)
    linear_indices = _matlab_linear_indices(coords, image_shape)
    return {
        int(linear_index): int(vertex_index)
        for vertex_index, linear_index in enumerate(linear_indices)
    }


def _trace_local_geodesic_between_vertices(
    energy: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    energy_sign: float,
    *,
    box_margin_voxels: int,
) -> np.ndarray | None:
    """Trace a local geodesic path between two vertices inside a bounded subvolume."""
    image_shape = energy.shape
    max_coord: Int32Array = np.asarray(image_shape, dtype=np.int32) - 1
    start_coord = np.clip(
        np.rint(np.asarray(start, dtype=np.float32)[:3]).astype(np.int32, copy=False),
        0,
        max_coord,
    )
    end_coord = np.clip(
        np.rint(np.asarray(end, dtype=np.float32)[:3]).astype(np.int32, copy=False),
        0,
        max_coord,
    )
    if np.array_equal(start_coord, end_coord):
        return None

    delta = np.abs(end_coord - start_coord)
    dynamic_margin = int(max(box_margin_voxels, 0) + math.ceil(float(np.max(delta)) * 0.25))
    lower = np.maximum(np.minimum(start_coord, end_coord) - dynamic_margin, 0)
    upper = np.minimum(np.maximum(start_coord, end_coord) + dynamic_margin + 1, image_shape)
    patch = np.asarray(
        energy[
            lower[0] : upper[0],
            lower[1] : upper[1],
            lower[2] : upper[2],
        ],
        dtype=np.float64,
    )
    if patch.size == 0:
        return None

    if energy_sign < 0:
        baseline = float(np.nanmin(patch))
        cost = patch - baseline + 1e-3
    else:
        baseline = float(np.nanmax(patch))
        cost = baseline - patch + 1e-3
    if not np.all(np.isfinite(cost)):
        return None

    local_start = tuple((start_coord - lower).tolist())
    local_end = tuple((end_coord - lower).tolist())
    try:
        local_coords, _weight = route_through_array(
            cost,
            local_start,
            local_end,
            fully_connected=True,
            geometric=True,
        )
    except (ValueError, RuntimeError):
        return None
    if len(local_coords) <= 1:
        return None

    global_coords = np.asarray(local_coords, dtype=np.int32) + lower
    deduped = [global_coords[0]]
    for coord in global_coords[1:]:
        if not np.array_equal(coord, deduped[-1]):
            deduped.append(coord)
    if len(deduped) <= 1:
        return None
    trace_coords: Float32Array = np.asarray(deduped, dtype=np.float32)
    return cast("np.ndarray", trace_coords)
