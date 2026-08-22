"""MATLAB-order linear indexing and path helpers for Edge Discovery."""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from skimage.graph import route_through_array

from slavv_python.pipeline.vertices.results import (
    matlab_linear_indices as _matlab_linear_indices,
)
from slavv_python.utils.matlab_order import matlab_linear_index_to_yxz, yxz_to_matlab_linear_indices

if TYPE_CHECKING:
    from slavv_python.pipeline.edges.edge_types import Float64Array, Int32Array

try:
    from numba import njit
except ImportError:
    njit = None

_NUMBA_AVAILABLE = njit is not None and os.environ.get("SLAVV_DISABLE_NUMBA", "0") != "1"


def _coord_to_matlab_linear_index(coord: np.ndarray, shape: tuple[int, int, int]) -> int:
    """Convert a 0-based ``(y, x, z)`` coordinate into MATLAB linear order."""
    return int(yxz_to_matlab_linear_indices(coord[None, :3], shape)[0])


def _matlab_linear_index_to_coord(index: int, shape: tuple[int, int, int]) -> np.ndarray:
    """Convert a 0-based MATLAB linear index into a ``(y, x, z)`` coordinate."""
    return cast("np.ndarray", matlab_linear_index_to_yxz(index, shape))


def _matlab_watershed_min_candidate_energies(energies: np.ndarray) -> np.ndarray:
    """Prepare strel energies for MATLAB ``min`` semantics (honor ``-Inf``, ignore ``NaN``/``+Inf``)."""
    working = np.asarray(energies, dtype=np.float64).copy()
    working[np.isnan(working)] = np.inf
    working[np.isposinf(working)] = np.inf
    return cast("np.ndarray", working)


def _argmin_with_linear_index_tiebreak_python(
    energy_values: np.ndarray,
    linear_values: np.ndarray,
) -> int:
    """Pure Python scalar tie-breaking finding min energy with lowest Fortran linear index."""
    n = len(energy_values)
    best_idx = 0
    best_energy = float(energy_values[0])
    best_linear = int(linear_values[0])
    for i in range(1, n):
        e = float(energy_values[i])
        if e < best_energy:
            best_energy = e
            best_linear = int(linear_values[i])
            best_idx = i
        elif e == best_energy:
            lin = int(linear_values[i])
            if lin < best_linear:
                best_linear = lin
                best_idx = i
    return best_idx


def _argmin_with_linear_index_tiebreak_numba_impl(
    energy_values: np.ndarray,
    linear_values: np.ndarray,
) -> int:
    n = len(energy_values)
    best_idx = 0
    best_energy = energy_values[0]
    best_linear = linear_values[0]
    for i in range(1, n):
        e = energy_values[i]
        if e < best_energy:
            best_energy = e
            best_linear = linear_values[i]
            best_idx = i
        elif e == best_energy:
            lin = linear_values[i]
            if lin < best_linear:
                best_linear = lin
                best_idx = i
    return best_idx


if _NUMBA_AVAILABLE:
    _numba_argmin_tiebreak = cast(
        "Any", njit(cache=False)(_argmin_with_linear_index_tiebreak_numba_impl)
    )
else:
    _numba_argmin_tiebreak = None


def _argmin_with_linear_index_tiebreak(
    energies: np.ndarray,
    linear_indices: np.ndarray,
) -> int:
    """Return strel index with minimum energy; ties break on lowest Fortran linear index."""
    energy_values = np.asarray(energies, dtype=np.float64).reshape(-1)
    linear_values = np.asarray(linear_indices, dtype=np.int64).reshape(-1)
    if energy_values.size == 0:
        raise ValueError("energies must be non-empty")

    global _NUMBA_AVAILABLE
    if _NUMBA_AVAILABLE and _numba_argmin_tiebreak is not None:
        try:
            return int(_numba_argmin_tiebreak(energy_values, linear_values))
        except Exception:
            _NUMBA_AVAILABLE = False

    return _argmin_with_linear_index_tiebreak_python(energy_values, linear_values)


def _path_coords_from_linear_indices(
    path_linear: list[int],
    shape: tuple[int, int, int],
) -> np.ndarray:
    """Convert a linear-index path into origin-to-terminal spatial coordinates."""
    coords = [_matlab_linear_index_to_coord(index, shape) for index in reversed(path_linear)]
    coord_array: Float64Array = np.asarray(coords, dtype=np.float64)
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
    coords = np.rint(np.asarray(vertex_positions, dtype=np.float64)).astype(np.int32, copy=False)
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
        np.rint(np.asarray(start, dtype=np.float64)[:3]).astype(np.int32, copy=False),
        0,
        max_coord,
    )
    end_coord = np.clip(
        np.rint(np.asarray(end, dtype=np.float64)[:3]).astype(np.int32, copy=False),
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
    trace_coords: Float64Array = np.asarray(deduped, dtype=np.float64)
    return cast("np.ndarray", trace_coords)
