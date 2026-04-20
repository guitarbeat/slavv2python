"""Shared helpers for edge-candidate generation and parity flows."""

from __future__ import annotations

import math
from typing import Any, cast

import numpy as np
from skimage.graph import route_through_array
from typing_extensions import TypeAlias

from .._vertices.payloads import matlab_linear_indices as _matlab_linear_indices

Int16Array: TypeAlias = "np.ndarray"
Int32Array: TypeAlias = "np.ndarray"
Int64Array: TypeAlias = "np.ndarray"
Float32Array: TypeAlias = "np.ndarray"
Float64Array: TypeAlias = "np.ndarray"
BoolArray: TypeAlias = "np.ndarray"


def _use_matlab_frontier_tracer(energy_data: dict[str, Any], params: dict[str, Any]) -> bool:
    """Enable the parity-specific frontier tracer only for MATLAB-energy parity runs."""
    if not bool(params.get("comparison_exact_network", False)):
        return False
    return energy_data.get("energy_origin") == "matlab_batch_hdf5"


def _parity_watershed_candidate_mode(params: dict[str, Any]) -> str:
    """Return the MATLAB-parity watershed candidate strategy."""
    requested_mode = str(params.get("parity_watershed_candidate_mode", "all_contacts")).strip()
    normalized_mode = requested_mode.lower()
    allowed_modes = {"all_contacts", "remaining_origin_contacts"}
    return normalized_mode if normalized_mode in allowed_modes else "all_contacts"


def _parity_watershed_metric_threshold_from_params(params: dict[str, Any]) -> float | None:
    """Return the optional parity-only watershed metric threshold."""
    threshold_raw = params.get("parity_watershed_metric_threshold")
    if threshold_raw in (None, ""):
        return None
    threshold_value = cast("Any", threshold_raw)
    return float(threshold_value)


def _parity_candidate_salvage_mode(
    params: dict[str, Any],
    candidate_mode: str,
) -> str:
    """Return the MATLAB-parity candidate salvage strategy."""
    requested_mode = str(params.get("parity_candidate_salvage_mode", "auto")).strip().lower()
    allowed_modes = {
        "auto",
        "none",
        "frontier_deficit_geodesic",
        "all_origins_geodesic",
    }
    normalized_mode = requested_mode if requested_mode in allowed_modes else "auto"
    if normalized_mode == "auto":
        return "frontier_deficit_geodesic"
    return normalized_mode


def _matlab_frontier_edge_budget(params: dict[str, Any]) -> int:
    """Return MATLAB's per-origin frontier edge budget from get_edges_for_vertex.m."""
    requested_edges = int(params.get("number_of_edges_per_vertex", 4))
    return max(1, requested_edges)


def _matlab_frontier_offsets(
    strel_apothem: int,
    microns_per_voxel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct MATLAB-style cube-neighborhood offsets with Y-fastest ordering."""
    local_range = np.arange(-strel_apothem, strel_apothem + 1, dtype=np.int32)
    offsets = np.array(
        [[y, x, z] for z in local_range for x in local_range for y in local_range],
        dtype=np.int32,
    )
    distances = np.sqrt(np.sum((offsets.astype(np.float64) * microns_per_voxel) ** 2, axis=1))
    return offsets, distances.astype(np.float32, copy=False)


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


def _candidate_endpoint_pair_set(connections: np.ndarray) -> set[tuple[int, int]]:
    """Return the orientation-independent terminal endpoint pairs in a candidate payload."""
    pairs: set[tuple[int, int]] = set()
    connections = np.asarray(connections, dtype=np.int32).reshape(-1, 2)
    for start_vertex, end_vertex in connections:
        if int(start_vertex) < 0 or int(end_vertex) < 0:
            continue
        u, v = int(start_vertex), int(end_vertex)
        pair: tuple[int, int] = (u, v) if u < v else (v, u)
        pairs.add(pair)
    return pairs


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


def _candidate_incident_pair_counts(
    connections: np.ndarray,
) -> dict[int, int]:
    """Count unique incident endpoint pairs for each vertex."""
    counts: dict[int, int] = {}
    for start_vertex, end_vertex in _candidate_endpoint_pair_set(connections):
        counts[int(start_vertex)] = counts.get(int(start_vertex), 0) + 1
        counts[int(end_vertex)] = counts.get(int(end_vertex), 0) + 1
    return counts
