"""Exact-MATLAB shared-state helpers for the global watershed edge discovery port."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from scipy import sparse

from .common import (
    _coord_to_matlab_linear_index,
    _matlab_frontier_scale_offsets,
    _matlab_linear_index_to_coord,
)


def _matlab_global_watershed_border_locations(shape: tuple[int, int, int]) -> np.ndarray:
    """Return MATLAB-order linear indices for the image border at ``strel_apothem = 1``."""
    border_locations: list[int] = []
    for z in range(shape[2]):
        for x in range(shape[1]):
            for y in range(shape[0]):
                if y in (0, shape[0] - 1) or x in (0, shape[1] - 1) or z in (0, shape[2] - 1):
                    border_locations.append(_coord_to_matlab_linear_index(np.array([y, x, z]), shape))
    return cast("np.ndarray", np.asarray(sorted(set(border_locations)), dtype=np.int64))


def _initialize_matlab_global_watershed_state(
    energy: np.ndarray,
    vertex_positions: np.ndarray,
) -> dict[str, Any]:
    """Build MATLAB-shaped shared maps for global watershed edge discovery."""
    shape: tuple[int, int, int] = (
        int(energy.shape[0]),
        int(energy.shape[1]),
        int(energy.shape[2]),
    )
    vertex_coords = np.rint(np.asarray(vertex_positions, dtype=np.float32)).astype(np.int32, copy=False)
    max_coord = np.asarray(shape, dtype=np.int32) - 1
    vertex_coords = np.clip(vertex_coords, 0, max_coord)
    vertex_locations = np.asarray(
        [_coord_to_matlab_linear_index(coord, shape) for coord in vertex_coords],
        dtype=np.int64,
    )
    number_of_vertices = len(vertex_locations)
    border_locations = _matlab_global_watershed_border_locations(shape)

    branch_order_map = np.zeros(shape, dtype=np.uint8)
    d_over_r_map = np.zeros(shape, dtype=np.float32)
    pointer_map = np.zeros(shape, dtype=np.uint64)
    vertex_index_map = np.zeros(shape, dtype=np.uint32)

    vertex_energies = np.empty((number_of_vertices,), dtype=np.float32)
    for vertex_offset, linear_index in enumerate(vertex_locations):
        coord = _matlab_linear_index_to_coord(int(linear_index), shape)
        vertex_index_map[coord[0], coord[1], coord[2]] = np.uint32(vertex_offset + 1)
        vertex_energies[vertex_offset] = np.float32(energy[coord[0], coord[1], coord[2]])

    for linear_index in border_locations:
        coord = _matlab_linear_index_to_coord(int(linear_index), shape)
        vertex_index_map[coord[0], coord[1], coord[2]] = np.uint32(number_of_vertices + 1)

    energy_map_temp = np.asarray(energy, dtype=np.float32).copy()
    for coord in vertex_coords:
        energy_map_temp[int(coord[0]), int(coord[1]), int(coord[2])] = float("-inf")

    available_locations = vertex_locations[::-1].astype(np.int64, copy=False)
    vertex_adjacency_matrix = sparse.identity(number_of_vertices + 1, format="csr", dtype=bool)

    return {
        "vertex_locations": vertex_locations,
        "border_locations": border_locations,
        "vertex_energies": vertex_energies,
        "energy_map_temp": energy_map_temp,
        "branch_order_map": branch_order_map,
        "d_over_r_map": d_over_r_map,
        "pointer_map": pointer_map,
        "vertex_index_map": vertex_index_map,
        "available_locations": available_locations,
        "vertex_adjacency_matrix": vertex_adjacency_matrix,
    }


def _matlab_global_watershed_current_strel(
    current_linear: int,
    *,
    current_scale_index: int,
    shape: tuple[int, int, int],
    lumen_radius_microns: np.ndarray,
    microns_per_voxel: np.ndarray,
    step_size_per_origin_radius: float,
) -> dict[str, np.ndarray]:
    """Build the in-bounds MATLAB strel around one current location."""
    current_coord = _matlab_linear_index_to_coord(int(current_linear), shape)
    offsets, offset_distances = _matlab_frontier_scale_offsets(
        int(current_scale_index),
        lumen_radius_microns,
        microns_per_voxel,
        step_size_per_origin_radius=step_size_per_origin_radius,
    )
    strel_coords = current_coord[None, :] + offsets
    valid_mask = (
        (strel_coords[:, 0] >= 0)
        & (strel_coords[:, 0] < shape[0])
        & (strel_coords[:, 1] >= 0)
        & (strel_coords[:, 1] < shape[1])
        & (strel_coords[:, 2] >= 0)
        & (strel_coords[:, 2] < shape[2])
    )
    valid_coords = np.asarray(strel_coords[valid_mask], dtype=np.int32)
    valid_offsets = np.asarray(offsets[valid_mask], dtype=np.int32)
    valid_distances = np.asarray(offset_distances[valid_mask], dtype=np.float32)
    valid_linear = np.asarray(
        [_coord_to_matlab_linear_index(coord, shape) for coord in valid_coords],
        dtype=np.int64,
    )
    pointer_indices = np.arange(1, len(valid_linear) + 1, dtype=np.uint64)
    origin_radius = max(float(lumen_radius_microns[int(current_scale_index)]), 1e-6)
    strel_r_over_R = valid_distances.astype(np.float64) / origin_radius
    return {
        "current_coord": current_coord.astype(np.int32, copy=False),
        "coords": valid_coords,
        "offsets": valid_offsets,
        "linear_indices": valid_linear,
        "pointer_indices": pointer_indices,
        "r_over_R": strel_r_over_R.astype(np.float32, copy=False),
    }


def _matlab_global_watershed_reveal_unclaimed_strel(
    *,
    current_vertex_index: int,
    current_scale_index: int,
    current_d_over_r: float,
    strel_coords: np.ndarray,
    strel_pointer_indices: np.ndarray,
    strel_r_over_R: np.ndarray,
    strel_adjusted_energies: np.ndarray,
    vertex_index_map: np.ndarray,
    energy_map: np.ndarray,
    pointer_map: np.ndarray,
    d_over_r_map: np.ndarray,
    size_map: np.ndarray,
) -> dict[str, np.ndarray]:
    """Reveal one MATLAB strel into the shared maps, claiming only previously unowned voxels."""
    vertices_of_current_strel = vertex_index_map[
        strel_coords[:, 0],
        strel_coords[:, 1],
        strel_coords[:, 2],
    ]
    is_without_vertex = vertices_of_current_strel == 0
    if np.any(is_without_vertex):
        claim_coords = strel_coords[is_without_vertex]
        vertex_index_map[
            claim_coords[:, 0],
            claim_coords[:, 1],
            claim_coords[:, 2],
        ] = np.uint32(current_vertex_index)
        energy_map[
            claim_coords[:, 0],
            claim_coords[:, 1],
            claim_coords[:, 2],
        ] = np.asarray(strel_adjusted_energies[is_without_vertex], dtype=np.float32)
        pointer_map[
            claim_coords[:, 0],
            claim_coords[:, 1],
            claim_coords[:, 2],
        ] = np.asarray(strel_pointer_indices[is_without_vertex], dtype=np.uint64)
        d_over_r_map[
            claim_coords[:, 0],
            claim_coords[:, 1],
            claim_coords[:, 2],
        ] = np.asarray(strel_r_over_R[is_without_vertex], dtype=np.float32) + np.float32(
            current_d_over_r
        )
        size_map[
            claim_coords[:, 0],
            claim_coords[:, 1],
            claim_coords[:, 2],
        ] = np.asarray(current_scale_index, dtype=size_map.dtype)

    return {
        "vertices_of_current_strel": np.asarray(vertices_of_current_strel, dtype=np.uint32),
        "is_without_vertex_in_strel": np.asarray(is_without_vertex, dtype=bool),
    }


def _matlab_global_watershed_insert_available_location(
    available_locations: list[int],
    *,
    next_location: int,
    next_energy: float,
    energy_lookup: dict[int, float],
    seed_idx: int,
    is_current_location_clear: bool,
) -> list[int]:
    """Insert one next location using MATLAB's seed-order-dependent list logic."""
    if not available_locations:
        return [int(next_location)]

    if seed_idx == 1:
        if energy_lookup[int(available_locations[0])] <= float(next_energy):
            location_idx = 1
        else:
            location_idx = 1 + max(
                idx
                for idx, linear_index in enumerate(available_locations, start=1)
                if energy_lookup[int(linear_index)] > float(next_energy)
            )
    else:
        if energy_lookup[int(available_locations[-1])] >= float(next_energy):
            location_idx = (
                len(available_locations) + 1
                if is_current_location_clear
                else len(available_locations)
            )
        else:
            location_idx = next(
                idx
                for idx, linear_index in enumerate(available_locations, start=1)
                if energy_lookup[int(linear_index)] < float(next_energy)
            )

    insert_at = location_idx - 1
    if is_current_location_clear:
        return [
            *available_locations[:insert_at],
            int(next_location),
            *available_locations[insert_at:],
        ]
    return [
        *available_locations[:insert_at],
        int(next_location),
        *available_locations[insert_at:-1],
    ]
