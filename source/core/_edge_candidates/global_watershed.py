"""Exact-MATLAB shared-state helpers for the global watershed edge discovery port."""

from __future__ import annotations

import time
from typing import Any, cast

import numpy as np
from scipy import sparse

from .._edge_payloads import _empty_edge_diagnostics
from ..edge_primitives import (
    _edge_metric_from_energy_trace,
)
from .common import (
    _build_matlab_global_watershed_lut,
    _coord_to_matlab_linear_index,
    _matlab_frontier_adjusted_neighbor_energies,
    _matlab_linear_index_to_coord,
)


def _matlab_global_watershed_border_locations(shape: tuple[int, int, int]) -> np.ndarray:
    """Return MATLAB-order linear indices for the image border at ``strel_apothem = 1``."""
    border_mask: np.ndarray = np.zeros(shape, dtype=bool)
    border_mask[0, :, :] = True
    border_mask[shape[0] - 1, :, :] = True
    border_mask[:, 0, :] = True
    border_mask[:, shape[1] - 1, :] = True
    border_mask[:, :, 0] = True
    border_mask[:, :, shape[2] - 1] = True
    return cast("np.ndarray", np.flatnonzero(border_mask.ravel(order="F")).astype(np.int64))


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
    vertex_coords = np.rint(np.asarray(vertex_positions, dtype=np.float32)).astype(
        np.int32, copy=False
    )
    max_coord: np.ndarray = np.asarray(shape, dtype=np.int32) - 1
    vertex_coords = np.clip(vertex_coords, 0, max_coord)
    vertex_locations = np.asarray(
        [_coord_to_matlab_linear_index(coord, shape) for coord in vertex_coords],
        dtype=np.int64,
    )
    number_of_vertices = len(vertex_locations)
    border_locations = _matlab_global_watershed_border_locations(shape)

    branch_order_map = np.zeros(shape, dtype=np.uint8)
    d_over_r_map = np.zeros(shape, dtype=np.float64)
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
    vertex_adjacency_matrix = sparse.identity(number_of_vertices + 1, format="lil", dtype=bool)

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
    current_scale_label: int,
    shape: tuple[int, int, int],
    lumen_radius_microns: np.ndarray,
    microns_per_voxel: np.ndarray,
    step_size_per_origin_radius: float,
) -> dict[str, np.ndarray]:
    """Build the in-bounds MATLAB strel around one current location."""
    current_coord = _matlab_linear_index_to_coord(int(current_linear), shape)
    current_scale_index = int(
        np.clip(int(current_scale_label) - 1, 0, len(lumen_radius_microns) - 1)
    )
    lut = _build_matlab_global_watershed_lut(
        current_scale_index,
        size_of_image=shape,
        lumen_radius_microns=lumen_radius_microns,
        microns_per_voxel=microns_per_voxel,
        step_size_per_origin_radius=step_size_per_origin_radius,
    )
    offsets = np.asarray(lut["local_subscripts"], dtype=np.int32)
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
    valid_linear = np.asarray(lut["linear_offsets"], dtype=np.int64)[valid_mask] + np.int64(
        current_linear
    )
    pointer_indices = np.arange(1, len(offsets) + 1, dtype=np.uint64)[valid_mask]
    return {
        "current_coord": current_coord.astype(np.int32, copy=False),
        "coords": valid_coords,
        "offsets": valid_offsets,
        "linear_indices": valid_linear,
        "pointer_indices": pointer_indices,
        "r_over_R": np.asarray(lut["r_over_R"], dtype=np.float32)[valid_mask],
        "unit_vectors": np.asarray(lut["unit_vectors"], dtype=np.float32)[valid_mask],
    }


def _matlab_global_watershed_reveal_unclaimed_strel(
    *,
    current_vertex_index: int,
    current_scale_label: int,
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
        ] = np.asarray(strel_r_over_R[is_without_vertex], dtype=np.float64) + np.float64(
            current_d_over_r
        )
        size_map[
            claim_coords[:, 0],
            claim_coords[:, 1],
            claim_coords[:, 2],
        ] = np.asarray(current_scale_label, dtype=size_map.dtype)

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


def _matlab_global_watershed_reset_join_locations(
    available_locations: list[int],
    *,
    next_vertex_locations: np.ndarray,
    is_current_location_clear: bool,
) -> tuple[list[int], bool]:
    """Mirror MATLAB's indexed available-location removal during watershed joins."""
    updated_available = [int(location) for location in available_locations]
    if not updated_available:
        return updated_available, is_current_location_clear

    locations_to_reset = np.intersect1d(
        np.asarray(updated_available, dtype=np.int64),
        np.asarray(next_vertex_locations, dtype=np.int64),
        assume_unique=False,
    ).astype(np.int64, copy=False)

    if not is_current_location_clear:
        is_current_location_clear = True
        current_available = int(updated_available.pop())
        if len(locations_to_reset):
            locations_to_reset = locations_to_reset[locations_to_reset != current_available]

    if not len(locations_to_reset):
        return updated_available, is_current_location_clear

    reset_indices: list[int] = []
    for location in locations_to_reset.tolist():
        try:
            reset_indices.append(updated_available.index(int(location)))
        except ValueError:
            continue
    if not reset_indices:
        return updated_available, is_current_location_clear

    reset_index_set = set(reset_indices)
    updated_available = [
        location
        for idx, location in enumerate(updated_available)
        if idx not in reset_index_set
    ]
    return updated_available, is_current_location_clear


def _matlab_global_watershed_unit_vectors(
    offsets: np.ndarray,
    microns_per_voxel: np.ndarray,
) -> np.ndarray:
    """Return MATLAB-style unit vectors for one local strel."""
    vectors: np.ndarray = np.asarray(offsets, dtype=np.float64) * np.asarray(
        microns_per_voxel,
        dtype=np.float64,
    )
    norms = np.linalg.norm(vectors, axis=1)
    unit_vectors: np.ndarray = np.zeros_like(vectors, dtype=np.float64)
    valid_mask = norms > 1e-12
    unit_vectors[valid_mask] = vectors[valid_mask] / norms[valid_mask, None]
    return cast("np.ndarray", unit_vectors.astype(np.float32, copy=False))


def _matlab_global_watershed_trace_half(
    start_linear: int,
    *,
    pointer_map: np.ndarray,
    size_map: np.ndarray,
    shape: tuple[int, int, int],
    lumen_radius_microns: np.ndarray,
    microns_per_voxel: np.ndarray,
    step_size_per_origin_radius: float,
) -> list[int]:
    """Trace one MATLAB watershed half-edge back to its zero-pointer origin."""
    traced: list[int] = []
    visited: set[int] = set()
    tracing_linear = int(start_linear)
    while True:
        if tracing_linear in visited:
            import logging
            logging.error(f"Cycle detected in global watershed backtrack at linear index {tracing_linear}. Breaking to prevent infinite loop.")
            break
        visited.add(tracing_linear)
        traced.append(int(tracing_linear))
        tracing_coord = _matlab_linear_index_to_coord(tracing_linear, shape)
        pointer_value = int(pointer_map[tracing_coord[0], tracing_coord[1], tracing_coord[2]])
        if pointer_value == 0:
            break
        tracing_scale_label = int(size_map[tracing_coord[0], tracing_coord[1], tracing_coord[2]])
        lut = _build_matlab_global_watershed_lut(
            int(np.clip(tracing_scale_label - 1, 0, len(lumen_radius_microns) - 1)),
            size_of_image=shape,
            lumen_radius_microns=lumen_radius_microns,
            microns_per_voxel=microns_per_voxel,
            step_size_per_origin_radius=step_size_per_origin_radius,
        )
        linear_offsets = np.asarray(lut["linear_offsets"], dtype=np.int64)
        tracing_linear = int(tracing_linear - int(linear_offsets[pointer_value - 1]))
    return traced


def _coords_from_linear_trace(
    linear_trace: list[int],
    shape: tuple[int, int, int],
) -> np.ndarray:
    """Convert a MATLAB-order linear trace into an ordered spatial path."""
    coords = [_matlab_linear_index_to_coord(index, shape) for index in linear_trace]
    return cast("np.ndarray", np.asarray(coords, dtype=np.float32))


def _sample_volume_from_matlab_linear_trace(
    linear_trace: list[int],
    volume: np.ndarray,
) -> np.ndarray:
    """Sample one volume exactly at normalized MATLAB-order linear indices."""
    if not linear_trace:
        return np.zeros((0,), dtype=np.asarray(volume).dtype)
    if np.asarray(volume).ndim == 1:
        flat_volume = np.asarray(volume)
    else:
        flat_volume = np.asarray(volume).ravel(order="F")
    linear_indices = np.asarray(linear_trace, dtype=np.int64)
    return cast("np.ndarray", flat_volume[linear_indices])


def _matlab_global_watershed_finalize_edge_trace(
    half_1: list[int],
    half_2: list[int],
    *,
    shape: tuple[int, int, int],
    energy_map: np.ndarray,
    scale_image: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build one MATLAB-style edge trace and sample its payloads by linear index."""
    full_linear_trace = [*list(reversed(half_1)), *half_2]
    trace = _coords_from_linear_trace(full_linear_trace, shape)
    energy_trace = np.asarray(
        _sample_volume_from_matlab_linear_trace(full_linear_trace, energy_map),
        dtype=np.float32,
    )
    if scale_image is None:
        scale_trace = np.zeros((len(full_linear_trace),), dtype=np.int16)
    else:
        scale_trace = np.asarray(
            _sample_volume_from_matlab_linear_trace(full_linear_trace, scale_image),
            dtype=np.int16,
        )
    return trace, energy_trace, scale_trace


def _matlab_global_watershed_scale_pointer_map(
    pointer_map: np.ndarray,
    size_map: np.ndarray,
    *,
    lumen_radius_microns: np.ndarray,
    microns_per_voxel: np.ndarray,
    step_size_per_origin_radius: float,
) -> np.ndarray:
    """Apply MATLAB's final pointer-map scaling by scale-specific strel length."""
    scaled_pointer_map = np.zeros(pointer_map.shape, dtype=np.float32)
    pointer_mask: np.ndarray = pointer_map > 0
    if not np.any(pointer_mask):
        return cast("np.ndarray", scaled_pointer_map)

    scale_labels = size_map[pointer_mask].astype(np.int64, copy=False)
    scale_indices = np.clip(scale_labels - 1, 0, len(lumen_radius_microns) - 1)
    strel_lengths = np.asarray(
        [
            len(
                _build_matlab_global_watershed_lut(
                    int(scale_index),
                    size_of_image=pointer_map.shape,
                    lumen_radius_microns=lumen_radius_microns,
                    microns_per_voxel=microns_per_voxel,
                    step_size_per_origin_radius=step_size_per_origin_radius,
                )["linear_offsets"]
            )
            for scale_index in scale_indices
        ],
        dtype=np.float32,
    )
    scaled_pointer_map[pointer_mask] = (
        1000.0 / np.maximum(strel_lengths, 1.0) * pointer_map[pointer_mask].astype(np.float32)
    )
    return cast("np.ndarray", scaled_pointer_map)


def _generate_edge_candidates_matlab_global_watershed(
    energy: np.ndarray,
    scale_indices: np.ndarray | None,
    vertex_positions: np.ndarray,
    vertex_scales: np.ndarray,
    lumen_radius_microns: np.ndarray,
    microns_per_voxel: np.ndarray,
    _vertex_center_image: np.ndarray,
    params: dict[str, Any],
    *,
    heartbeat: Any | None = None,
) -> dict[str, Any]:
    """Generate candidates with MATLAB's one-pass global shared-state watershed search."""
    del _vertex_center_image
    if len(vertex_positions) == 0:
        return {
            "traces": [],
            "connections": np.zeros((0, 2), dtype=np.int32),
            "metrics": np.zeros((0,), dtype=np.float32),
            "energy_traces": [],
            "scale_traces": [],
            "origin_indices": np.zeros((0,), dtype=np.int32),
            "connection_sources": [],
            "diagnostics": _empty_edge_diagnostics(),
            "matlab_global_watershed_exact": True,
        }

    energy_map_raw = np.asarray(energy, dtype=np.float32)
    shape: tuple[int, int, int] = (
        int(energy_map_raw.shape[0]),
        int(energy_map_raw.shape[1]),
        int(energy_map_raw.shape[2]),
    )
    state = _initialize_matlab_global_watershed_state(energy_map_raw, vertex_positions)
    vertex_locations = cast("np.ndarray", state["vertex_locations"])
    vertex_energies = cast("np.ndarray", state["vertex_energies"])
    energy_map_temp = cast("np.ndarray", state["energy_map_temp"])
    branch_order_map = cast("np.ndarray", state["branch_order_map"])
    d_over_r_map = cast("np.ndarray", state["d_over_r_map"])
    pointer_map = cast("np.ndarray", state["pointer_map"])
    vertex_index_map = cast("np.ndarray", state["vertex_index_map"])
    available_locations = [int(value) for value in cast("np.ndarray", state["available_locations"])]
    vertex_adjacency_matrix = cast("Any", state["vertex_adjacency_matrix"])
    number_of_vertices = len(vertex_locations)

    energy_map = np.asarray(energy_map_temp, dtype=np.float32).copy()
    original_scale_image: np.ndarray
    if scale_indices is None:
        size_map: np.ndarray = np.ones(shape, dtype=np.int16)
        original_scale_image = np.zeros(shape, dtype=np.int16)
    else:
        original_scale_image = np.asarray(
            cast("np.ndarray", scale_indices),
            dtype=np.int16,
        )
        size_map = np.asarray(original_scale_image, dtype=np.int16).copy()
        size_map += np.int16(1)
    vertex_coords = np.rint(np.asarray(vertex_positions, dtype=np.float32)).astype(
        np.int32, copy=False
    )
    vertex_coords[:, 0] = np.clip(vertex_coords[:, 0], 0, shape[0] - 1)
    vertex_coords[:, 1] = np.clip(vertex_coords[:, 1], 0, shape[1] - 1)
    vertex_coords[:, 2] = np.clip(vertex_coords[:, 2], 0, shape[2] - 1)
    size_map[
        vertex_coords[:, 0],
        vertex_coords[:, 1],
        vertex_coords[:, 2],
    ] = np.asarray(vertex_scales, dtype=size_map.dtype) + np.int16(1)

    edge_number_tolerance = 2
    energy_tolerance = 1.0
    step_size_per_origin_radius = float(params.get("step_size_per_origin_radius", 1.0))

    edge_halves: list[tuple[list[int], list[int]]] = []
    edge_pairs: list[tuple[int, int]] = []
    iteration_count = 0
    last_heartbeat_at = time.monotonic()

    while True:
        if not available_locations:
            break
        iteration_count += 1

        current_location = int(available_locations[-1])
        current_coord = _matlab_linear_index_to_coord(current_location, shape)
        min_available_energy = float(
            energy_map_temp[current_coord[0], current_coord[1], current_coord[2]]
        )
        if min_available_energy == float("-inf"):
            energy_map_temp[current_coord[0], current_coord[1], current_coord[2]] = energy_map[
                current_coord[0], current_coord[1], current_coord[2]
            ]
            min_available_energy = float(
                energy_map_temp[current_coord[0], current_coord[1], current_coord[2]]
            )
        if min_available_energy >= 0.0:
            break

        current_vertex_index = int(
            vertex_index_map[current_coord[0], current_coord[1], current_coord[2]]
        )
        if current_vertex_index <= 0:
            available_locations.pop()
            continue
        current_scale_label = int(size_map[current_coord[0], current_coord[1], current_coord[2]])
        current_scale_index = int(
            np.clip(current_scale_label - 1, 0, len(lumen_radius_microns) - 1)
        )
        current_strel = _matlab_global_watershed_current_strel(
            current_location,
            current_scale_label=current_scale_label,
            shape=shape,
            lumen_radius_microns=lumen_radius_microns,
            microns_per_voxel=microns_per_voxel,
            step_size_per_origin_radius=step_size_per_origin_radius,
        )
        current_strel_coords = cast("np.ndarray", current_strel["coords"])
        current_strel_linear = cast("np.ndarray", current_strel["linear_indices"])
        current_strel_offsets = cast("np.ndarray", current_strel["offsets"])
        current_strel_pointer_indices = cast("np.ndarray", current_strel["pointer_indices"])
        current_strel_r_over_r = cast("np.ndarray", current_strel["r_over_R"])
        current_strel_unit_vectors = cast("np.ndarray", current_strel["unit_vectors"])

        current_strel_energies = energy_map_temp[
            current_strel_coords[:, 0],
            current_strel_coords[:, 1],
            current_strel_coords[:, 2],
        ].astype(np.float32, copy=False)
        current_d_over_r = float(d_over_r_map[current_coord[0], current_coord[1], current_coord[2]])
        current_forward_unit: np.ndarray | None = None
        current_pointer_value = int(
            pointer_map[current_coord[0], current_coord[1], current_coord[2]]
        )
        if current_pointer_value > 0:
            lut = _build_matlab_global_watershed_lut(
                current_scale_index,
                size_of_image=shape,
                lumen_radius_microns=lumen_radius_microns,
                microns_per_voxel=microns_per_voxel,
                step_size_per_origin_radius=step_size_per_origin_radius,
            )
            full_unit_vectors = np.asarray(lut["unit_vectors"], dtype=np.float32)
            current_forward_unit = full_unit_vectors[current_pointer_value - 1]

        current_strel_energies = _matlab_frontier_adjusted_neighbor_energies(
            current_strel_energies,
            neighbor_offsets=current_strel_offsets,
            neighbor_distances_microns=current_strel_r_over_r
            * max(float(lumen_radius_microns[current_scale_index]), 1e-6),
            neighbor_scale_indices=size_map[
                current_strel_coords[:, 0],
                current_strel_coords[:, 1],
                current_strel_coords[:, 2],
            ],
            propagated_scale_index=current_scale_label,
            current_distance_microns=current_d_over_r
            * max(float(lumen_radius_microns[current_scale_index]), 1e-6),
            origin_radius_microns=max(float(lumen_radius_microns[current_scale_index]), 1e-6),
            current_forward_unit=current_forward_unit,
            microns_per_voxel=microns_per_voxel,
            lumen_radius_microns=lumen_radius_microns,
            distance_tolerance=3.0,
        )

        revealed = _matlab_global_watershed_reveal_unclaimed_strel(
            current_vertex_index=current_vertex_index,
            current_scale_label=current_scale_label,
            current_d_over_r=current_d_over_r,
            strel_coords=current_strel_coords,
            strel_pointer_indices=current_strel_pointer_indices,
            strel_r_over_R=current_strel_r_over_r,
            strel_adjusted_energies=current_strel_energies,
            vertex_index_map=vertex_index_map,
            energy_map=energy_map,
            pointer_map=pointer_map,
            d_over_r_map=d_over_r_map,
            size_map=size_map,
        )
        vertices_of_current_strel = cast("np.ndarray", revealed["vertices_of_current_strel"])
        is_new_trace_from_current_location = (
            int(pointer_map[current_coord[0], current_coord[1], current_coord[2]]) == 0
        )
        if is_new_trace_from_current_location:
            seed_index_range = range(1, edge_number_tolerance + 1)
        else:
            seed_index_range = range(1, 2)

        is_current_location_clear = False
        for seed_idx in seed_index_range:
            is_energy_tolerated_in_strel = current_strel_energies < (
                float(vertex_energies[current_vertex_index - 1]) * (1.0 - energy_tolerance)
            )
            strel_idx = int(np.argmin(current_strel_energies))
            next_location = int(current_strel_linear[strel_idx])
            next_coord = current_strel_coords[strel_idx]
            next_vertex_index = int(vertices_of_current_strel[strel_idx])

            if not bool(is_energy_tolerated_in_strel[strel_idx]):
                if not is_current_location_clear and available_locations:
                    is_current_location_clear = True
                    available_locations.pop()
            else:
                if next_vertex_index == 0:
                    branch_order = (
                        int(branch_order_map[current_coord[0], current_coord[1], current_coord[2]])
                        + seed_idx
                        - 1
                    )
                    branch_order_map[next_coord[0], next_coord[1], next_coord[2]] = np.uint8(
                        branch_order
                    )
                    if branch_order < edge_number_tolerance:
                        energy_lookup = {
                            location: float(
                                energy_map_temp[
                                    _matlab_linear_index_to_coord(location, shape)[0],
                                    _matlab_linear_index_to_coord(location, shape)[1],
                                    _matlab_linear_index_to_coord(location, shape)[2],
                                ]
                            )
                            for location in [*available_locations, next_location]
                        }
                        available_locations = _matlab_global_watershed_insert_available_location(
                            available_locations,
                            next_location=next_location,
                            next_energy=energy_lookup[next_location],
                            energy_lookup=energy_lookup,
                            seed_idx=seed_idx,
                            is_current_location_clear=is_current_location_clear,
                        )
                        if not is_current_location_clear:
                            is_current_location_clear = True
                else:
                    is_next_vertex_in_strel = vertices_of_current_strel == next_vertex_index
                    available_locations, is_current_location_clear = (
                        _matlab_global_watershed_reset_join_locations(
                            available_locations,
                            next_vertex_locations=current_strel_linear[is_next_vertex_in_strel],
                            is_current_location_clear=is_current_location_clear,
                        )
                    )

                    if not bool(
                        vertex_adjacency_matrix[next_vertex_index - 1, current_vertex_index - 1]
                    ):
                        vertex_adjacency_matrix[
                            current_vertex_index - 1,
                            next_vertex_index - 1,
                        ] = True
                        vertex_adjacency_matrix[
                            next_vertex_index - 1,
                            current_vertex_index - 1,
                        ] = True

                        half_1 = _matlab_global_watershed_trace_half(
                            current_location,
                            pointer_map=pointer_map,
                            size_map=size_map,
                            shape=shape,
                            lumen_radius_microns=lumen_radius_microns,
                            microns_per_voxel=microns_per_voxel,
                            step_size_per_origin_radius=step_size_per_origin_radius,
                        )

                        other_half_start = next_location
                        if next_vertex_index != number_of_vertices + 1:
                            is_vertex_b_origin = (
                                pointer_map[
                                    current_strel_coords[:, 0],
                                    current_strel_coords[:, 1],
                                    current_strel_coords[:, 2],
                                ]
                                == 0
                            ) & (vertices_of_current_strel == next_vertex_index)
                            if np.any(is_vertex_b_origin):
                                other_half_start = int(current_strel_linear[is_vertex_b_origin][0])

                        half_2 = _matlab_global_watershed_trace_half(
                            other_half_start,
                            pointer_map=pointer_map,
                            size_map=size_map,
                            shape=shape,
                            lumen_radius_microns=lumen_radius_microns,
                            microns_per_voxel=microns_per_voxel,
                            step_size_per_origin_radius=step_size_per_origin_radius,
                        )
                        edge_halves.append((half_1, half_2))
                        edge_pairs.append((current_vertex_index, next_vertex_index))

            cosine_to_selected = np.sum(
                current_strel_unit_vectors * current_strel_unit_vectors[strel_idx],
                axis=1,
            )
            current_strel_energies = current_strel_energies * (
                (1.0 - cosine_to_selected.astype(np.float32, copy=False)) / 2.0
            )

        if heartbeat is not None:
            now = time.monotonic()
            if (
                iteration_count == 1
                or iteration_count % 512 == 0
                or (now - last_heartbeat_at) >= 5.0
            ):
                heartbeat(iteration_count, len(edge_pairs))
                last_heartbeat_at = now

    traces: list[np.ndarray] = []
    connections: list[list[int]] = []
    metrics: list[float] = []
    energy_traces: list[np.ndarray] = []
    scale_traces: list[np.ndarray] = []
    origin_indices: list[int] = []
    connection_sources: list[str] = []
    diagnostics = _empty_edge_diagnostics()

    flat_energy_map = np.asarray(energy_map, dtype=np.float32).ravel(order="F")
    if original_scale_image is not None:
        flat_scale_image = np.asarray(original_scale_image, dtype=np.int16).ravel(order="F")
    else:
        flat_scale_image = None

    for (start_vertex_index, end_vertex_index), (half_1, half_2) in zip(edge_pairs, edge_halves):
        if end_vertex_index == number_of_vertices + 1:
            continue
        trace, energy_trace, scale_trace = _matlab_global_watershed_finalize_edge_trace(
            half_1,
            half_2,
            shape=shape,
            energy_map=flat_energy_map,
            scale_image=flat_scale_image,
        )
        traces.append(trace)
        connections.append([start_vertex_index - 1, end_vertex_index - 1])
        metrics.append(_edge_metric_from_energy_trace(energy_trace))
        energy_traces.append(energy_trace)
        scale_traces.append(scale_trace)
        origin_indices.append(start_vertex_index - 1)
        connection_sources.append("global_watershed")

    diagnostics["candidate_traced_edge_count"] = len(traces)
    diagnostics["terminal_edge_count"] = len(traces)
    diagnostics["terminal_direct_hit_count"] = len(traces)
    diagnostics["frontier_origins_with_candidates"] = len(set(origin_indices))
    diagnostics["frontier_origins_without_candidates"] = len(vertex_positions) - len(
        set(origin_indices)
    )
    diagnostics["frontier_per_origin_candidate_counts"] = {
        str(origin_index): origin_indices.count(origin_index)
        for origin_index in sorted(set(origin_indices))
    }
    raw_pointer_map = np.asarray(pointer_map)
    scaled_pointer_map = _matlab_global_watershed_scale_pointer_map(
        raw_pointer_map,
        size_map,
        lumen_radius_microns=lumen_radius_microns,
        microns_per_voxel=microns_per_voxel,
        step_size_per_origin_radius=step_size_per_origin_radius,
    )
    return {
        "traces": traces,
        "connections": np.asarray(connections, dtype=np.int32).reshape(-1, 2),
        "metrics": np.asarray(metrics, dtype=np.float32),
        "energy_traces": energy_traces,
        "scale_traces": scale_traces,
        "origin_indices": np.asarray(origin_indices, dtype=np.int32),
        "connection_sources": connection_sources,
        "diagnostics": diagnostics,
        "matlab_global_watershed_exact": True,
        "candidate_source": "global_watershed",
        "energy_map": np.asarray(energy_map, dtype=np.float32),
        "vertex_index_map": np.asarray(vertex_index_map, dtype=np.uint32),
        "pointer_map": scaled_pointer_map,
        "raw_pointer_map": raw_pointer_map,
        "d_over_r_map": np.asarray(d_over_r_map, dtype=np.float64),
        "branch_order_map": np.asarray(branch_order_map, dtype=np.uint8),
    }
