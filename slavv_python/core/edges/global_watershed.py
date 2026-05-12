"""Exact-MATLAB shared-state helpers for the global watershed discovery route."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from scipy import sparse

if TYPE_CHECKING:

    from slavv_python.core.edges.common import (
        BoolArray,
        Float32Array,
        Float64Array,
        Int16Array,
        Int32Array,
        Int64Array,
    )
else:
    Int16Array = np.ndarray
    Int32Array = np.ndarray
    Int64Array = np.ndarray
    Float32Array = np.ndarray
    Float64Array = np.ndarray
    BoolArray = np.ndarray

from slavv_python.core.edges.common import (
    _build_matlab_global_watershed_lut,
    _coord_to_matlab_linear_index,
    _matlab_frontier_adjusted_neighbor_energies,
    _matlab_linear_index_to_coord,
)
from slavv_python.core.edges.execution_tracing import ExecutionTracer, NullExecutionTracer
from slavv_python.core.edges.payloads import _edge_metric_from_energy_trace, _empty_edge_diagnostics


def _coords_from_linear_trace(
    linear_trace: list[int],
    shape: tuple[int, int, int],
) -> np.ndarray:
    """Return a (N, 3) coordinate array from a list of linear indices."""
    if not linear_trace:
        return np.zeros((0, 3), dtype=np.float32)
    coords = np.asarray(
        [_matlab_linear_index_to_coord(int(idx), shape) for idx in linear_trace],
        dtype=np.float32,
    )
    return cast("np.ndarray", coords)


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
    unique_lengths = np.zeros(len(lumen_radius_microns), dtype=np.float32)
    for i in range(len(lumen_radius_microns)):
        unique_lengths[i] = len(
            _build_matlab_global_watershed_lut(
                i,
                size_of_image=pointer_map.shape,
                lumen_radius_microns=lumen_radius_microns,
                microns_per_voxel=microns_per_voxel,
                step_size_per_origin_radius=step_size_per_origin_radius,
            )["linear_offsets"]
        )
    strel_lengths = unique_lengths[scale_indices]
    scaled_pointer_map[pointer_mask] = (
        1000.0 / np.maximum(strel_lengths, 1.0) * pointer_map[pointer_mask].astype(np.float32)
    )
    return cast("np.ndarray", scaled_pointer_map)


def _matlab_global_watershed_border_locations(shape: tuple[int, int, int]) -> Int64Array:
    """Return MATLAB-order linear indices for the image border at ``strel_apothem = 1``."""
    border_mask: np.ndarray = np.zeros(shape, dtype=bool)
    border_mask[0, :, :] = True
    border_mask[shape[0] - 1, :, :] = True
    border_mask[:, 0, :] = True
    border_mask[:, shape[1] - 1, :] = True
    border_mask[:, :, 0] = True
    border_mask[:, :, shape[2] - 1] = True
    return cast("Int64Array", np.flatnonzero(border_mask.ravel(order="F")).astype(np.int64))


def _initialize_matlab_global_watershed_state(
    energy: Float32Array,
    vertex_positions: Float32Array,
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

    branch_order_map = np.zeros(shape, dtype=np.uint8, order="F")
    d_over_r_map = np.zeros(shape, dtype=np.float64, order="F")
    pointer_map = np.zeros(shape, dtype=np.uint64, order="F")
    vertex_index_map = np.zeros(shape, dtype=np.uint32, order="F")

    # Apply border labels first
    for linear_index in border_locations:
        coord = _matlab_linear_index_to_coord(int(linear_index), shape)
        vertex_index_map[coord[0], coord[1], coord[2]] = np.uint32(number_of_vertices + 1)

    # Vertices take precedence over border
    vertex_energies = np.empty((number_of_vertices,), dtype=np.float32)
    for vertex_offset, linear_index in enumerate(vertex_locations):
        coord = _matlab_linear_index_to_coord(int(linear_index), shape)
        vertex_index_map[coord[0], coord[1], coord[2]] = np.uint32(vertex_offset + 1)
        vertex_energies[vertex_offset] = np.float32(energy[coord[0], coord[1], coord[2]])

    energy_map_temp = np.array(energy, dtype=np.float32, order="F", copy=True)
    for linear_index in vertex_locations:
        coord = _matlab_linear_index_to_coord(int(linear_index), shape)
        energy_map_temp[coord[0], coord[1], coord[2]] = np.float32(-np.inf)

    available_locations = vertex_locations.astype(np.int64, copy=False)[::-1]
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


def _matlab_global_watershed_prepare_size_map(
    shape: tuple[int, int, int],
    scale_indices: Int16Array | None,
    vertex_positions: Float32Array,
    vertex_scales: Int32Array,
    lumen_radius_microns: Float32Array,
) -> tuple[Int16Array, Int16Array]:
    """Build the scale-aware size_map and original_scale_image."""
    original_scale_image: Int16Array
    if scale_indices is None:
        size_map = np.ones(shape, dtype=np.int16, order="F")
        original_scale_image = np.zeros(shape, dtype=np.int16, order="F")
    else:
        original_scale_image = np.asarray(scale_indices, dtype=np.int16, order="F")
        size_map = np.asarray(original_scale_image, dtype=np.int16, order="F").copy()
        size_map += np.int16(1)
        # CRITICAL FIX: Clip size_map to valid range to prevent out-of-range scale labels
        size_map = np.clip(size_map, 1, len(lumen_radius_microns))
        # Ensure F-contiguity for persisting writes
        size_map = np.asfortranarray(size_map)

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

    return size_map, original_scale_image


def _matlab_global_watershed_current_strel(
    current_linear: int,
    *,
    current_scale_label: int,
    shape: tuple[int, int, int],
    lumen_radius_microns: Float32Array,
    microns_per_voxel: Float32Array,
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
    linear_offsets_full = np.asarray(lut["linear_offsets"], dtype=np.int64)

    # Verify LUT consistency
    assert len(offsets) == len(linear_offsets_full), (
        f"LUT inconsistency: local_subscripts has {len(offsets)} elements "
        f"but linear_offsets has {len(linear_offsets_full)} elements"
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
    valid_linear_raw = linear_offsets_full[valid_mask] + np.int64(current_linear)

    img_size = shape[0] * shape[1] * shape[2]
    linear_valid_mask = (valid_linear_raw >= 0) & (valid_linear_raw < img_size)
    if not np.all(linear_valid_mask):
        bad_linear = valid_linear_raw[~linear_valid_mask][:5].tolist()
        raise AssertionError(
            "Global watershed produced out-of-bounds linear indices for one in-bounds strel: "
            f"current={current_linear}, scale={current_scale_label}, sample={bad_linear}"
        )
    valid_linear = valid_linear_raw

    # Pointer indices are 1-based indices into the FULL LUT (before filtering)
    # Corrected: Use the back-pointing indices from the LUT to ensure traces go to the center.
    pointer_indices = np.asarray(lut["pointer_indices"], dtype=np.uint64)[valid_mask]

    if not (np.all(pointer_indices >= 1) and np.all(pointer_indices <= len(offsets))):
        raise AssertionError(
            f"Invalid pointer indices: min={np.min(pointer_indices)}, "
            f"max={np.max(pointer_indices)}, LUT size={len(offsets)}"
        )

    return {
        "current_coord": current_coord.astype(np.int32, copy=False),
        "coords": valid_coords,
        "offsets": valid_offsets,
        "linear_indices": valid_linear,
        "pointer_indices": pointer_indices,
        "r_over_R": np.asarray(lut["r_over_R"], dtype=np.float32)[valid_mask],
        "distance_microns": np.asarray(lut["distance_lut"], dtype=np.float32)[valid_mask],
        "unit_vectors": np.asarray(lut["unit_vectors"], dtype=np.float32)[valid_mask],
        "lut_size": len(offsets),  # Store for debugging
        "scale_label_clipped": current_scale_index
        + 1,  # Clipped scale for consistent pointer/size_map usage
    }


def _matlab_global_watershed_reveal_unclaimed_strel(
    *,
    current_vertex_index: int,
    current_scale_label: int,
    current_d_over_r: float,
    valid_linear: Int64Array,
    strel_pointer_indices: Int64Array,
    strel_r_over_R: Float32Array,
    adjusted_energies: Float32Array,
    vertex_index_map_flat: Int32Array,
    pointer_map_flat: Int64Array,
    energy_map_flat: Float32Array,
    d_over_r_map_flat: Float64Array,
    size_map_flat: Int16Array,
    lut_size: int,
) -> dict[str, Any]:
    """Reveal one MATLAB strel into the shared maps, claiming only previously unowned voxels.

    CRITICAL: Only write to locations that have vertex_index == 0 AND pointer_map == 0 to prevent
    overwriting existing pointers and creating cycles.

    CRITICAL: The pointer indices in strel_pointer_indices are 1-based indices into the
    FULL LUT for current_scale_label. They must be in range [1, lut_size]. The size_map
    is also written with current_scale_label so that during backtracking, we can reconstruct
    the correct LUT and interpret the pointer correctly.

    NOTE: This function updates the energy map with penalized energies. MATLAB uses
    these penalized values for the final edge energy traces.
    """
    if len(strel_pointer_indices) != len(valid_linear):
        raise AssertionError(
            "Global watershed strel arrays must stay aligned: "
            f"{len(strel_pointer_indices)} pointers for {len(valid_linear)} locations"
        )

    vertices_of_current_strel = np.asarray(vertex_index_map_flat[valid_linear], dtype=np.uint32)
    is_without_vertex = (vertices_of_current_strel == 0) & (pointer_map_flat[valid_linear] == 0)
    if np.any(is_without_vertex):
        claim_linear = valid_linear[is_without_vertex]
        claim_pointers = np.asarray(strel_pointer_indices[is_without_vertex], dtype=np.uint64)
        if np.any(claim_pointers < 1) or np.any(claim_pointers > lut_size):
            bad_pointers = claim_pointers[(claim_pointers < 1) | (claim_pointers > lut_size)]
            raise AssertionError(
                "Global watershed produced invalid claim pointers for one strel: "
                f"scale={current_scale_label}, lut_size={lut_size}, "
                f"sample={bad_pointers[:5].tolist()}"
            )
        if np.any(pointer_map_flat[claim_linear] != 0):
            raise AssertionError(
                "Global watershed attempted to overwrite an existing pointer on the exact route."
            )
        vertex_index_map_flat[claim_linear] = np.uint32(current_vertex_index)
        pointer_map_flat[claim_linear] = claim_pointers
        energy_map_flat[claim_linear] = adjusted_energies[is_without_vertex]
        d_over_r_map_flat[claim_linear] = (
            np.asarray(strel_r_over_R[is_without_vertex], dtype=np.float32) + current_d_over_r
        )
        size_map_flat[claim_linear] = np.int16(current_scale_label)

    return {
        "vertices_of_current_strel": vertices_of_current_strel,
        "is_without_vertex_in_strel": is_without_vertex,
    }


def _matlab_global_watershed_insert_available_location(
    available_locations: list[int],
    *,
    next_location: int,
    next_energy: float,
    energy_lookup: Float32Array,
    seed_idx: int,
    is_current_location_clear: bool,
) -> tuple[list[int], bool]:
    """Insert one location into MATLAB's worst-to-best available-location list.

    CRITICAL: available_locations is sorted by energy descending from high (worst) at
    index 0 to low/best (-Inf) at the end. Elements are popped from the right (best).

    Exact replica of MATLAB get_edges_by_watershed.m lines 532-562.

    MATLAB list convention:
      - Index 0 (MATLAB: 1) = worst energy (largest value).
      - Index -1 (MATLAB: end) = best energy (most negative / -Inf for vertices).
      - Elements are popped from the right (best first).
      - When is_current_location_clear is False, the LAST element is the active
        node (unsorted relative to the rest). MATLAB's splice ALWAYS removes this
        slot, so the current node is always popped.

    seed_idx == 1 (MATLAB lines 532-541):
        Fast path: if available_locations[0] <= target -> location_idx = 1.
        Otherwise: linear scan from the END across ALL elements (including the
        unsorted current at [-1]) to find the last element with energy > target.

    seed_idx > 1 (MATLAB lines 542-553):
        Fast path checks available_locations[-1] (the current / unsorted node):
            if end_energy >= target and ~is_current_location_clear:
                location_idx = n      (inserts before current, which is then dropped)
            if end_energy >= target and  is_current_location_clear:
                location_idx = n + 1  (append at end)
        Otherwise: linear scan across ALL elements for first element < target.

    MATLAB splice (lines 558-562) always pops the 'end' slot when ~clear,
    so was_current_popped is always True.

    Returns:
        (updated_available_locations, was_current_popped)
    """

    def _energy_for(linear_index: int) -> float:
        return float(energy_lookup[int(linear_index)])

    if not available_locations:
        return [int(next_location)], True

    target_energy = float(next_energy)
    n = len(available_locations)

    # 1. Identify the reliably sorted boundary
    if not is_current_location_clear:
        sorted_len = n - 1
        energy_last = _energy_for(available_locations[-1])
    else:
        sorted_len = n
        energy_last = None

    should_pop_current = not is_current_location_clear
    insert_at = -1
    append_at_end = False

    if seed_idx == 1:
        # MATLAB: location_idx = 1 + find(energy > target, 1, 'last')
        if sorted_len > 0 and _energy_for(available_locations[0]) <= target_energy:
            insert_at = 0
        else:
            # MATLAB scans from the 'end'. If unsorted active node qualifies, it takes priority.
            # This results in location_idx = n+1, causing an append with the current node kept.
            if (
                not is_current_location_clear
                and energy_last is not None
                and energy_last > target_energy
            ):
                # QUIRK: active node stays in list (MATLAB location_idx = n+1 after splice).
                append_at_end = True
                should_pop_current = False
            else:
                low = 0
                high = sorted_len
                while low < high:
                    mid = (low + high) // 2
                    if _energy_for(available_locations[mid]) > target_energy:
                        low = mid + 1
                    else:
                        high = mid
                insert_at = low
    else:
        # MATLAB: find(energy < target, 1, 'first')
        # Fast-path checks available_locations[-1] (the current unsorted node).
        search_end_val = (
            _energy_for(available_locations[-1]) if is_current_location_clear else energy_last
        )
        if search_end_val >= target_energy:
            if not is_current_location_clear:
                insert_at = n - 1  # insert before current slot
            else:
                append_at_end = True
        else:
            low = 0
            high = sorted_len
            while low < high:
                mid = (low + high) // 2
                if _energy_for(available_locations[mid]) >= target_energy:
                    low = mid + 1
                else:
                    high = mid
            if not is_current_location_clear and low == sorted_len:
                insert_at = n - 1
            else:
                insert_at = low

    # Apply insertion
    if should_pop_current:
        available_locations.pop()
        if append_at_end:
            available_locations.append(int(next_location))
        else:
            available_locations.insert(insert_at, int(next_location))
        return available_locations, True
    # QUIRK: active node stays in list; but MATLAB sets is_current_location_clear=true,
    # so we return True to prevent the safety pop from removing the current node.
    if append_at_end:
        available_locations.append(int(next_location))
    else:
        available_locations.insert(insert_at, int(next_location))
    return available_locations, True


def _matlab_global_watershed_reset_join_locations(
    available_locations: list[int],
    *,
    next_vertex_locations: Int64Array,
    is_current_location_clear: bool,
) -> tuple[list[int], bool]:
    """Mirror MATLAB's indexed available-location removal during watershed joins.

    Returns the updated list and the current-location cleared flag.
    """
    if not available_locations:
        return [], is_current_location_clear

    updated_locations = list(available_locations)
    locations_to_reset = sorted(
        {
            int(value) for value in np.asarray(next_vertex_locations, dtype=np.int64).tolist()
        }.intersection(updated_locations)
    )

    if not is_current_location_clear:
        current_location = updated_locations[-1]
        is_current_location_clear = True
        updated_locations = updated_locations[:-1]
        locations_to_reset = [value for value in locations_to_reset if value != current_location]

    if not locations_to_reset:
        return updated_locations, is_current_location_clear

    indices_to_remove: list[int] = []
    for location in locations_to_reset:
        try:
            indices_to_remove.append(updated_locations.index(location))
        except ValueError:
            continue

    for index in sorted(set(indices_to_remove), reverse=True):
        del updated_locations[index]

    return updated_locations, is_current_location_clear


def _matlab_global_watershed_unit_vectors(
    offsets: Int32Array,
    microns_per_voxel: Float32Array,
) -> Float32Array:
    """Return MATLAB-style unit vectors for one local strel."""
    vectors: np.ndarray = np.asarray(offsets, dtype=np.float64) * np.asarray(
        microns_per_voxel,
        dtype=np.float64,
    )
    norms = np.linalg.norm(vectors, axis=1)
    unit_vectors: np.ndarray = np.zeros_like(vectors, dtype=np.float64)
    valid_mask = norms > 1e-12
    unit_vectors[valid_mask] = vectors[valid_mask] / norms[valid_mask, None]
    return cast("Float32Array", unit_vectors.astype(np.float32, copy=False))


def _matlab_global_watershed_tolerance_mask(
    adjusted_energies: Float64Array,
    *,
    current_vertex_energy: float,
    energy_tolerance: float,
) -> BoolArray:
    """Mirror MATLAB's per-seed energy tolerance test on the current penalized strel energies."""
    threshold = float(current_vertex_energy) * (1.0 - float(energy_tolerance))
    return cast("BoolArray", np.asarray(adjusted_energies, dtype=np.float32) < threshold)


def _matlab_global_watershed_seed_index_range(
    *,
    current_pointer_value: int,
    edge_number_tolerance: int,
) -> range:
    """Mirror MATLAB's seed count: only true origins emit multiple seeds."""
    if int(current_pointer_value) == 0:
        return range(1, int(edge_number_tolerance) + 1)
    return range(1, 2)


def _matlab_global_watershed_trace_half(
    start_linear: int,
    *,
    pointer_map: Int64Array,
    size_map: Int16Array,
    shape: tuple[int, int, int],
    lumen_radius_microns: Float32Array,
    microns_per_voxel: Float32Array,
    step_size_per_origin_radius: float,
) -> list[int]:
    """Trace one-half of a watershed edge from endpoint back to origin."""
    trace: list[int] = [int(start_linear)]
    current_linear = int(start_linear)
    seen: set[int] = {current_linear}

    while True:
        pointer_value = int(pointer_map.ravel(order="F")[current_linear])
        if pointer_value == 0:
            break

        current_scale_label = int(size_map.ravel(order="F")[current_linear])
        current_scale_index = int(
            np.clip(current_scale_label - 1, 0, len(lumen_radius_microns) - 1)
        )
        lut = _build_matlab_global_watershed_lut(
            current_scale_index,
            size_of_image=shape,
            lumen_radius_microns=lumen_radius_microns,
            microns_per_voxel=microns_per_voxel,
            step_size_per_origin_radius=step_size_per_origin_radius,
        )
        linear_offsets = np.asarray(lut["linear_offsets"], dtype=np.int64)

        if pointer_value > len(linear_offsets):
            break

        next_linear = int(current_linear - linear_offsets[pointer_value - 1])
        if next_linear in seen:
            # Prevent infinite loops from cycles
            break
        trace.append(next_linear)
        current_linear = next_linear
        seen.add(current_linear)

    return trace


def _matlab_global_watershed_assemble_results(
    *,
    edge_pairs: list[tuple[int, int]],
    edge_halves: list[tuple[list[int], list[int]]],
    shape: tuple[int, int, int],
    energy_map: Float32Array,
    original_scale_image: Int16Array | None,
    vertex_positions: Float32Array,
    vertex_index_map: Int32Array,
    pointer_map: Int64Array,
    size_map: Int16Array,
    d_over_r_map: Float64Array,
    branch_order_map: Int16Array,
    lumen_radius_microns: Float32Array,
    microns_per_voxel: Float32Array,
    step_size_per_origin_radius: float,
) -> dict[str, Any]:
    """Finalize candidate traces, calculate metrics, and build the return payload."""
    number_of_vertices = len(vertex_positions)
    traces: list[np.ndarray] = []
    connections: list[list[int]] = []
    metrics: list[float] = []
    energy_traces: list[np.ndarray] = []
    scale_traces: list[np.ndarray] = []
    origin_indices: list[int] = []
    connection_sources: list[str] = []
    diagnostics = _empty_edge_diagnostics()

    flat_energy_map = energy_map.ravel(order="F")
    flat_scale_image = (
        original_scale_image.ravel(order="F") if original_scale_image is not None else None
    )

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


def _generate_edge_candidates_matlab_global_watershed(
    energy: Float32Array,
    scale_indices: Int16Array | None,
    vertex_positions: Float32Array,
    vertex_scales: Int32Array,
    lumen_radius_microns: Float32Array,
    microns_per_voxel: Float32Array,
    _vertex_center_image: np.ndarray,
    params: dict[str, Any],
    *,
    heartbeat: Any | None = None,
    tracer: ExecutionTracer | None = None,
) -> dict[str, Any]:
    """Generate candidates with MATLAB's one-pass global shared-state watershed search."""
    del _vertex_center_image
    active_tracer = tracer or NullExecutionTracer()
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
    vertex_locations = cast("Int64Array", state["vertex_locations"])
    vertex_energies = cast("Float32Array", state["vertex_energies"])
    energy_map_temp = cast("Float32Array", state["energy_map_temp"])
    branch_order_map = cast("Int16Array", state["branch_order_map"])
    d_over_r_map = cast("Float64Array", state["d_over_r_map"])
    pointer_map = cast("Int64Array", state["pointer_map"])
    vertex_index_map = cast("Int32Array", state["vertex_index_map"])
    available_locations = [int(value) for value in cast("Int64Array", state["available_locations"])]
    vertex_adjacency_matrix = cast("Any", state["vertex_adjacency_matrix"])
    number_of_vertices = len(vertex_locations)

    energy_map = np.array(energy_map_raw, dtype=np.float32, order="F", copy=True)
    size_map, original_scale_image = _matlab_global_watershed_prepare_size_map(
        shape, scale_indices, vertex_positions, vertex_scales, lumen_radius_microns
    )

    # MATLAB parity parameters
    edge_number_tolerance = int(
        params.get("edge_number_tolerance", params.get("number_of_edges_per_vertex", 4))
    )
    energy_tolerance = float(params.get("energy_tolerance", 1.0))
    radius_tolerance = float(params.get("radius_tolerance", 0.5))
    step_size_per_origin_radius = float(params.get("step_size_per_origin_radius", 1.0))
    distance_tolerance = float(
        params.get("distance_tolerance", params.get("distance_tolerance_per_origin_radius", 3.0))
    )

    energy_map_temp_flat = energy_map_temp.ravel(order="F")
    d_over_r_map_flat = d_over_r_map.ravel(order="F")
    pointer_map_flat = pointer_map.ravel(order="F")
    vertex_index_map_flat = vertex_index_map.ravel(order="F")
    size_map_flat = size_map.ravel(order="F")
    branch_order_map_flat = branch_order_map.ravel(order="F")

    edge_halves: list[tuple[list[int], list[int]]] = []
    edge_pairs: list[tuple[int, int]] = []

    last_heartbeat_at = time.monotonic()

    # Track the raw vertex energies for resetting -inf values
    vertex_energies_raw_flat = energy_map_raw.ravel(order="F")

    iteration = 0
    while available_locations:
        iteration += 1
        current_linear = int(available_locations[-1])
        is_current_location_clear = False

        current_energy = float(energy_map_temp_flat[current_linear])
        active_tracer.on_iteration_start(iteration, current_linear, current_energy)

        if float(current_energy) == float("-inf"):
            current_energy = float(vertex_energies_raw_flat[current_linear])
            energy_map_temp_flat[current_linear] = current_energy

        if current_energy >= 0.0:
            available_locations.pop()
            continue

        current_vertex_index = int(vertex_index_map_flat[current_linear])
        current_scale_label = int(size_map_flat[current_linear])
        current_scale_index = int(
            np.clip(current_scale_label - 1, 0, len(lumen_radius_microns) - 1)
        )
        current_pointer_value = int(pointer_map_flat[current_linear])

        current_strel = _matlab_global_watershed_current_strel(
            current_linear,
            current_scale_label=current_scale_label,
            shape=shape,
            lumen_radius_microns=lumen_radius_microns,
            microns_per_voxel=microns_per_voxel,
            step_size_per_origin_radius=step_size_per_origin_radius,
        )
        # Extract clipped scale for consistent pointer creation and size_map writing
        current_scale_label_for_writing = current_strel.get(
            "scale_label_clipped", current_scale_label
        )

        current_strel_r_over_R = cast("Float32Array", current_strel["r_over_R"])
        current_strel_coords = cast("Int32Array", current_strel["coords"])
        current_strel_linear = cast("Int64Array", current_strel["linear_indices"])
        current_strel_offsets = cast("Int32Array", current_strel["offsets"])
        current_strel_pointer_indices = cast("Int64Array", current_strel["pointer_indices"])

        current_strel_energies = energy_map_temp_flat[current_strel_linear].astype(
            np.float32, copy=False
        )
        current_d_over_r = float(d_over_r_map_flat[current_linear])
        current_forward_unit: np.ndarray | None = None
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

        adjusted = _matlab_frontier_adjusted_neighbor_energies(
            current_strel_energies,
            neighbor_offsets=current_strel_offsets,
            neighbor_r_over_R=current_strel_r_over_R,
            neighbor_scale_indices=size_map_flat[current_strel_linear],
            propagated_scale_index=current_scale_label,
            current_d_over_r=current_d_over_r,
            origin_radius_microns=max(float(lumen_radius_microns[current_scale_index]), 1e-6),
            current_forward_unit=current_forward_unit,
            microns_per_voxel=microns_per_voxel,
            lumen_radius_microns=lumen_radius_microns,
            radius_tolerance=radius_tolerance,
            distance_tolerance=distance_tolerance,
        )

        # Sample ownership BEFORE claiming to determine growability/connectivity
        vertices_of_current_strel = vertex_index_map_flat[current_strel_linear]

        _matlab_global_watershed_reveal_unclaimed_strel(
            current_vertex_index=current_vertex_index,
            current_scale_label=current_scale_label_for_writing,
            current_d_over_r=current_d_over_r,
            valid_linear=current_strel_linear,
            strel_pointer_indices=current_strel_pointer_indices,
            strel_r_over_R=current_strel_r_over_R,
            adjusted_energies=adjusted,
            vertex_index_map_flat=vertex_index_map_flat,
            pointer_map_flat=pointer_map_flat,
            energy_map_flat=energy_map.ravel(order="F"),
            d_over_r_map_flat=d_over_r_map_flat,
            size_map_flat=size_map_flat,
            lut_size=current_strel["lut_size"],
        )

        for seed_idx in _matlab_global_watershed_seed_index_range(
            current_pointer_value=current_pointer_value,
            edge_number_tolerance=edge_number_tolerance,
        ):
            # MATLAB recomputes the tolerated-energy mask each seed after directional suppression
            # mutates the current strel energies.
            is_energy_tolerated_in_strel = _matlab_global_watershed_tolerance_mask(
                adjusted,
                current_vertex_energy=float(vertex_energies[current_vertex_index - 1]),
                energy_tolerance=energy_tolerance,
            )
            strel_idx = int(np.argmin(adjusted))
            next_location = int(current_strel_linear[strel_idx])
            next_vertex_index = int(vertices_of_current_strel[strel_idx])
            active_tracer.on_seed_selected(seed_idx, next_location, float(adjusted[strel_idx]))

            if not bool(is_energy_tolerated_in_strel[strel_idx]):
                if not is_current_location_clear:
                    available_locations.pop()
                    is_current_location_clear = True
            else:
                if next_vertex_index == 0:
                    branch_order = int(branch_order_map_flat[current_linear]) + seed_idx - 1
                    branch_order_map_flat[next_location] = np.uint8(branch_order)
                    if branch_order < edge_number_tolerance:
                        available_locations, _was_popped = (
                            _matlab_global_watershed_insert_available_location(
                                available_locations,
                                next_location=next_location,
                                next_energy=float(energy_map_temp_flat[next_location]),
                                energy_lookup=energy_map_temp_flat,
                                seed_idx=seed_idx,
                                is_current_location_clear=is_current_location_clear,
                            )
                        )
                        if _was_popped:
                            is_current_location_clear = True
                else:
                    is_next_vertex_in_strel = vertices_of_current_strel == next_vertex_index
                    (
                        available_locations,
                        is_current_location_clear,
                    ) = _matlab_global_watershed_reset_join_locations(
                        available_locations,
                        next_vertex_locations=current_strel_linear[is_next_vertex_in_strel],
                        is_current_location_clear=is_current_location_clear,
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
                            current_linear,
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
                        active_tracer.on_join(
                            int(current_vertex_index),
                            int(next_vertex_index),
                            half_1,
                            half_2,
                        )

            # MATLAB unconditionally applies suppression to mutate the strel landscape
            # before the next seed is selected.

            from slavv_python.core.edges.common import (
                _matlab_frontier_directional_suppression_factors,
            )

            adjusted *= _matlab_frontier_directional_suppression_factors(
                current_strel_offsets,
                selected_index=strel_idx,
                microns_per_voxel=microns_per_voxel,
            )

        if not is_current_location_clear:
            available_locations.pop()
            is_current_location_clear = True

        if heartbeat is not None:
            now = time.monotonic()
            if iteration == 1 or iteration % 512 == 0 or (now - last_heartbeat_at) >= 5.0:
                heartbeat(iteration, len(edge_pairs))
                last_heartbeat_at = now

    return _matlab_global_watershed_assemble_results(
        edge_pairs=edge_pairs,
        edge_halves=edge_halves,
        shape=shape,
        energy_map=energy_map,
        original_scale_image=original_scale_image,
        vertex_positions=vertex_positions,
        vertex_index_map=vertex_index_map,
        pointer_map=pointer_map,
        size_map=size_map,
        d_over_r_map=d_over_r_map,
        branch_order_map=branch_order_map,
        lumen_radius_microns=lumen_radius_microns,
        microns_per_voxel=microns_per_voxel,
        step_size_per_origin_radius=step_size_per_origin_radius,
    )
