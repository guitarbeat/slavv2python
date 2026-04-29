"""Exact-MATLAB shared-state helpers for the global watershed edge discovery port."""

from __future__ import annotations

import heapq
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

    branch_order_map = np.zeros(shape, dtype=np.uint8, order="F")
    d_over_r_map = np.zeros(shape, dtype=np.float64, order="F")
    pointer_map = np.zeros(shape, dtype=np.uint64, order="F")
    vertex_index_map = np.zeros(shape, dtype=np.uint32, order="F")

    vertex_energies = np.empty((number_of_vertices,), dtype=np.float32)
    for vertex_offset, linear_index in enumerate(vertex_locations):
        coord = _matlab_linear_index_to_coord(int(linear_index), shape)
        vertex_index_map[coord[0], coord[1], coord[2]] = np.uint32(vertex_offset + 1)
        vertex_energies[vertex_offset] = np.float32(energy[coord[0], coord[1], coord[2]])

    for linear_index in border_locations:
        coord = _matlab_linear_index_to_coord(int(linear_index), shape)
        vertex_index_map[coord[0], coord[1], coord[2]] = np.uint32(number_of_vertices + 1)

    energy_map_temp = np.asarray(energy, dtype=np.float32, order="F").copy()
    for vertex_offset, linear_index in enumerate(vertex_locations):
        coord = _matlab_linear_index_to_coord(int(linear_index), shape)
        energy_map_temp[coord[0], coord[1], coord[2]] = float(vertex_energies[vertex_offset])

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

    # Additional bounds check for linear indices
    img_size = shape[0] * shape[1] * shape[2]
    linear_valid_mask = (valid_linear_raw >= 0) & (valid_linear_raw < img_size)
    if not np.all(linear_valid_mask):
        import logging

        logging.error(
            f"Linear index out of bounds at current={current_linear}, scale={current_scale_label}: "
            f"{np.sum(~linear_valid_mask)} indices out of range [0, {img_size})"
        )
        # Filter again
        valid_coords = valid_coords[linear_valid_mask]
        valid_offsets = valid_offsets[linear_valid_mask]
        valid_linear = valid_linear_raw[linear_valid_mask]
        # Update valid_mask to reflect the additional filtering
        temp_mask = valid_mask.copy()
        temp_mask[valid_mask] = linear_valid_mask
        valid_mask = temp_mask
    else:
        valid_linear = valid_linear_raw

    # Pointer indices are 1-based indices into the FULL LUT (before filtering)
    # These must be in range [1, len(offsets)]
    pointer_indices = np.arange(1, len(offsets) + 1, dtype=np.uint64)[valid_mask]

    # Verify all pointer indices are valid
    if not (np.all(pointer_indices >= 1) and np.all(pointer_indices <= len(offsets))):
        import logging

        logging.error(
            f"ASSERTION FAILED: Invalid pointer indices at scale {current_scale_label}: "
            f"min={np.min(pointer_indices)}, max={np.max(pointer_indices)}, LUT size={len(offsets)}"
        )
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
    valid_linear: np.ndarray,
    strel_pointer_indices: np.ndarray,
    strel_distance_microns: np.ndarray,
    strel_adjusted_energies: np.ndarray,
    vertex_index_map_flat: np.ndarray,
    energy_map_flat: np.ndarray,
    pointer_map_flat: np.ndarray,
    d_over_r_map_flat: np.ndarray,
    size_map_flat: np.ndarray,
    lut_size: int,  # Add this parameter for validation
    filtered_pointer_stats: dict[str, int] | None = None,  # Track filtering stats
) -> dict[str, np.ndarray]:
    """Reveal one MATLAB strel into the shared maps, claiming only previously unowned voxels.

    CRITICAL: Only write to locations that have vertex_index == 0 AND pointer_map == 0 to prevent
    overwriting existing pointers and creating cycles.

    CRITICAL: The pointer indices in strel_pointer_indices are 1-based indices into the
    FULL LUT for current_scale_label. They must be in range [1, lut_size]. The size_map
    is also written with current_scale_label so that during backtracking, we can reconstruct
    the correct LUT and interpret the pointer correctly.
    """
    # DIAGNOSTIC: Log when processing location 12532290's neighbors
    if 12532290 in valid_linear:
        import logging
        idx_12532290 = np.where(valid_linear == 12532290)[0][0]
        current_vertex_at_12532290 = int(vertex_index_map_flat[12532290])
        current_pointer_at_12532290 = int(pointer_map_flat[12532290])
        current_scale_at_12532290 = int(size_map_flat[12532290])
        logging.error(
            f"REVEAL CALLED WITH 12532290 IN STREL: "
            f"current_vertex_index={current_vertex_index}, "
            f"current_scale_label={current_scale_label}, "
            f"existing_vertex_at_12532290={current_vertex_at_12532290}, "
            f"existing_pointer_at_12532290={current_pointer_at_12532290}, "
            f"existing_scale_at_12532290={current_scale_at_12532290}, "
            f"will_write_pointer={strel_pointer_indices[idx_12532290]}"
        )
    
    is_without_vertex = (vertex_index_map_flat[valid_linear] == 0) & (pointer_map_flat[valid_linear] == 0)
    if np.any(is_without_vertex):
        claim_linear = valid_linear[is_without_vertex]
        claim_pointers = np.asarray(strel_pointer_indices[is_without_vertex], dtype=np.uint64)

        # Sanity check: verify array lengths match
        if len(strel_pointer_indices) != len(valid_linear):
            import logging

            logging.error(
                f"CRITICAL: Array length mismatch! strel_pointer_indices has {len(strel_pointer_indices)} elements "
                f"but valid_linear has {len(valid_linear)} elements at scale {current_scale_label}"
            )
            return {
                "vertices_of_current_strel": vertex_index_map_flat[valid_linear],
                "is_without_vertex_in_strel": is_without_vertex,
            }

        # Validate pointer indices before writing - they should match the LUT size for current_scale_label
        if np.any(claim_pointers < 1) or np.any(claim_pointers > lut_size):
            import logging

            bad_pointers = claim_pointers[(claim_pointers < 1) | (claim_pointers > lut_size)]
            bad_locations = claim_linear[(claim_pointers < 1) | (claim_pointers > lut_size)]
            logging.error(
                f"CRITICAL BUG: Attempting to write {len(bad_pointers)} invalid pointers at scale {current_scale_label}. "
                f"Pointer range [{np.min(bad_pointers)}, {np.max(bad_pointers)}] vs LUT size {lut_size}. "
                f"This indicates a bug in pointer index generation! "
                f"Sample bad locations: {bad_locations[:5].tolist()}"
            )
            if filtered_pointer_stats is not None:
                filtered_pointer_stats["total_filtered"] = filtered_pointer_stats.get(
                    "total_filtered", 0
                ) + len(bad_pointers)
                filtered_pointer_stats["total_attempts"] = filtered_pointer_stats.get(
                    "total_attempts", 0
                ) + len(claim_pointers)

            # Filter out invalid pointers as a defensive measure
            valid_pointer_mask = (claim_pointers >= 1) & (claim_pointers <= lut_size)
            claim_linear = claim_linear[valid_pointer_mask]
            claim_pointers = claim_pointers[valid_pointer_mask]
            is_without_vertex_filtered = is_without_vertex.copy()
            is_without_vertex_filtered[is_without_vertex] = valid_pointer_mask
            is_without_vertex = is_without_vertex_filtered

        if len(claim_linear) > 0:
            # CRITICAL FIX: Filter out locations with existing pointers to prevent overwrites
            # The is_without_vertex check should have caught these, but defensive filtering
            # or race conditions might have allowed them through
            existing_pointers = pointer_map_flat[claim_linear]
            existing_scales = size_map_flat[claim_linear]
            no_existing_pointer_mask = existing_pointers == 0
            
            if not np.all(no_existing_pointer_mask):
                import logging

                overwrite_count = np.sum(~no_existing_pointer_mask)
                overwrite_mask = ~no_existing_pointer_mask
                logging.error(
                    f"CRITICAL: Prevented overwrite of {overwrite_count} existing pointers at scale {current_scale_label}! "
                    f"This indicates a bug in the is_without_vertex logic. "
                    f"Existing pointer values: {existing_pointers[overwrite_mask][:5]}, "
                    f"Existing scales: {existing_scales[overwrite_mask][:5]}, "
                    f"Would-be new pointers: {claim_pointers[overwrite_mask][:5]}"
                )
                
                # DIAGNOSTIC: Check if location 12532290 is being overwritten
                if 12532290 in claim_linear[overwrite_mask]:
                    idx_in_overwrite = np.where(claim_linear[overwrite_mask] == 12532290)[0][0]
                    logging.error(
                        f"OVERWRITE PREVENTED FOR LOCATION 12532290: "
                        f"existing_pointer={existing_pointers[overwrite_mask][idx_in_overwrite]}, "
                        f"existing_scale={existing_scales[overwrite_mask][idx_in_overwrite]}, "
                        f"would-be new_pointer={claim_pointers[overwrite_mask][idx_in_overwrite]}, "
                        f"would-be new_scale={current_scale_label}"
                    )
                
                # Filter out locations with existing pointers
                claim_linear = claim_linear[no_existing_pointer_mask]
                claim_pointers = claim_pointers[no_existing_pointer_mask]
                
                # Update is_without_vertex to reflect the additional filtering
                is_without_vertex_no_overwrite = is_without_vertex.copy()
                is_without_vertex_no_overwrite[is_without_vertex] = no_existing_pointer_mask
                is_without_vertex = is_without_vertex_no_overwrite


            # DIAGNOSTIC: Log a sample write for debugging
            if len(claim_linear) > 0 and np.random.random() < 0.001:  # Log 0.1% of writes
                import logging

                sample_idx = 0
                logging.info(
                    f"WRITE: location={claim_linear[sample_idx]}, "
                    f"pointer={claim_pointers[sample_idx]}, "
                    f"scale={current_scale_label}, "
                    f"lut_size={lut_size}, "
                    f"vertex={current_vertex_index}"
                )
            
            # DIAGNOSTIC: Log writes to specific problematic locations - BEFORE write
            if 12532290 in claim_linear:
                import logging
                idx = np.where(claim_linear == 12532290)[0][0]
                # Read BEFORE write
                before_pointer = int(pointer_map_flat[12532290])
                before_scale = int(size_map_flat[12532290])
                before_vertex = int(vertex_index_map_flat[12532290])
                logging.error(
                    f"ABOUT TO WRITE TO LOCATION 12532290: "
                    f"will_write_pointer={claim_pointers[idx]}, will_write_scale={current_scale_label}, "
                    f"lut_size={lut_size}, will_write_vertex={current_vertex_index}"
                )
                logging.error(
                    f"READBACK BEFORE WRITE: "
                    f"current_pointer={before_pointer}, "
                    f"current_scale={before_scale}, "
                    f"current_vertex={before_vertex}"
                )
            
            # Perform the writes
            vertex_index_map_flat[claim_linear] = np.uint32(current_vertex_index)
            energy_map_flat[claim_linear] = np.asarray(
                strel_adjusted_energies[is_without_vertex], dtype=np.float32
            )
            pointer_map_flat[claim_linear] = claim_pointers
            d_over_r_map_flat[claim_linear] = (
                strel_distance_microns[is_without_vertex] + current_d_over_r
            )
            size_map_flat[claim_linear] = np.int16(current_scale_label)
            
            # DIAGNOSTIC: Read back AFTER write for location 12532290
            if 12532290 in claim_linear:
                import logging
                after_pointer = int(pointer_map_flat[12532290])
                after_scale = int(size_map_flat[12532290])
                after_vertex = int(vertex_index_map_flat[12532290])
                logging.error(
                    f"READBACK AFTER WRITE: "
                    f"pointer_map[12532290]={after_pointer}, "
                    f"size_map[12532290]={after_scale}, "
                    f"vertex_index_map[12532290]={after_vertex}"
                )

            readback_pointers = pointer_map_flat[claim_linear]
            if not np.array_equal(readback_pointers, claim_pointers):
                import logging

                mismatches = np.sum(readback_pointers != claim_pointers)
                logging.error(
                    f"CRITICAL CORRUPTION: {mismatches}/{len(claim_pointers)} pointers corrupted immediately after writing! "
                    f"Scale {current_scale_label}. "
                    f"Written: {claim_pointers[:5]}, Read back: {readback_pointers[:5]}"
                )
            
            # DIAGNOSTIC: Verify write to location 12532290
            if 12532290 in claim_linear:
                import logging
                logging.error(
                    f"READBACK AFTER WRITE: "
                    f"pointer_map[12532290]={pointer_map_flat[12532290]}, "
                    f"size_map[12532290]={size_map_flat[12532290]} (after write)"
                )

    return {
        "vertices_of_current_strel": vertex_index_map_flat[valid_linear],
        "is_without_vertex_in_strel": is_without_vertex,
    }


def _matlab_global_watershed_insert_available_location(
    *,
    next_location: int,
    next_energy: float,
    available_locations: list[tuple[float, int, int]],
    energy_map_flat: np.ndarray,
    insertion_counter: list[int],
) -> None:
    """Insert into heap with energy priority (most negative first) and LIFO tie-breaking."""
    insertion_counter[0] += 1
    heapq.heappush(
        available_locations, (float(next_energy), -insertion_counter[0], int(next_location))
    )


def _matlab_global_watershed_reset_join_locations(
    available_locations: list[tuple[float, int, int]],
    *,
    next_vertex_locations: np.ndarray,
    is_current_location_clear: bool,
    current_location: int,
) -> bool:
    """Mirror MATLAB's indexed available-location removal during watershed joins.

    Mutates available_locations in-place for efficiency.
    """
    if not available_locations and is_current_location_clear:
        return is_current_location_clear

    locations_to_reset = set(next_vertex_locations.tolist())

    if not is_current_location_clear:
        # If not clear, the 'current_location' we popped at the start of the loop
        # is conceptually still in the list and might be one of the reset targets.
        if current_location in locations_to_reset:
            locations_to_reset.remove(current_location)
        is_current_location_clear = True

    if not locations_to_reset:
        return is_current_location_clear

    # Filter the list. O(N) but better than multiple .remove() calls.
    # We filter in reverse if it helps, but simple comprehension is usually fast.
    new_list = [loc for loc in available_locations if loc[2] not in locations_to_reset]
    available_locations.clear()
    available_locations.extend(new_list)
    heapq.heapify(available_locations)

    return is_current_location_clear


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
    """Trace one MATLAB watershed half-edge back to its zero-pointer origin using flat views."""
    pointer_map_flat = pointer_map.ravel(order="F")
    size_map_flat = size_map.ravel(order="F")
    
    # CRITICAL DIAGNOSTIC: Verify size_map_flat is a view
    import logging
    if not np.shares_memory(size_map, size_map_flat):
        logging.error(f"TRACE BUG: size_map_flat is NOT a view of size_map in trace function!")
    
    # DIAGNOSTIC: Log if we're tracing from location 12532290
    if start_linear == 12532290 or start_linear == 12470094:
        logging.error(
            f"TRACE STARTING FROM {start_linear}: "
            f"pointer={int(pointer_map_flat[start_linear])}, "
            f"scale={int(size_map_flat[start_linear])}"
        )

    traced: list[int] = []
    visited: set[int] = set()
    tracing_linear = int(start_linear)
    img_size = pointer_map_flat.size

    # DIAGNOSTIC: Track first few steps for debugging
    trace_log: list[tuple[int, int, int]] = []  # (location, pointer, scale)

    while 0 <= tracing_linear < img_size:
        if tracing_linear in visited:
            import logging

            logging.error(
                f"Cycle detected in global watershed backtrack at {tracing_linear}. Breaking."
            )
            break
        visited.add(tracing_linear)
        traced.append(int(tracing_linear))

        pointer_value = int(pointer_map_flat[tracing_linear])
        if pointer_value == 0:
            break

        tracing_scale_label = int(size_map_flat[tracing_linear])

        # DIAGNOSTIC: Log first few steps
        if len(trace_log) < 5:
            trace_log.append((tracing_linear, pointer_value, tracing_scale_label))

        tracing_scale_index = int(
            np.clip(tracing_scale_label - 1, 0, len(lumen_radius_microns) - 1)
        )
        lut = _build_matlab_global_watershed_lut(
            tracing_scale_index,
            size_of_image=shape,
            lumen_radius_microns=lumen_radius_microns,
            microns_per_voxel=microns_per_voxel,
            step_size_per_origin_radius=step_size_per_origin_radius,
        )
        linear_offsets = np.asarray(lut["linear_offsets"], dtype=np.int64)

        # Pointer is 1-based, so valid range is [1, len(linear_offsets)]
        if pointer_value < 1 or pointer_value > len(linear_offsets):
            import logging

            logging.error(
                f"Pointer index {pointer_value} out of range for scale {tracing_scale_label} "
                f"(size {len(linear_offsets)}) at location {tracing_linear}. "
                f"Valid range: [1, {len(linear_offsets)}]. "
                f"Pointer dtype: {pointer_map_flat.dtype}, value type: {type(pointer_value)}. "
                f"Trace history: {trace_log}"
            )
            break

        # Use pointer_value - 1 since it's 1-based
        offset = int(linear_offsets[pointer_value - 1])
        tracing_linear = int(tracing_linear - offset)

        # Bounds check the new location
        if tracing_linear < 0 or tracing_linear >= img_size:
            import logging

            logging.error(
                f"Backtrack resulted in out-of-bounds location {tracing_linear} "
                f"(image size {img_size}) from pointer {pointer_value} with offset {offset}"
            )
            break

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
        size_map: np.ndarray = np.ones(shape, dtype=np.int16, order="F")
        original_scale_image = np.zeros(shape, dtype=np.int16, order="F")
    else:
        original_scale_image = np.asarray(
            cast("np.ndarray", scale_indices),
            dtype=np.int16,
            order="F",
        )
        size_map = np.asarray(original_scale_image, dtype=np.int16, order="F").copy()
        size_map += np.int16(1)
        # CRITICAL FIX: Clip size_map to valid range to prevent out-of-range scale labels
        # The input scale_indices might have values outside [0, len(lumen_radius_microns)-1]
        # After adding 1, size_map should be in range [1, len(lumen_radius_microns)]
        size_map = np.clip(size_map, 1, len(lumen_radius_microns))
        # CRITICAL FIX: np.clip() may return a non-F-contiguous array, causing ravel(order="F")
        # to create a COPY instead of a view. Ensure F-contiguity so writes persist.
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

    edge_number_tolerance = int(params.get("edge_number_tolerance", 2))
    energy_tolerance = float(params.get("energy_tolerance", 1.0))
    step_size_per_origin_radius = float(params.get("step_size_per_origin_radius", 1.0))
    distance_tolerance = float(params.get("distance_tolerance", 3.0))

    # Pre-flatten maps for performance
    energy_map_temp_flat = energy_map_temp.ravel(order="F")
    d_over_r_map_flat = d_over_r_map.ravel(order="F")
    pointer_map_flat = pointer_map.ravel(order="F")
    vertex_index_map_flat = vertex_index_map.ravel(order="F")
    size_map_flat = size_map.ravel(order="F")
    branch_order_map_flat = branch_order_map.ravel(order="F")
    
    # CRITICAL DIAGNOSTIC: Verify size_map_flat is a view of size_map
    import logging
    logging.error(f"size_map flags: C_CONTIGUOUS={size_map.flags['C_CONTIGUOUS']}, F_CONTIGUOUS={size_map.flags['F_CONTIGUOUS']}, OWNDATA={size_map.flags['OWNDATA']}")
    if not np.shares_memory(size_map, size_map_flat):
        logging.error("CRITICAL BUG: size_map_flat is NOT a view of size_map! Writes will not persist!")
        logging.error(f"size_map_flat flags: C_CONTIGUOUS={size_map_flat.flags['C_CONTIGUOUS']}, F_CONTIGUOUS={size_map_flat.flags['F_CONTIGUOUS']}, OWNDATA={size_map_flat.flags['OWNDATA']}")
    else:
        logging.info(f"size_map_flat is a view of size_map: {size_map_flat.base is size_map}")

    edge_halves: list[tuple[list[int], list[int]]] = []
    edge_pairs: list[tuple[int, int]] = []

    last_heartbeat_at = time.monotonic()

    # Track the raw vertex energies for resetting -inf values
    vertex_energies_raw_flat = energy_map_raw.ravel(order="F")

    # Track pointer filtering statistics
    filtered_pointer_stats: dict[str, int] = {}

    available_locations_heap: list[tuple[float, int, int]] = []
    insertion_counter = [0]
    for loc in available_locations:
        _matlab_global_watershed_insert_available_location(
            next_location=loc,
            next_energy=energy_map_temp_flat[loc],
            available_locations=available_locations_heap,
            energy_map_flat=energy_map_temp_flat,
            insertion_counter=insertion_counter,
        )

    iteration = 0
    while available_locations_heap:
        iteration += 1
        current_energy, _, current_linear = heapq.heappop(available_locations_heap)
        is_current_location_clear = False

        current_energy = energy_map_temp_flat[current_linear]

        # Reset vertex energy to original value if first time popping it
        if float(current_energy) == float("-inf"):
            current_energy = vertex_energies_raw_flat[current_linear]
            energy_map_temp_flat[current_linear] = current_energy

        current_vertex_index = int(vertex_index_map_flat[current_linear])
        current_scale_label = int(size_map_flat[current_linear])
        current_scale_index = int(
            np.clip(current_scale_label - 1, 0, len(lumen_radius_microns) - 1)
        )

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
        
        # DIAGNOSTIC: Log when scale clipping occurs
        if current_scale_label_for_writing != current_scale_label:
            import logging
            logging.warning(
                f"SCALE CLIPPING: original={current_scale_label}, clipped={current_scale_label_for_writing}, "
                f"location={current_linear}, lut_size={current_strel['lut_size']}"
            )
        
        # Extract strel arrays first
        current_strel_r_over_R = cast("np.ndarray", current_strel["r_over_R"])
        current_strel_coords = cast("np.ndarray", current_strel["coords"])
        current_strel_linear = cast("np.ndarray", current_strel["linear_indices"])
        current_strel_offsets = cast("np.ndarray", current_strel["offsets"])
        current_strel_pointer_indices = cast("np.ndarray", current_strel["pointer_indices"])
        
        # DIAGNOSTIC: Log for specific problematic iteration
        if current_linear == 12532290 or (current_linear in [12470094, 12532290]):
            import logging
            logging.error(
                f"PROCESSING PROBLEM LOCATION {current_linear}: "
                f"original_scale={current_scale_label}, clipped_scale={current_scale_label_for_writing}, "
                f"lut_size={current_strel['lut_size']}, "
                f"scale_label_clipped_in_strel={current_strel.get('scale_label_clipped', 'NOT FOUND')}, "
                f"pointer_indices_range=[{np.min(current_strel_pointer_indices)}, {np.max(current_strel_pointer_indices)}]"
            )
        
        # DIAGNOSTIC: Check if any neighbor locations include 12532290
        if 12532290 in current_strel_linear:
            import logging
            idx = np.where(current_strel_linear == 12532290)[0][0]
            logging.error(
                f"LOCATION 12532290 IS A NEIGHBOR of {current_linear}: "
                f"will write pointer={current_strel_pointer_indices[idx]}, "
                f"scale={current_scale_label_for_writing}, "
                f"lut_size={current_strel['lut_size']}, "
                f"vertex={current_vertex_index}"
            )

        current_strel_energies = energy_map_temp_flat[current_strel_linear].astype(
            np.float32, copy=False
        )
        current_d_over_r = float(d_over_r_map_flat[current_linear])
        current_forward_unit: np.ndarray | None = None
        current_pointer_value = int(pointer_map_flat[current_linear])
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
            neighbor_distances_microns=current_strel_r_over_R
            * max(float(lumen_radius_microns[current_scale_index]), 1e-6),
            neighbor_scale_indices=size_map_flat[current_strel_linear],
            propagated_scale_index=current_scale_label,
            current_distance_microns=current_d_over_r
            * max(float(lumen_radius_microns[current_scale_index]), 1e-6),
            origin_radius_microns=max(float(lumen_radius_microns[current_scale_index]), 1e-6),
            current_forward_unit=current_forward_unit,
            microns_per_voxel=microns_per_voxel,
            lumen_radius_microns=lumen_radius_microns,
            distance_tolerance=distance_tolerance,
        )

        # Sample ownership BEFORE claiming to determine growability/connectivity
        vertices_of_current_strel = vertex_index_map_flat[current_strel_linear]

        # Identify tolerated candidates from PENALIZED energies
        # MATLAB: is_energy_tolerated_in_strel = current_strel_energies < vertex_energies( current_vertex_index ) * energy_tolerance_coeff ;
        is_energy_tolerated_in_strel = adjusted < (
            float(vertex_energies[current_vertex_index - 1]) * (1.0 - energy_tolerance)
        )

        # Validate pointer indices before passing to reveal
        if np.any(current_strel_pointer_indices < 1) or np.any(
            current_strel_pointer_indices > current_strel["lut_size"]
        ):
            import logging

            bad_ptrs = current_strel_pointer_indices[
                (current_strel_pointer_indices < 1)
                | (current_strel_pointer_indices > current_strel["lut_size"])
            ]
            logging.error(
                f"CRITICAL BUG DETECTED: Invalid pointers BEFORE reveal at iteration {iteration}, "
                f"scale {current_scale_label}, location {current_linear}. "
                f"Bad pointer range: [{np.min(bad_ptrs)}, {np.max(bad_ptrs)}] vs LUT size {current_strel['lut_size']}. "
                f"Total pointers: {len(current_strel_pointer_indices)}"
            )

        _matlab_global_watershed_reveal_unclaimed_strel(
            current_vertex_index=current_vertex_index,
            current_scale_label=current_scale_label_for_writing,
            current_d_over_r=current_d_over_r,
            valid_linear=current_strel_linear,
            strel_pointer_indices=current_strel_pointer_indices,
            strel_distance_microns=current_strel["distance_microns"],
            strel_adjusted_energies=adjusted,
            vertex_index_map_flat=vertex_index_map_flat,
            energy_map_flat=energy_map_temp_flat,
            pointer_map_flat=pointer_map_flat,
            d_over_r_map_flat=d_over_r_map_flat,
            size_map_flat=size_map_flat,
            lut_size=current_strel["lut_size"],
            filtered_pointer_stats=filtered_pointer_stats,
        )
        
        # DIAGNOSTIC: Verify what was actually written
        if np.random.random() < 0.001:  # Log 0.1% of iterations
            import logging
            is_without_vertex = vertex_index_map_flat[current_strel_linear] == current_vertex_index
            if np.any(is_without_vertex):
                sample_idx = np.where(is_without_vertex)[0][0]
                sample_location = current_strel_linear[sample_idx]
                written_scale = int(size_map_flat[sample_location])
                written_pointer = int(pointer_map_flat[sample_location])
                logging.info(
                    f"VERIFY: location={sample_location}, written_pointer={written_pointer}, "
                    f"written_scale={written_scale}, expected_scale={current_scale_label_for_writing}, "
                    f"original_scale={current_scale_label}, lut_size={current_strel['lut_size']}"
                )

        for seed_idx in range(1, edge_number_tolerance + 1):
            strel_idx = int(np.argmin(adjusted))
            next_location = int(current_strel_linear[strel_idx])
            next_vertex_index = int(vertices_of_current_strel[strel_idx])

            if not bool(is_energy_tolerated_in_strel[strel_idx]):
                if not is_current_location_clear:
                    is_current_location_clear = True
            else:
                # Apply directional suppression for next seed
                from .common import _matlab_frontier_directional_suppression_factors

                adjusted *= _matlab_frontier_directional_suppression_factors(
                    current_strel_offsets,
                    selected_index=strel_idx,
                    microns_per_voxel=microns_per_voxel,
                )

                if next_vertex_index == 0:
                    branch_order = int(branch_order_map_flat[current_linear]) + seed_idx - 1
                    branch_order_map_flat[next_location] = np.uint8(branch_order)
                    if branch_order < edge_number_tolerance:
                        _matlab_global_watershed_insert_available_location(
                            next_location=next_location,
                            next_energy=float(energy_map_temp_flat[next_location]),
                            available_locations=available_locations_heap,
                            energy_map_flat=energy_map_temp_flat,
                            insertion_counter=insertion_counter,
                        )
                        if not is_current_location_clear:
                            is_current_location_clear = True
                else:
                    is_next_vertex_in_strel = vertices_of_current_strel == next_vertex_index
                    is_current_location_clear = _matlab_global_watershed_reset_join_locations(
                        available_locations=available_locations_heap,
                        next_vertex_locations=current_strel_linear[is_next_vertex_in_strel],
                        is_current_location_clear=is_current_location_clear,
                        current_location=current_linear,
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

        if heartbeat is not None:
            now = time.monotonic()
            if iteration == 1 or iteration % 512 == 0 or (now - last_heartbeat_at) >= 5.0:
                heartbeat(iteration, len(edge_pairs))
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

    # Log pointer filtering statistics
    if filtered_pointer_stats:
        import logging

        total_filtered = filtered_pointer_stats.get("total_filtered", 0)
        total_attempts = filtered_pointer_stats.get("total_attempts", 0)
        if total_filtered > 0:
            logging.warning(
                f"Pointer filtering summary: {total_filtered}/{total_attempts} pointers filtered "
                f"({100.0 * total_filtered / max(total_attempts, 1):.1f}%)"
            )
        diagnostics["filtered_pointer_count"] = total_filtered
        diagnostics["total_pointer_attempts"] = total_attempts

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
