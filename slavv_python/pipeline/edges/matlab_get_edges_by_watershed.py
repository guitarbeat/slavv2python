"""MATLAB port: ``get_edges_by_watershed.m`` — global watershed edge discovery.

Role: shared spatial maps, strel propagation, frontier queue, and candidate assembly.
MATLAB source: ``external/Vectorization-Public/source/get_edges_by_watershed.m``
Uses: ``matlab_watershed_heap.py`` (heap/claim structures), ``matlab_get_edges_v300_geometry.py``
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:
    from slavv_python.pipeline.edges.edge_types import (
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

from slavv_python.pipeline.edges.execution_tracing import (
    ExecutionTracer,
    NullExecutionTracer,
)
from slavv_python.pipeline.edges.matlab_calculate_linear_strel_range import (
    _build_matlab_global_watershed_lut,
)
from slavv_python.pipeline.edges.matlab_get_edges_v300_geometry import (
    _matlab_frontier_adjusted_neighbor_energies,
    _matlab_frontier_directional_suppression_factors,
)
from slavv_python.pipeline.edges.matlab_indexing import (
    _argmin_with_linear_index_tiebreak,
    _matlab_linear_index_to_coord,
)
from slavv_python.pipeline.edges.matlab_watershed_heap import (
    FrontierQueue,
    VoxelClaimMap,
    _matlab_global_watershed_border_locations,
)
from slavv_python.pipeline.edges.payloads import (
    _edge_metric_from_energy_trace,
    _empty_edge_diagnostics,
)

__all__ = ["_matlab_global_watershed_border_locations"]


def _coords_from_linear_trace(
    linear_trace: list[int],
    shape: tuple[int, int, int],
) -> np.ndarray:
    """Return a (N, 3) coordinate array from a list of linear indices."""
    if not linear_trace:
        return np.zeros((0, 3), dtype=np.float64)
    # _matlab_linear_index_to_coord returns [y, x, z]
    coords_yxz = np.asarray(
        [_matlab_linear_index_to_coord(int(idx), shape) for idx in linear_trace],
        dtype=np.float64,
    )
    # Reorient [y, x, z] -> [z, y, x]
    coords_zyx = np.zeros_like(coords_yxz)
    coords_zyx[:, 0] = coords_yxz[:, 2]  # Z
    coords_zyx[:, 1] = coords_yxz[:, 0]  # Y
    coords_zyx[:, 2] = coords_yxz[:, 1]  # X
    return cast("np.ndarray", coords_zyx)


def _sample_volume_from_matlab_linear_trace(
    linear_trace: list[int],
    volume: np.ndarray,
) -> np.ndarray:
    """Sample one volume exactly at normalized MATLAB-order linear indices."""
    if not linear_trace:
        return cast("np.ndarray", np.zeros((0,), dtype=np.asarray(volume).dtype))
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
        dtype=np.float64,
    )
    if scale_image is None:
        scale_trace: np.ndarray = np.zeros((len(full_linear_trace),), dtype=np.int16)
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
    scaled_pointer_map = np.zeros(pointer_map.shape, dtype=np.float64)
    pointer_mask: np.ndarray = pointer_map > 0
    if not np.any(pointer_mask):
        return cast("np.ndarray", scaled_pointer_map)

    scale_labels = size_map[pointer_mask].astype(np.int64, copy=False)
    scale_indices = np.clip(scale_labels - 1, 0, len(lumen_radius_microns) - 1)
    unique_lengths: np.ndarray = np.zeros(len(lumen_radius_microns), dtype=np.float64)
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
        1000.0 / np.maximum(strel_lengths, 1.0) * pointer_map[pointer_mask].astype(np.float64)
    )
    return cast("np.ndarray", scaled_pointer_map)


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
        size_map: np.ndarray = np.ones(shape, dtype=np.int16, order="F")
        original_scale_image = np.zeros(shape, dtype=np.int16, order="F")
    else:
        original_scale_image = np.asarray(scale_indices, dtype=np.int16, order="F")
        size_map = np.asarray(original_scale_image, dtype=np.int16, order="F").copy()
        size_map += np.int16(1)
        # CRITICAL FIX: Clip size_map to valid range to prevent out-of-range scale labels
        size_map = np.clip(size_map, 1, len(lumen_radius_microns))
        # Ensure F-contiguity for persisting writes
        size_map = np.asfortranarray(size_map)

    # Note: vertex positions are in ZYX order, but size_map is currently oriented based on input.
    vertex_coords = np.rint(np.asarray(vertex_positions, dtype=np.float64)).astype(
        np.int32, copy=False
    )

    # Standard clipping
    vertex_coords[:, 0] = np.clip(vertex_coords[:, 0], 0, shape[0] - 1)
    vertex_coords[:, 1] = np.clip(vertex_coords[:, 1], 0, shape[1] - 1)
    vertex_coords[:, 2] = np.clip(vertex_coords[:, 2], 0, shape[2] - 1)

    size_map[
        vertex_coords[:, 0],
        vertex_coords[:, 1],
        vertex_coords[:, 2],
    ] = np.rint(np.asarray(vertex_scales, dtype=np.float64)).astype(size_map.dtype) + np.int16(1)

    return size_map, original_scale_image


def _matlab_global_watershed_current_strel(
    current_linear: int,
    *,
    current_scale_label: int,
    shape: tuple[int, int, int],
    lumen_radius_microns: Float32Array,
    microns_per_voxel: Float32Array,
    step_size_per_origin_radius: float,
) -> dict[str, Any]:
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
        "r_over_R": np.asarray(lut["r_over_R"], dtype=np.float64)[valid_mask],
        "distance_microns": np.asarray(lut["distance_lut"], dtype=np.float64)[valid_mask],
        "unit_vectors": np.asarray(lut["unit_vectors"], dtype=np.float64)[valid_mask],
        "lut_size": len(offsets),  # Store for debugging
        "scale_label_clipped": current_scale_index
        + 1,  # Clipped scale for consistent pointer/size_map usage
    }


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
    return cast("Float32Array", unit_vectors.astype(np.float64, copy=False))


def _matlab_global_watershed_tolerance_mask(
    adjusted_energies: Float64Array,
    *,
    current_vertex_energy: float,
    energy_tolerance: float,
) -> BoolArray:
    """Mirror MATLAB's per-seed energy tolerance test on the current penalized strel energies."""
    threshold = float(current_vertex_energy) * (1.0 - float(energy_tolerance))
    return cast("BoolArray", np.asarray(adjusted_energies, dtype=np.float64) < threshold)


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
    n_voxels = int(pointer_map.size)
    if not 0 <= int(start_linear) < n_voxels:
        raise IndexError(
            f"watershed trace start index {int(start_linear)} out of bounds for a "
            f"{n_voxels}-voxel map (shape {shape}); the energy/size_map are likely "
            f"mis-oriented — they must be [Z, Y, X] matching the oracle "
            f"(check energy_axis_permutation)."
        )
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
        if not 0 <= next_linear < n_voxels:
            raise IndexError(
                f"watershed pointer chain stepped out of bounds to {next_linear} from "
                f"{current_linear} (map size {n_voxels}, shape {shape}); this indicates a "
                f"size_map/pointer_map axis-order inconsistency — verify the inputs are "
                f"oriented [Z, Y, X] (energy_axis_permutation)."
            )
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
    energy_map_matlab: np.ndarray,
    original_scale_image_matlab: Int16Array | None,
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
    connection_sources: list[int] = []
    diagnostics = _empty_edge_diagnostics()

    # Inputs are already in MATLAB [Y, X, Z] order
    flat_energy_map = energy_map_matlab.ravel(order="F")
    flat_scale_image = (
        original_scale_image_matlab.ravel(order="F")
        if original_scale_image_matlab is not None
        else None
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
        connection_sources.append(1)  # Watershed

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

    # Transpose result maps back to physical [Z, Y, X] order for persistence
    def _to_zyx(arr):
        return np.transpose(arr, (2, 0, 1)).copy(order="C")

    return {
        "traces": traces,
        "connections": np.asarray(connections, dtype=np.int32).reshape(-1, 2),
        "metrics": np.asarray(metrics, dtype=np.float64),
        "energy_traces": energy_traces,
        "scale_traces": scale_traces,
        "origin_indices": np.asarray(origin_indices, dtype=np.int32),
        "connection_sources": connection_sources,
        "diagnostics": diagnostics,
        "matlab_global_watershed_exact": True,
        "candidate_source": "global_watershed",
        "energy_map": _to_zyx(energy_map_matlab),
        "vertex_index_map": _to_zyx(vertex_index_map),
        "pointer_map": _to_zyx(scaled_pointer_map),
        "raw_pointer_map": _to_zyx(raw_pointer_map),
        "d_over_r_map": _to_zyx(d_over_r_map),
        "branch_order_map": _to_zyx(branch_order_map),
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
            "metrics": np.zeros((0,), dtype=np.float64),
            "energy_traces": [],
            "scale_traces": [],
            "origin_indices": np.zeros((0,), dtype=np.int32),
            "connection_sources": [],
            "diagnostics": _empty_edge_diagnostics(),
            "matlab_global_watershed_exact": True,
        }

    # Reorient inputs to MATLAB [Y, X, Z] order for watershed processing.
    energy_matlab = np.transpose(np.asarray(energy, dtype=np.float64), (1, 2, 0)).copy(order="F")

    scale_indices_matlab: np.ndarray | None = None
    if scale_indices is not None:
        scale_indices_matlab = np.transpose(
            np.asarray(scale_indices, dtype=np.int16), (1, 2, 0)
        ).copy(order="F")

    from slavv_python.utils.matlab_order import zyx_to_matlab_linear_indices

    vertex_locations = zyx_to_matlab_linear_indices(vertex_positions, energy.shape)

    shape: tuple[int, int, int] = (
        int(energy_matlab.shape[0]),
        int(energy_matlab.shape[1]),
        int(energy_matlab.shape[2]),
    )

    claim_map = VoxelClaimMap(shape, vertex_positions, energy_matlab)
    claim_map.initial_locations = [int(loc) for loc in vertex_locations[::-1]]

    queue = FrontierQueue(claim_map.initial_locations, claim_map.energy_temp_flat)
    number_of_vertices = claim_map.number_of_vertices

    vertex_coords_yxz = np.zeros_like(vertex_positions, dtype=np.float64)
    vertex_coords_yxz[:, 0] = vertex_positions[:, 1]  # Y
    vertex_coords_yxz[:, 1] = vertex_positions[:, 2]  # X
    vertex_coords_yxz[:, 2] = vertex_positions[:, 0]  # Z

    mpv_matlab = np.asarray(microns_per_voxel, dtype=np.float64)[[1, 2, 0]]

    size_map, original_scale_image = _matlab_global_watershed_prepare_size_map(
        shape, scale_indices_matlab, vertex_coords_yxz, vertex_scales, lumen_radius_microns
    )

    size_map_flat = size_map.ravel(order="F")

    # MATLAB get_edges_V300.m (line 100) hard-codes edge_number_tolerance = 2 for the
    # watershed seed count, overriding the deprecated number_of_edges_per_vertex input.
    # (number_of_edges_per_vertex = 4 is used later only for degree-excess cleanup, not
    # for seeding.) Honoring the param here (=4) doubled the seeds per vertex and shifted
    # the entire watershed, diverging from MATLAB.
    edge_number_tolerance = 2
    energy_tolerance = float(params.get("energy_tolerance", 1.0))
    radius_tolerance = float(params.get("radius_tolerance", 0.5))
    step_size_per_origin_radius = float(params.get("step_size_per_origin_radius", 1.0))
    distance_tolerance = float(
        params.get("distance_tolerance", params.get("distance_tolerance_per_origin_radius", 3.0))
    )

    edge_halves: list[tuple[list[int], list[int]]] = []
    edge_pairs: list[tuple[int, int]] = []

    last_heartbeat_at = time.monotonic()
    iteration = 0

    while queue:
        iteration += 1
        current_linear = queue.pop_best()

        current_energy = claim_map.restore_vertex_energy(current_linear)
        active_tracer.on_iteration_start(iteration, current_linear, current_energy)

        if current_energy >= 0.0:
            break

        current_vertex_index = int(claim_map.vertex_index_flat[current_linear])
        current_scale_label = int(size_map_flat[current_linear])
        # MATLAB get_edges_by_watershed.m:243 references the ORIGIN VERTEX's scale
        # (size_map(vertex_locations(current_strel_vertex_index))) for the size penalty —
        # constant across the whole watershed — NOT the current voxel's propagated/clipped
        # scale. Using the current voxel's scale lets clipped propagation drift on long
        # traces, inflating the size Gaussian penalty and terminating long edges early.
        origin_vertex_location = int(claim_map.vertex_locations[current_vertex_index - 1])
        origin_scale_label = int(size_map_flat[origin_vertex_location])
        current_scale_index = int(
            np.clip(current_scale_label - 1, 0, len(lumen_radius_microns) - 1)
        )
        current_pointer_value = int(claim_map.pointer_flat[current_linear])
        current_d_over_r = float(claim_map.d_over_r_flat[current_linear])

        current_strel = _matlab_global_watershed_current_strel(
            current_linear,
            current_scale_label=current_scale_label,
            shape=shape,
            lumen_radius_microns=lumen_radius_microns,
            microns_per_voxel=mpv_matlab,
            step_size_per_origin_radius=step_size_per_origin_radius,
        )
        current_scale_label_for_writing = current_strel.get(
            "scale_label_clipped", current_scale_label
        )

        current_strel_r_over_R = cast("Float32Array", current_strel["r_over_R"])
        current_strel_coords = cast("Int32Array", current_strel["coords"])
        current_strel_linear = cast("Int64Array", current_strel["linear_indices"])
        current_strel_offsets = cast("Int32Array", current_strel["offsets"])
        current_strel_pointer_indices = cast("Int64Array", current_strel["pointer_indices"])

        current_strel_energies = claim_map.energy_temp_flat[current_strel_linear]

        current_forward_unit: np.ndarray | None = None
        if current_pointer_value > 0:
            lut = _build_matlab_global_watershed_lut(
                current_scale_index,
                size_of_image=shape,
                lumen_radius_microns=lumen_radius_microns,
                microns_per_voxel=microns_per_voxel,
                step_size_per_origin_radius=step_size_per_origin_radius,
            )
            full_unit_vectors = np.asarray(lut["unit_vectors"], dtype=np.float64)
            current_forward_unit = full_unit_vectors[current_pointer_value - 1]

        adjusted = _matlab_frontier_adjusted_neighbor_energies(
            current_strel_energies,
            neighbor_offsets=current_strel_offsets,
            neighbor_r_over_R=current_strel_r_over_R,
            neighbor_scale_indices=size_map_flat[current_strel_linear],
            propagated_scale_index=origin_scale_label,
            current_d_over_r=current_d_over_r,
            origin_radius_microns=max(float(lumen_radius_microns[current_scale_index]), 1e-6),
            current_forward_unit=current_forward_unit,
            microns_per_voxel=mpv_matlab,
            lumen_radius_microns=lumen_radius_microns,
            radius_tolerance=radius_tolerance,
            distance_tolerance=distance_tolerance,
        )

        vertices_of_current_strel, _is_without_vertex = claim_map.claim_unowned_strel(
            current_vertex_index=current_vertex_index,
            current_scale_label=current_scale_label_for_writing,
            current_d_over_r=current_d_over_r,
            valid_linear=current_strel_linear,
            strel_pointer_indices=current_strel_pointer_indices,
            strel_r_over_R=current_strel_r_over_R,
            adjusted_energies=adjusted,
            size_map_flat=size_map_flat,
            lut_size=current_strel["lut_size"],
        )

        for seed_idx in _matlab_global_watershed_seed_index_range(
            current_pointer_value=current_pointer_value,
            edge_number_tolerance=edge_number_tolerance,
        ):
            is_energy_tolerated_in_strel = _matlab_global_watershed_tolerance_mask(
                adjusted,
                current_vertex_energy=float(claim_map.vertex_energies[current_vertex_index - 1]),
                energy_tolerance=energy_tolerance,
            )
            strel_idx = _argmin_with_linear_index_tiebreak(adjusted, current_strel_linear)
            next_location = int(current_strel_linear[strel_idx])
            next_vertex_index = int(vertices_of_current_strel[strel_idx])
            active_tracer.on_seed_selected(seed_idx, next_location, float(adjusted[strel_idx]))

            if bool(is_energy_tolerated_in_strel[strel_idx]):
                if next_vertex_index == 0:
                    branch_order = int(claim_map.branch_order_flat[current_linear]) + seed_idx - 1
                    claim_map.branch_order_flat[next_location] = np.uint8(branch_order)
                    if branch_order < edge_number_tolerance:
                        queue.push(
                            next_location,
                            float(claim_map.energy_temp_flat[next_location]),
                            seed_idx,
                        )
                else:
                    is_next_vertex_in_strel = vertices_of_current_strel == next_vertex_index
                    queue.remove_first_occurrence(current_strel_linear[is_next_vertex_in_strel])

                    if not bool(
                        claim_map.adjacency_matrix[next_vertex_index - 1, current_vertex_index - 1]
                    ):
                        claim_map.adjacency_matrix[
                            current_vertex_index - 1,
                            next_vertex_index - 1,
                        ] = True
                        claim_map.adjacency_matrix[
                            next_vertex_index - 1,
                            current_vertex_index - 1,
                        ] = True

                        half_1 = _matlab_global_watershed_trace_half(
                            current_linear,
                            pointer_map=claim_map.pointer_map,
                            size_map=size_map,
                            shape=shape,
                            lumen_radius_microns=lumen_radius_microns,
                            # MUST match the strel LUT used when the pointers were
                            # written (claim path uses mpv_matlab = mpv[[1,2,0]]).
                            # Using raw microns rebuilds different linear offsets, so
                            # the pointer chain diverges (and overruns the bound at
                            # full-volume scale with large strels).
                            microns_per_voxel=mpv_matlab,
                            step_size_per_origin_radius=step_size_per_origin_radius,
                        )

                        other_half_start = next_location
                        if next_vertex_index != number_of_vertices + 1:
                            is_vertex_b_origin = (
                                claim_map.pointer_map[
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
                            pointer_map=claim_map.pointer_map,
                            size_map=size_map,
                            shape=shape,
                            lumen_radius_microns=lumen_radius_microns,
                            microns_per_voxel=mpv_matlab,  # see half_1: match claim-path strel LUT
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

            adjusted[strel_idx] = np.inf

            adjusted *= _matlab_frontier_directional_suppression_factors(
                current_strel_offsets,
                selected_index=strel_idx,
                microns_per_voxel=microns_per_voxel,
            )
            adjusted[~np.isfinite(adjusted)] = np.inf

        if heartbeat is not None:
            now = time.monotonic()
            if iteration == 1 or iteration % 512 == 0 or (now - last_heartbeat_at) >= 5.0:
                heartbeat(iteration, len(edge_pairs))
                last_heartbeat_at = now

    return _matlab_global_watershed_assemble_results(
        edge_pairs=edge_pairs,
        edge_halves=edge_halves,
        shape=shape,
        energy_map_matlab=energy_matlab,
        original_scale_image_matlab=original_scale_image,
        vertex_positions=vertex_positions,
        vertex_index_map=claim_map.vertex_index_map,
        pointer_map=claim_map.pointer_map,
        size_map=size_map,
        d_over_r_map=claim_map.d_over_r_map,
        branch_order_map=claim_map.branch_order_map,
        lumen_radius_microns=lumen_radius_microns,
        microns_per_voxel=microns_per_voxel,
        step_size_per_origin_radius=step_size_per_origin_radius,
    )


def _initialize_matlab_global_watershed_state(
    energy: np.ndarray,
    vertex_positions: np.ndarray,
) -> dict[str, np.ndarray]:
    """Compatibility shim for unit tests."""
    claim_map = VoxelClaimMap(energy.shape, vertex_positions, energy)
    return {
        "vertex_locations": claim_map.vertex_locations,
        "vertex_energies": claim_map.vertex_energies,
        "energy_map_temp": claim_map.energy_map_temp,
        "branch_order_map": claim_map.branch_order_map,
        "d_over_r_map": claim_map.d_over_r_map,
        "pointer_map": claim_map.pointer_map,
        "vertex_index_map": claim_map.vertex_index_map,
        "initial_locations": claim_map.initial_locations,
    }


def _matlab_global_watershed_reveal_unclaimed_strel(
    *,
    current_vertex_index: int,
    current_scale_label: int,
    current_d_over_r: float,
    valid_linear: np.ndarray,
    strel_pointer_indices: np.ndarray,
    strel_r_over_R: np.ndarray,
    adjusted_energies: np.ndarray,
    vertex_index_map_flat: np.ndarray,
    pointer_map_flat: np.ndarray,
    energy_map_flat: np.ndarray,
    d_over_r_map_flat: np.ndarray,
    size_map_flat: np.ndarray,
    lut_size: int,
) -> dict[str, np.ndarray]:
    """Compatibility shim for unit tests."""
    if len(strel_pointer_indices) != len(valid_linear):
        raise AssertionError("Strel arrays must stay aligned")

    vertices_of_current_strel = np.asarray(vertex_index_map_flat[valid_linear], dtype=np.uint32)
    is_without_vertex = vertices_of_current_strel == 0

    if np.any(is_without_vertex):
        claim_linear = valid_linear[is_without_vertex]
        claim_pointers = np.asarray(strel_pointer_indices[is_without_vertex], dtype=np.uint64)
        if np.any(claim_pointers < 1) or np.any(claim_pointers > lut_size):
            raise AssertionError("invalid claim pointers")

        vertex_index_map_flat[claim_linear] = np.uint32(current_vertex_index)
        pointer_map_flat[claim_linear] = claim_pointers
        energy_map_flat[claim_linear] = adjusted_energies[is_without_vertex]
        d_over_r_map_flat[claim_linear] = (
            np.asarray(strel_r_over_R[is_without_vertex], dtype=np.float64) + current_d_over_r
        )
        size_map_flat[claim_linear] = np.int16(current_scale_label)

    return {
        "vertices_of_current_strel": vertices_of_current_strel,
        "is_without_vertex_in_strel": is_without_vertex,
    }


def _matlab_global_watershed_insert_available_location(
    available_locations: list[int],
    next_location: int,
    next_energy: float,
    energy_lookup: dict | np.ndarray,
    seed_idx: int,
    is_current_location_clear: bool,
) -> tuple[list[int], bool]:
    """Compatibility shim for unit tests."""
    is_clear = is_current_location_clear
    updated = list(available_locations)
    if not is_current_location_clear and seed_idx > 1:
        if updated:
            updated.pop()
        is_clear = True

    target_energy = float(next_energy)

    insert_at = len(updated)
    for idx, loc in enumerate(updated):
        mid_energy = float(energy_lookup[loc])
        if seed_idx == 1:
            is_mid_worse = mid_energy > target_energy
        else:
            is_mid_worse = mid_energy >= target_energy
        if not is_mid_worse:
            insert_at = idx
            break

    updated.insert(insert_at, int(next_location))
    return updated, is_clear


def _matlab_global_watershed_reset_join_locations(
    available_locations: list[int],
    *,
    next_vertex_locations: np.ndarray,
    is_current_location_clear: bool,
) -> tuple[list[int], bool]:
    """Compatibility shim for unit tests."""
    is_clear = is_current_location_clear
    updated = list(available_locations)
    if not is_clear:
        if updated:
            updated.pop()
        is_clear = True

    locations_to_reset = set(np.asarray(next_vertex_locations, dtype=np.int64).tolist())
    updated = [loc for loc in updated if loc not in locations_to_reset]
    return updated, is_clear
