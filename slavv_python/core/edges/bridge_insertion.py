"""MATLAB-style bridge-vertex insertion helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

if TYPE_CHECKING:

    from slavv_python.core.edges.common import (
        BoolArray,
        Float32Array,
        Int16Array,
        Int32Array,
        Int64Array,
    )
else:
    Int16Array = np.ndarray
    Int32Array = np.ndarray
    Int64Array = np.ndarray
    Float32Array = np.ndarray
    BoolArray = np.ndarray

from slavv_python.core.edges.common import _matlab_frontier_offsets
from slavv_python.core.edges.selection_payloads import (
    build_selected_edges_result,
    normalize_candidate_connection_sources,
    prepare_candidate_indices_for_cleanup,
)
from slavv_python.core.graph import _matlab_edge_metrics
from slavv_python.core.vertices.painting import paint_vertex_center_image, paint_vertex_image


def _matlab_linear_indices_from_points(
    points: Float32Array,
    image_shape: tuple[int, int, int],
) -> Int64Array:
    coords = np.rint(np.asarray(points, dtype=np.float32)[:, :3]).astype(np.int64, copy=False)
    coords[:, 0] = np.clip(coords[:, 0], 0, image_shape[0] - 1)
    coords[:, 1] = np.clip(coords[:, 1], 0, image_shape[1] - 1)
    coords[:, 2] = np.clip(coords[:, 2], 0, image_shape[2] - 1)
    linear = (
        coords[:, 0]
        + coords[:, 1] * image_shape[0]
        + coords[:, 2] * image_shape[0] * image_shape[1]
    ).astype(np.int64, copy=False)
    return cast("Int64Array", linear)


def _matlab_position_from_linear_index(
    linear_index: int,
    image_shape: tuple[int, int, int],
) -> Int32Array:
    y_dim, x_dim, _z_dim = image_shape
    xy_area = y_dim * x_dim
    z_coord = linear_index // xy_area
    remainder = linear_index - z_coord * xy_area
    x_coord = remainder // y_dim
    y_coord = remainder - x_coord * y_dim
    return cast("Int32Array", np.asarray([y_coord, x_coord, z_coord], dtype=np.int32))


def _matlab_repeated_endpoint_interior_indices(
    edge_index_cells: list[Int64Array],
    vertex_linear_indices: Int64Array,
) -> list[int]:
    if not edge_index_cells:
        return []

    edge_indices = np.concatenate(edge_index_cells, axis=0)
    if edge_indices.size == 0:
        return []

    last_occurrence: dict[int, int] = {}
    for flat_index, linear_index in enumerate(edge_indices.tolist()):
        last_occurrence[int(linear_index)] = int(flat_index)

    repeated_in_order: list[int] = []
    seen_repeat_values: set[int] = set()
    for flat_index, linear_index in enumerate(edge_indices.tolist()):
        linear_value = int(linear_index)
        if flat_index == last_occurrence[linear_value]:
            continue
        if linear_value not in seen_repeat_values:
            repeated_in_order.append(linear_value)
            seen_repeat_values.add(linear_value)

    exterior_indices = {
        int(edge_indices_at_edge[0])
        for edge_indices_at_edge in edge_index_cells
        if edge_indices_at_edge.size > 0
    }
    exterior_indices.update(
        {
            int(edge_indices_at_edge[-1])
            for edge_indices_at_edge in edge_index_cells
            if edge_indices_at_edge.size > 0
        }
    )
    interior_indices = {
        int(linear_index)
        for edge_indices_at_edge in edge_index_cells
        for linear_index in edge_indices_at_edge[1:-1].tolist()
    }
    vertex_index_set = {int(linear_index) for linear_index in vertex_linear_indices.tolist()}

    new_vertex_indices: list[int] = []
    seen_new_vertices: set[int] = set()
    for linear_value in repeated_in_order:
        if linear_value in vertex_index_set:
            continue
        if linear_value not in exterior_indices or linear_value not in interior_indices:
            continue
        if linear_value not in seen_new_vertices:
            new_vertex_indices.append(linear_value)
            seen_new_vertices.add(linear_value)
    return new_vertex_indices


def _can_place_bridge_vertex(
    coord: Int32Array,
    scale_index: int,
    existing_vertex_volume_image: Float32Array,
    lumen_radius_pixels_axes: Float32Array,
    image_shape: tuple[int, int, int],
) -> bool:
    test_positions = np.asarray([coord], dtype=np.float32)
    test_scales = np.asarray([scale_index], dtype=np.int16)
    candidate_image = paint_vertex_image(
        test_positions,
        test_scales,
        np.asarray(lumen_radius_pixels_axes, dtype=np.float32),
        image_shape,
    )
    candidate_mask = candidate_image > 0
    if not np.any(candidate_mask):
        return False
    return not bool(np.any(existing_vertex_volume_image[candidate_mask]))


def _matlab_bridge_initialize_size_maps(
    *,
    traces: list[Float32Array],
    scale_traces: list[Float32Array],
    shape: tuple[int, int, int],
    scale_indices: Int16Array | None,
    energy: Float32Array,
) -> tuple[Float32Array, Int16Array, BoolArray]:
    """Initialize local energy and size maps over active traces."""
    edge_energy_map: np.ndarray = np.zeros(shape, dtype=np.float32)
    edge_size_map: np.ndarray = np.zeros(shape, dtype=np.int16)
    edge_mask: np.ndarray = np.zeros(shape, dtype=bool)

    for trace, scale_trace in zip(traces, scale_traces):
        coords = np.rint(np.asarray(trace, dtype=np.float32)[:, :3]).astype(np.int32, copy=False)
        coords[:, 0] = np.clip(coords[:, 0], 0, shape[0] - 1)
        coords[:, 1] = np.clip(coords[:, 1], 0, shape[1] - 1)
        coords[:, 2] = np.clip(coords[:, 2], 0, shape[2] - 1)
        edge_mask[coords[:, 0], coords[:, 1], coords[:, 2]] = True
        edge_energy_map[coords[:, 0], coords[:, 1], coords[:, 2]] = energy[
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
        ].astype(np.float32, copy=False)
        if scale_indices is not None and scale_indices.size:
            # Corrected: scale_indices is 0-based. Convert to 1-based labels for internal discovery state.
            edge_size_map[coords[:, 0], coords[:, 1], coords[:, 2]] = scale_indices[
                coords[:, 0],
                coords[:, 1],
                coords[:, 2],
            ].astype(np.int16, copy=False) + np.int16(1)
        else:
            # scale_trace from discovery is already 1-based.
            rounded_scales = np.rint(np.asarray(scale_trace, dtype=np.float32)).astype(
                np.int16,
                copy=False,
            )
            edge_size_map[coords[:, 0], coords[:, 1], coords[:, 2]] = rounded_scales

    return (
        cast("Float32Array", edge_energy_map),
        cast("Int16Array", edge_size_map),
        cast("BoolArray", edge_mask),
    )


def _matlab_bridge_separate_active_edges(
    *,
    traces: list[Float32Array],
    scale_traces: list[Float32Array],
    energy_traces: list[Float32Array],
    connection_sources: list[str],
    chosen_candidate_indices: Int32Array,
    connections: Int32Array,
    edge_index_cells: list[Int64Array],
    overlap_indices: list[int],
) -> tuple[
    list[Float32Array],
    list[Float32Array],
    list[Float32Array],
    list[str],
    Int32Array,
    Int32Array,
    list[Float32Array],
    list[Float32Array],
    list[Float32Array],
    list[str],
    Int32Array,
    Int32Array,
]:
    """Separate edges into active (containing structural overlaps) and inactive sets."""
    inactive_traces: list[Float32Array] = []
    inactive_scale_traces: list[Float32Array] = []
    inactive_energy_traces: list[Float32Array] = []
    inactive_connection_sources: list[str] = []
    inactive_candidate_indices: Int32Array = np.empty((0,), dtype=np.int32)
    inactive_connections: Int32Array = np.empty((0, 2), dtype=np.int32)

    overlap_index_set = {int(index) for index in overlap_indices}
    active_edge_indices = [
        edge_index
        for edge_index, edge_indices_at_edge in enumerate(edge_index_cells)
        if any(
            int(linear_index) in overlap_index_set for linear_index in edge_indices_at_edge.tolist()
        )
    ]
    inactive_edge_indices = [
        edge_index
        for edge_index in range(len(traces))
        if edge_index not in set(active_edge_indices)
    ]
    if inactive_edge_indices:
        inactive_traces = [traces[edge_index] for edge_index in inactive_edge_indices]
        inactive_scale_traces = [scale_traces[edge_index] for edge_index in inactive_edge_indices]
        inactive_energy_traces = [energy_traces[edge_index] for edge_index in inactive_edge_indices]
        inactive_connection_sources = [
            connection_sources[edge_index] for edge_index in inactive_edge_indices
        ]
        inactive_candidate_indices = chosen_candidate_indices[
            np.asarray(inactive_edge_indices, dtype=np.int32)
        ]
        inactive_connections = np.asarray(
            connections[np.asarray(inactive_edge_indices, dtype=np.int32)],
            dtype=np.int32,
        ).reshape(-1, 2)

    active_traces = [traces[edge_index] for edge_index in active_edge_indices]
    active_scale_traces = [scale_traces[edge_index] for edge_index in active_edge_indices]
    active_energy_traces = [energy_traces[edge_index] for edge_index in active_edge_indices]
    active_connection_sources = [
        connection_sources[edge_index] for edge_index in active_edge_indices
    ]
    active_candidate_indices = chosen_candidate_indices[
        np.asarray(active_edge_indices, dtype=np.int32)
    ]
    active_connections = np.asarray(
        connections[np.asarray(active_edge_indices, dtype=np.int32)],
        dtype=np.int32,
    ).reshape(-1, 2)

    return (
        active_traces,
        active_scale_traces,
        active_energy_traces,
        active_connection_sources,
        active_candidate_indices,
        active_connections,
        inactive_traces,
        inactive_scale_traces,
        inactive_energy_traces,
        inactive_connection_sources,
        inactive_candidate_indices,
        inactive_connections,
    )


def _matlab_bridge_build_bridge_payload(
    *,
    bridge_connections: list[list[int]],
    bridge_edges2vertices: list[list[int]],
    bridge_traces: list[Float32Array],
    bridge_scale_traces: list[Float32Array],
    bridge_energy_traces: list[Float32Array],
    bridge_mean_energies: Float32Array,
) -> dict[str, Any]:
    """Build the internal bridge-edge artifact payload."""
    return {
        "connections": (
            np.asarray(bridge_connections, dtype=np.int32).reshape(-1, 2)
            if bridge_connections
            else np.empty((0, 2), dtype=np.int32)
        ),
        "edges2vertices": (
            np.asarray(bridge_edges2vertices, dtype=np.int32).reshape(-1, 2)
            if bridge_edges2vertices
            else np.empty((0, 2), dtype=np.int32)
        ),
        "traces": bridge_traces,
        "edge_space_subscripts": bridge_traces,
        "scale_traces": bridge_scale_traces,
        "edge_scale_subscripts": bridge_scale_traces,
        "energy_traces": bridge_energy_traces,
        "edge_energies": bridge_energy_traces,
        "energies": bridge_mean_energies,
        "mean_edge_energies": bridge_mean_energies.copy(),
    }


def _matlab_bridge_search_target(
    overlap_linear_index: int,
    traces: list[Float32Array],
    scale_traces: list[Float32Array],
    child_edges: list[tuple[int, int]],
    *,
    scale_indices: Int16Array | None,
    energy: Float32Array,
    vertex_center_image: Int32Array,
    vertex_volume_image: Float32Array,
    lumen_radius_pixels_axes: Float32Array,
    lumen_radius_microns: Float32Array,
    microns_per_voxel: Float32Array,
    image_shape: tuple[int, int, int],
    strel_apothem: int,
    max_edge_length_per_origin_radius: float,
) -> dict[str, Any] | None:
    """Port MATLAB's local ``add_vertex_to_edge`` search over edge voxels."""
    if not traces:
        return None

    shape = image_shape
    edge_energy_map, edge_size_map, edge_mask = _matlab_bridge_initialize_size_maps(
        traces=traces,
        scale_traces=scale_traces,
        shape=shape,
        scale_indices=scale_indices,
        energy=energy,
    )

    for child_edge_index, _overlap_position in child_edges:
        child_coords = np.rint(
            np.asarray(traces[child_edge_index], dtype=np.float32)[:, :3]
        ).astype(
            np.int32,
            copy=False,
        )
        child_coords[:, 0] = np.clip(child_coords[:, 0], 0, shape[0] - 1)
        child_coords[:, 1] = np.clip(child_coords[:, 1], 0, shape[1] - 1)
        child_coords[:, 2] = np.clip(child_coords[:, 2], 0, shape[2] - 1)
        child_interior = child_coords[:-1]
        if child_interior.size == 0:
            continue
        edge_mask[child_interior[:, 0], child_interior[:, 1], child_interior[:, 2]] = False
        edge_energy_map[child_interior[:, 0], child_interior[:, 1], child_interior[:, 2]] = 0.0
        edge_size_map[child_interior[:, 0], child_interior[:, 1], child_interior[:, 2]] = 0

    current_linear = int(overlap_linear_index)
    current_coord = _matlab_position_from_linear_index(current_linear, shape)
    # Corrected: edge_size_map contains 1-based labels. Subtract 1 to get index.
    current_scale_label = int(
        edge_size_map[int(current_coord[0]), int(current_coord[1]), int(current_coord[2])]
    )
    current_scale_index = int(
        np.clip(current_scale_label - 1, 0, max(len(lumen_radius_microns) - 1, 0))
    )
    if len(lumen_radius_microns) == 0:
        return None
    max_edge_length_in_microns = float(max_edge_length_per_origin_radius) * float(
        lumen_radius_microns[current_scale_index]
    )
    max_edge_length_in_voxels = int(
        round(float(np.max(max_edge_length_in_microns / microns_per_voxel))) + 1
    )
    max_number_of_indices = max(1, max_edge_length_in_voxels)
    cube_offsets, cube_distances = _matlab_frontier_offsets(int(strel_apothem), microns_per_voxel)

    previous_indices_visited: list[int] = []
    pointer_index_map: dict[int, int] = {}
    available_energy_map: dict[int, float] = {}
    distance_map: dict[int, float] = {current_linear: 1.0}
    pointer_energy_map: dict[int, float] = {}
    terminal_vertex_index: int | None = None
    terminal_kind: str | None = None
    current_energy = float(
        edge_energy_map[int(current_coord[0]), int(current_coord[1]), int(current_coord[2])]
    )
    if current_energy >= 0.0:
        return None

    number_of_indices = 0
    safe_lower = np.asarray([int(strel_apothem)] * 3, dtype=np.int32)
    safe_upper: np.ndarray = np.asarray(shape, dtype=np.int32) - int(strel_apothem)
    there_exists_possible_move = bool(
        np.all(current_coord >= safe_lower) and np.all(current_coord < safe_upper)
    )

    while number_of_indices < max_number_of_indices and there_exists_possible_move:
        number_of_indices += 1
        previous_indices_visited.append(current_linear)
        current_energy = float(
            edge_energy_map[int(current_coord[0]), int(current_coord[1]), int(current_coord[2])]
        )
        pointer_energy_map[current_linear] = float("-inf")

        neighbor_coords = current_coord[None, :] + cube_offsets
        valid_mask = (
            (neighbor_coords[:, 0] >= 0)
            & (neighbor_coords[:, 0] < shape[0])
            & (neighbor_coords[:, 1] >= 0)
            & (neighbor_coords[:, 1] < shape[1])
            & (neighbor_coords[:, 2] >= 0)
            & (neighbor_coords[:, 2] < shape[2])
        )
        valid_coords = np.asarray(neighbor_coords[valid_mask], dtype=np.int32)
        valid_distances = np.asarray(cube_distances[valid_mask], dtype=np.float32)
        valid_linear = _matlab_linear_indices_from_points(valid_coords.astype(np.float32), shape)

        new_indices_considered: list[int] = []
        for _neighbor_coord, neighbor_linear, neighbor_distance in zip(
            valid_coords,
            valid_linear.tolist(),
            valid_distances.tolist(),
        ):
            previous_pointer_energy = float(pointer_energy_map.get(int(neighbor_linear), 0.0))
            if previous_pointer_energy <= current_energy:
                continue
            new_distance = float(distance_map.get(current_linear, 1.0)) + float(neighbor_distance)
            pointer_index_map[int(neighbor_linear)] = number_of_indices
            pointer_energy_map[int(neighbor_linear)] = current_energy
            distance_map[int(neighbor_linear)] = new_distance
            if new_distance < max_edge_length_in_microns:
                new_indices_considered.append(int(neighbor_linear))

        if terminal_kind is not None:
            trace_linear: list[int] = []
            tracing_linear = current_linear
            while pointer_index_map.get(tracing_linear, 0) > 0:
                trace_linear.append(tracing_linear)
                tracing_linear = previous_indices_visited[pointer_index_map[tracing_linear] - 1]
            trace_linear.append(tracing_linear)
            trace_linear.reverse()
            trace_coords = np.asarray(
                [
                    _matlab_position_from_linear_index(int(linear_index), shape)
                    for linear_index in trace_linear
                ],
                dtype=np.float32,
            )
            trace_energy = energy[
                trace_coords[:, 0].astype(np.int32),
                trace_coords[:, 1].astype(np.int32),
                trace_coords[:, 2].astype(np.int32),
            ].astype(np.float32, copy=False)
            if scale_indices is not None and scale_indices.size:
                # Corrected: scale_indices is 0-based. Convert to 1-based for trace metadata.
                trace_scale = scale_indices[
                    trace_coords[:, 0].astype(np.int32),
                    trace_coords[:, 1].astype(np.int32),
                    trace_coords[:, 2].astype(np.int32),
                ].astype(np.float32, copy=False) + np.float32(1.0)
            else:
                # edge_size_map is already 1-based.
                trace_scale = edge_size_map[
                    trace_coords[:, 0].astype(np.int32),
                    trace_coords[:, 1].astype(np.int32),
                    trace_coords[:, 2].astype(np.int32),
                ].astype(np.float32, copy=False)
            metric = float(np.max(trace_energy)) if trace_energy.size else float("inf")
            if terminal_kind == "existing_vertex":
                return {
                    "kind": "existing_vertex",
                    "terminal_vertex": int(terminal_vertex_index)
                    if terminal_vertex_index is not None
                    else -1,
                    "trace": trace_coords,
                    "scale_trace": trace_scale.copy(),
                    "energy_trace": trace_energy.copy(),
                    "metric": metric,
                    "length": len(trace_coords),
                }
            return {
                "kind": "new_vertex",
                "coord": trace_coords[-1].astype(np.int32, copy=False),
                "scale": int(trace_scale[-1]) if len(trace_scale) else 0,
                "energy": float(trace_energy[-1]) if len(trace_energy) else float("inf"),
                "trace": trace_coords,
                "scale_trace": trace_scale.copy(),
                "energy_trace": trace_energy.copy(),
                "metric": metric,
                "length": len(trace_coords),
            }

        for neighbor_linear in new_indices_considered:
            neighbor_coord = _matlab_position_from_linear_index(int(neighbor_linear), shape)
            available_energy_map[int(neighbor_linear)] = float(
                edge_energy_map[
                    int(neighbor_coord[0]),
                    int(neighbor_coord[1]),
                    int(neighbor_coord[2]),
                ]
            )

        available_energy_map[current_linear] = 0.0
        if not available_energy_map:
            break
        min_energy, next_linear = min(
            (float(energy_value), int(linear_index))
            for linear_index, energy_value in available_energy_map.items()
        )
        if min_energy >= 0.0:
            break

        current_linear = int(next_linear)
        current_coord = _matlab_position_from_linear_index(current_linear, shape)
        terminal_label = int(
            vertex_center_image[int(current_coord[0]), int(current_coord[1]), int(current_coord[2])]
        )
        terminal_vertex_index = None
        terminal_kind = None
        if terminal_label > 0:
            terminal_vertex_index = terminal_label - 1
            terminal_kind = "existing_vertex"
        else:
            current_scale = int(
                np.clip(
                    edge_size_map[
                        int(current_coord[0]), int(current_coord[1]), int(current_coord[2])
                    ],
                    0,
                    max(len(lumen_radius_microns) - 1, 0),
                )
            )
            if _can_place_bridge_vertex(
                current_coord,
                current_scale,
                vertex_volume_image,
                lumen_radius_pixels_axes,
                shape,
            ):
                terminal_vertex_index = -1
                terminal_kind = "new_vertex"

        there_exists_possible_move = bool(min_energy < 0.0)

    return None


def add_vertices_to_edges_matlab_style(
    chosen_edges: dict[str, Any],
    vertices: dict[str, Any],
    *,
    energy: Float32Array,
    scale_indices: Int16Array | None,
    microns_per_voxel: Float32Array,
    lumen_radius_microns: Float32Array,
    lumen_radius_pixels_axes: Float32Array,
    size_of_image: tuple[int, int, int],
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Mirror MATLAB ``add_vertices_to_edges`` structural bridge insertion."""
    bridge_params = params or {}
    strel_apothem = int(
        bridge_params.get(
            "space_strel_apothem_edges",
            bridge_params.get("space_strel_apothem", 1),
        )
    )
    max_edge_length_per_origin_radius = float(
        bridge_params.get("max_edge_length_per_origin_radius", 60.0)
    )
    traces = [
        np.asarray(trace, dtype=np.float32).copy() for trace in chosen_edges.get("traces", [])
    ]
    if not traces:
        chosen_edges["bridge_vertex_positions"] = np.empty((0, 3), dtype=np.float32)
        chosen_edges["bridge_vertex_scales"] = np.empty((0,), dtype=np.int16)
        chosen_edges["bridge_vertex_energies"] = np.empty((0,), dtype=np.float32)
        chosen_edges["bridge_edges"] = _matlab_bridge_build_bridge_payload(
            bridge_connections=[],
            bridge_edges2vertices=[],
            bridge_traces=[],
            bridge_scale_traces=[],
            bridge_energy_traces=[],
            bridge_mean_energies=np.empty((0,), dtype=np.float32),
        )
        return chosen_edges

    scale_traces = [
        np.asarray(trace, dtype=np.float32).reshape(-1).copy()
        for trace in chosen_edges.get("scale_traces", [])
    ]
    energy_traces = [
        np.asarray(trace, dtype=np.float32).reshape(-1).copy()
        for trace in chosen_edges.get("energy_traces", [])
    ]
    connections = np.asarray(chosen_edges.get("connections", []), dtype=np.int32).reshape(-1, 2)
    connection_sources = normalize_candidate_connection_sources(
        chosen_edges.get("connection_sources"),
        len(connections),
    )
    chosen_candidate_indices = np.asarray(
        chosen_edges.get("chosen_candidate_indices", np.arange(len(connections), dtype=np.int32)),
        dtype=np.int32,
    ).reshape(-1)

    vertex_positions = np.asarray(vertices["positions"], dtype=np.float32)
    vertex_scales = np.asarray(vertices["scales"], dtype=np.int16).reshape(-1)
    edge_index_cells = [
        _matlab_linear_indices_from_points(np.asarray(trace, dtype=np.float32), size_of_image)
        for trace in traces
    ]
    vertex_linear_indices = _matlab_linear_indices_from_points(vertex_positions, size_of_image)
    overlap_indices = _matlab_repeated_endpoint_interior_indices(
        edge_index_cells,
        vertex_linear_indices,
    )

    if not overlap_indices:
        # Pass-through if no structural bridge candidates found
        return chosen_edges

    # MATLAB sorts new structural vertices by local energy before processing them
    overlap_energies = energy.ravel(order="F")[np.asarray(overlap_indices, dtype=np.int64)]
    sort_order = np.argsort(overlap_energies, kind="stable")
    overlap_indices = [overlap_indices[i] for i in sort_order.tolist()]

    (
        traces,
        scale_traces,
        energy_traces,
        connection_sources,
        chosen_candidate_indices,
        connections,
        inactive_traces,
        inactive_scale_traces,
        inactive_energy_traces,
        inactive_connection_sources,
        inactive_candidate_indices,
        inactive_connections,
    ) = _matlab_bridge_separate_active_edges(
        traces=traces,
        scale_traces=scale_traces,
        energy_traces=energy_traces,
        connection_sources=connection_sources,
        chosen_candidate_indices=chosen_candidate_indices,
        connections=connections,
        edge_index_cells=edge_index_cells,
        overlap_indices=overlap_indices,
    )

    bridge_vertex_positions: list[np.ndarray] = []
    bridge_vertex_scales: list[int] = []
    bridge_vertex_energies: list[float] = []
    bridge_connections: list[list[int]] = []
    bridge_edges2vertices: list[list[int]] = []
    bridge_traces: list[Float32Array] = []
    bridge_scale_traces: list[Float32Array] = []
    bridge_energy_traces: list[Float32Array] = []

    for overlap_linear_index in overlap_indices:
        if not traces:
            break
        current_edge_index_cells = [
            _matlab_linear_indices_from_points(np.asarray(trace, dtype=np.float32), size_of_image)
            for trace in traces
        ]
        occurrences: list[tuple[int, int]] = []
        for edge_index, edge_indices_at_edge in enumerate(current_edge_index_cells):
            match_positions = np.flatnonzero(edge_indices_at_edge == overlap_linear_index)
            for match_position in match_positions.tolist():
                occurrences.append((edge_index, int(match_position)))
        if len(occurrences) < 2:
            continue

        child_edges = [
            (edge_index, overlap_position)
            for edge_index, overlap_position in occurrences
            if overlap_position == len(current_edge_index_cells[edge_index]) - 1
        ]
        parent_edges = [
            (edge_index, overlap_position)
            for edge_index, overlap_position in occurrences
            if overlap_position != len(current_edge_index_cells[edge_index]) - 1
        ]
        if not child_edges or not parent_edges:
            continue

        all_vertex_positions = (
            np.vstack([vertex_positions, np.vstack(bridge_vertex_positions)])
            if bridge_vertex_positions
            else vertex_positions
        )
        all_vertex_scales = (
            np.concatenate([vertex_scales, np.asarray(bridge_vertex_scales, dtype=np.int16)])
            if bridge_vertex_scales
            else vertex_scales
        )
        vertex_volume_image = paint_vertex_image(
            all_vertex_positions,
            all_vertex_scales,
            np.asarray(lumen_radius_pixels_axes, dtype=np.float32),
            size_of_image,
        ).astype(bool, copy=False)
        vertex_center_image = paint_vertex_center_image(all_vertex_positions, size_of_image)

        target = _matlab_bridge_search_target(
            overlap_linear_index,
            traces,
            scale_traces,
            child_edges=child_edges,
            scale_indices=scale_indices,
            energy=energy,
            vertex_center_image=vertex_center_image,
            vertex_volume_image=vertex_volume_image,
            lumen_radius_pixels_axes=np.asarray(lumen_radius_pixels_axes, dtype=np.float32),
            lumen_radius_microns=np.asarray(lumen_radius_microns, dtype=np.float32),
            microns_per_voxel=np.asarray(microns_per_voxel, dtype=np.float32),
            image_shape=size_of_image,
            strel_apothem=strel_apothem,
            max_edge_length_per_origin_radius=max_edge_length_per_origin_radius,
        )
        if target is None:
            continue

        if target["kind"] == "existing_vertex":
            target_vertex_index = int(target["terminal_vertex"])
            for child_edge_index, _overlap_position in child_edges:
                connections[child_edge_index, 1] = target_vertex_index
        else:
            target_vertex_index = int(len(vertex_positions) + len(bridge_vertex_positions))
            bridge_vertex_positions.append(np.asarray(target["coord"], dtype=np.float32).copy())
            bridge_vertex_scales.append(int(target["scale"]))
            bridge_vertex_energies.append(float(target["energy"]))
            for child_edge_index, _overlap_position in child_edges:
                connections[child_edge_index, 1] = target_vertex_index

            parent_indices_desc = sorted(
                {edge_index for edge_index, _ in parent_edges}, reverse=True
            )
            replacement_traces: list[np.ndarray] = []
            replacement_scale_traces: list[np.ndarray] = []
            replacement_energy_traces: list[np.ndarray] = []
            replacement_connections: list[np.ndarray] = []
            replacement_sources: list[str] = []
            replacement_candidate_indices: list[int] = []
            for parent_edge_index in parent_indices_desc:
                overlap_position = next(
                    int(position)
                    for edge_index, position in parent_edges
                    if edge_index == parent_edge_index
                )
                parent_connection = np.asarray(connections[parent_edge_index], dtype=np.int32)
                parent_trace = np.asarray(traces[parent_edge_index], dtype=np.float32)
                parent_scale_trace = np.asarray(scale_traces[parent_edge_index], dtype=np.float32)
                parent_energy_trace = np.asarray(energy_traces[parent_edge_index], dtype=np.float32)
                replacement_traces.extend(
                    [
                        parent_trace[: overlap_position + 1].copy(),
                        parent_trace[overlap_position:].copy(),
                    ]
                )
                replacement_scale_traces.extend(
                    [
                        parent_scale_trace[: overlap_position + 1].copy(),
                        parent_scale_trace[overlap_position:].copy(),
                    ]
                )
                replacement_energy_traces.extend(
                    [
                        parent_energy_trace[: overlap_position + 1].copy(),
                        parent_energy_trace[overlap_position:].copy(),
                    ]
                )
                replacement_connections.extend(
                    [
                        np.asarray([parent_connection[0], target_vertex_index], dtype=np.int32),
                        np.asarray([target_vertex_index, parent_connection[1]], dtype=np.int32),
                    ]
                )
                replacement_sources.extend(
                    [
                        connection_sources[parent_edge_index],
                        connection_sources[parent_edge_index],
                    ]
                )
                parent_candidate_index = (
                    int(chosen_candidate_indices[parent_edge_index])
                    if parent_edge_index < len(chosen_candidate_indices)
                    else -1
                )
                replacement_candidate_indices.extend(
                    [parent_candidate_index, parent_candidate_index]
                )

                del traces[parent_edge_index]
                del scale_traces[parent_edge_index]
                del energy_traces[parent_edge_index]
                connections = np.delete(connections, parent_edge_index, axis=0)
                del connection_sources[parent_edge_index]
                chosen_candidate_indices = np.delete(
                    chosen_candidate_indices, parent_edge_index, axis=0
                )

            traces.extend(replacement_traces)
            scale_traces.extend(replacement_scale_traces)
            energy_traces.extend(replacement_energy_traces)
            if replacement_connections:
                connections = np.vstack(
                    [connections, np.vstack(replacement_connections).astype(np.int32, copy=False)]
                )
            connection_sources.extend(replacement_sources)
            chosen_candidate_indices = np.concatenate(
                [
                    chosen_candidate_indices,
                    np.asarray(replacement_candidate_indices, dtype=np.int32),
                ]
            )

        bridge_connections.append([target_vertex_index, -1])
        bridge_edges2vertices.append([target_vertex_index + 1, 0])
        bridge_traces.append(np.asarray(target["trace"], dtype=np.float32).copy())
        bridge_scale_traces.append(np.asarray(target["scale_trace"], dtype=np.float32).copy())
        bridge_energy_traces.append(np.asarray(target["energy_trace"], dtype=np.float32).copy())

    if inactive_traces:
        traces = inactive_traces + traces
        scale_traces = inactive_scale_traces + scale_traces
        energy_traces = inactive_energy_traces + energy_traces
        connection_sources = inactive_connection_sources + connection_sources
        connections = np.vstack([inactive_connections, connections]).astype(np.int32, copy=False)
        chosen_candidate_indices = np.concatenate(
            [inactive_candidate_indices, chosen_candidate_indices]
        )

    bridge_mean_energies = np.asarray(
        [
            float(np.mean(np.asarray(energy_trace, dtype=np.float32)))
            for energy_trace in bridge_energy_traces
        ],
        dtype=np.float32,
    )
    diagnostics = chosen_edges.get("diagnostics", {})
    if not isinstance(diagnostics, dict):
        diagnostics = {}
    diagnostics["bridge_vertex_count"] = len(bridge_vertex_positions)
    diagnostics["bridge_edge_count"] = len(bridge_traces)

    metrics = _matlab_edge_metrics(energy_traces)
    keep_indices = prepare_candidate_indices_for_cleanup(
        connections,
        metrics,
        energy_traces,
        diagnostics,
        reject_nonnegative_energy_edges=False,
    )
    rebuilt = build_selected_edges_result(
        keep_indices,
        traces,
        connections,
        metrics,
        energy_traces,
        scale_traces,
        connection_sources,
        vertex_positions,
        diagnostics,
    )
    if chosen_candidate_indices.size:
        rebuilt["chosen_candidate_indices"] = chosen_candidate_indices[
            np.asarray(keep_indices, dtype=np.int32)
        ]
    else:
        rebuilt["chosen_candidate_indices"] = np.asarray(keep_indices, dtype=np.int32)
    rebuilt["bridge_vertex_positions"] = (
        np.vstack(bridge_vertex_positions).astype(np.float32, copy=False)
        if bridge_vertex_positions
        else np.empty((0, 3), dtype=np.float32)
    )
    rebuilt["bridge_vertex_scales"] = (
        np.asarray(bridge_vertex_scales, dtype=np.int16)
        if bridge_vertex_scales
        else np.empty((0,), dtype=np.int16)
    )
    rebuilt["bridge_vertex_energies"] = (
        np.asarray(bridge_vertex_energies, dtype=np.float32)
        if bridge_vertex_energies
        else np.empty((0,), dtype=np.float32)
    )
    rebuilt["bridge_edges"] = _matlab_bridge_build_bridge_payload(
        bridge_connections=bridge_connections,
        bridge_edges2vertices=bridge_edges2vertices,
        bridge_traces=bridge_traces,
        bridge_scale_traces=bridge_scale_traces,
        bridge_energy_traces=bridge_energy_traces,
        bridge_mean_energies=bridge_mean_energies,
    )
    rebuilt["lumen_radius_microns"] = np.asarray(lumen_radius_microns, dtype=np.float32).copy()
    return cast("dict[str, Any]", rebuilt)


__all__ = [
    "add_vertices_to_edges_matlab_style",
]
