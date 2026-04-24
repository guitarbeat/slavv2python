"""MATLAB-style frontier tracing for a single origin vertex."""

from __future__ import annotations

from typing import Any

import numpy as np

from .._edge_payloads import _empty_edge_diagnostics
from ..edge_primitives import (
    _edge_metric_from_energy_trace,
    _trace_energy_series,
    _trace_scale_series,
)
from .common import (
    BoolArray,
    Int32Array,
    _coord_to_matlab_linear_index,
    _matlab_frontier_adjusted_neighbor_energies,
    _matlab_frontier_edge_budget,
    _matlab_frontier_insert_available_location,
    _matlab_frontier_pop_best_available_location,
    _matlab_frontier_scale_offsets,
    _matlab_frontier_select_seed_moves,
    _matlab_linear_index_to_coord,
    _path_coords_from_linear_indices,
)
from .frontier_resolution import _build_frontier_lifecycle_event


def _trace_origin_edges_matlab_frontier(
    energy: np.ndarray,
    scale_indices: np.ndarray | None,
    vertex_positions: np.ndarray,
    vertex_scales: np.ndarray,
    lumen_radius_microns: np.ndarray,
    microns_per_voxel: np.ndarray,
    vertex_center_image: np.ndarray,
    origin_vertex_idx: int,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Trace a single origin using a MATLAB-style best-first voxel frontier."""
    shape = energy.shape
    max_edges_per_vertex = _matlab_frontier_edge_budget(params)
    max_length_ratio = float(params.get("max_edge_length_per_origin_radius", 60.0))
    step_size_per_origin_radius = float(params.get("step_size_per_origin_radius", 1.0))
    origin_coord = np.rint(vertex_positions[origin_vertex_idx]).astype(np.int32)
    origin_coord[0] = np.clip(origin_coord[0], 0, shape[0] - 1)
    origin_coord[1] = np.clip(origin_coord[1], 0, shape[1] - 1)
    origin_coord[2] = np.clip(origin_coord[2], 0, shape[2] - 1)
    origin_linear = _coord_to_matlab_linear_index(origin_coord, shape)
    origin_position_microns = origin_coord.astype(np.float64) * microns_per_voxel
    origin_scale = int(vertex_scales[origin_vertex_idx])
    origin_radius_microns = float(lumen_radius_microns[origin_scale])
    max_edge_length_microns = max_length_ratio * origin_radius_microns
    max_edge_length_voxels = int(np.round(max_edge_length_microns / np.min(microns_per_voxel))) + 1
    max_number_of_indices = max(1, max_edge_length_voxels * max_edges_per_vertex)

    diagnostics = _empty_edge_diagnostics()
    traces: list[np.ndarray] = []
    connections: list[list[int]] = []
    metrics: list[float] = []
    energy_traces: list[np.ndarray] = []
    scale_traces: list[np.ndarray] = []
    origin_indices: list[int] = []
    frontier_lifecycle_events: list[dict[str, Any]] = []
    terminal_paths_linear: list[list[int]] = []
    terminal_pairs: list[tuple[int, int]] = []
    displacement_vectors: list[np.ndarray] = []
    valid_terminal_count = 0
    terminal_hit_budget_count = 0
    previous_indices_visited: list[int] = []
    pointer_index_map: dict[int, int] = {origin_linear: 0}
    pointer_energy_map: dict[int, float] = {}
    distance_map: dict[int, float] = {origin_linear: 0.0}
    branch_order_map: dict[int, int] = {origin_linear: 0}
    predecessor_linear_map: dict[int, int] = {}
    propagated_scale_map: dict[int, int] = {origin_linear: origin_scale}
    available_map: dict[int, float] = {}
    available_entries: list[tuple[float, int]] = []

    current_linear = origin_linear

    while (
        terminal_hit_budget_count < max_edges_per_vertex
        and len(previous_indices_visited) < max_number_of_indices
    ):
        from .. import edge_candidates as edge_candidates_facade

        current_coord = _matlab_linear_index_to_coord(current_linear, shape)
        current_distance_microns = float(distance_map.get(current_linear, 0.0))
        current_propagated_scale = int(propagated_scale_map.get(current_linear, origin_scale))
        offsets, offset_distances = _matlab_frontier_scale_offsets(
            current_propagated_scale,
            lumen_radius_microns,
            microns_per_voxel,
            step_size_per_origin_radius=step_size_per_origin_radius,
        )
        terminal_vertex_idx = (
            int(vertex_center_image[current_coord[0], current_coord[1], current_coord[2]]) - 1
        )
        if terminal_vertex_idx == origin_vertex_idx:
            terminal_vertex_idx = -1

        previous_indices_visited.append(current_linear)
        current_visit_order = len(previous_indices_visited)
        pointer_energy_map[current_linear] = float("-inf")

        neighbor_coords: Int32Array = np.asarray(current_coord + offsets, dtype=np.int32)
        valid_mask = (
            (neighbor_coords[:, 0] >= 0)
            & (neighbor_coords[:, 0] < shape[0])
            & (neighbor_coords[:, 1] >= 0)
            & (neighbor_coords[:, 1] < shape[1])
            & (neighbor_coords[:, 2] >= 0)
            & (neighbor_coords[:, 2] < shape[2])
        )
        neighbor_coords = np.asarray(neighbor_coords[valid_mask], dtype=np.int32)
        neighbor_distances = offset_distances[valid_mask]
        neighbor_offsets = offsets[valid_mask]
        current_forward_unit = None
        predecessor_linear = predecessor_linear_map.get(current_linear)
        if predecessor_linear is not None:
            predecessor_coord = _matlab_linear_index_to_coord(predecessor_linear, shape)
            forward_vector = (
                current_coord.astype(np.float64) - predecessor_coord.astype(np.float64)
            ) * microns_per_voxel
            forward_norm = float(np.linalg.norm(forward_vector))
            if forward_norm > 1e-12:
                current_forward_unit = forward_vector / forward_norm

        neighbor_scale_values = None
        if scale_indices is not None and len(neighbor_coords):
            neighbor_scale_values = scale_indices[
                neighbor_coords[:, 0],
                neighbor_coords[:, 1],
                neighbor_coords[:, 2],
            ].astype(np.int16, copy=False)

        raw_neighbor_energies = energy[
            neighbor_coords[:, 0],
            neighbor_coords[:, 1],
            neighbor_coords[:, 2],
        ].astype(np.float32, copy=False)
        adjusted_neighbor_energies = _matlab_frontier_adjusted_neighbor_energies(
            raw_neighbor_energies,
            neighbor_offsets=neighbor_offsets,
            neighbor_distances_microns=neighbor_distances,
            neighbor_scale_indices=neighbor_scale_values,
            propagated_scale_index=current_propagated_scale,
            current_distance_microns=current_distance_microns,
            origin_radius_microns=origin_radius_microns,
            current_forward_unit=current_forward_unit,
            microns_per_voxel=microns_per_voxel,
            lumen_radius_microns=lumen_radius_microns,
            distance_tolerance=3.0,
        )
        new_coords: list[np.ndarray] = []
        new_distances: list[float] = []
        new_adjusted_energies: list[float] = []
        current_branch_order = int(branch_order_map.get(current_linear, 0))
        for coord_row, distance, adjusted_energy in zip(
            neighbor_coords,
            neighbor_distances,
            adjusted_neighbor_energies,
        ):
            coord: Int32Array = np.asarray(coord_row, dtype=np.int32)
            linear_index = _coord_to_matlab_linear_index(coord, shape)
            if linear_index in pointer_index_map:
                continue
            pointer_index_map[linear_index] = current_visit_order
            pointer_energy_map[linear_index] = float(adjusted_energy)
            predecessor_linear_map[linear_index] = current_linear
            propagated_scale_map[linear_index] = current_propagated_scale
            distance_map[linear_index] = current_distance_microns + float(distance)
            new_coords.append(coord.astype(np.int32, copy=False))
            new_distances.append(float(distance_map[linear_index]))
            new_adjusted_energies.append(float(adjusted_energy))

        new_coords_array: Int32Array = np.zeros((0, 3), dtype=np.int32)
        if new_coords:
            new_coords_array = np.asarray(new_coords, dtype=np.int32)
            new_distances_array = np.asarray(new_distances, dtype=np.float32)
            new_adjusted_energies_array = np.asarray(new_adjusted_energies, dtype=np.float32)
            diagnostics["stop_reason_counts"]["length_limit"] += int(
                np.sum(new_distances_array >= max_edge_length_microns)
            )
            within_length: BoolArray = new_distances_array < max_edge_length_microns
            new_coords_array = new_coords_array[within_length]
            new_adjusted_energies_array = new_adjusted_energies_array[within_length]
            if len(new_coords_array) and valid_terminal_count > 0:
                kept_after_prune = (
                    edge_candidates_facade._prune_frontier_indices_beyond_found_vertices(
                        new_coords_array,
                        origin_position_microns,
                        displacement_vectors,
                        microns_per_voxel,
                    )
                )
                if len(kept_after_prune) != len(new_coords_array):
                    kept_linear = {
                        _coord_to_matlab_linear_index(coord, shape) for coord in kept_after_prune
                    }
                    keep_mask = np.array(
                        [
                            _coord_to_matlab_linear_index(coord, shape) in kept_linear
                            for coord in new_coords_array
                        ],
                        dtype=bool,
                    )
                    new_adjusted_energies_array = new_adjusted_energies_array[keep_mask]
                new_coords_array = kept_after_prune

        if terminal_vertex_idx >= 0:
            terminal_hit_budget_count += 1
            diagnostics["stop_reason_counts"]["terminal_frontier_hit"] += 1
            diagnostics.setdefault("frontier_per_origin_terminal_hits", {})
            diagnostics["frontier_per_origin_terminal_hits"][str(origin_vertex_idx)] = (
                int(diagnostics["frontier_per_origin_terminal_hits"].get(str(origin_vertex_idx), 0))
                + 1
            )
            terminal_hit_sequence = int(
                diagnostics["frontier_per_origin_terminal_hits"].get(str(origin_vertex_idx), 0)
            )
            path_linear = [current_linear]
            tracing_linear = current_linear
            while int(pointer_index_map.get(tracing_linear, 0)) > 0:
                tracing_linear = previous_indices_visited[
                    int(pointer_index_map[tracing_linear]) - 1
                ]
                path_linear.append(tracing_linear)

            origin_idx, terminal_idx, resolution_reason, resolution_debug = (
                edge_candidates_facade._normalize_frontier_resolution_result(
                    edge_candidates_facade._resolve_frontier_edge_connection_details(
                        path_linear,
                        terminal_vertex_idx,
                        origin_vertex_idx,
                        terminal_paths_linear,
                        terminal_pairs,
                        pointer_index_map,
                        energy,
                        shape,
                    )
                )
            )
            diagnostics.setdefault("frontier_terminal_resolution_counts", {})
            diagnostics["frontier_terminal_resolution_counts"][resolution_reason] = (
                int(diagnostics["frontier_terminal_resolution_counts"].get(resolution_reason, 0))
                + 1
            )

            record_rejected_child_path = resolution_reason != "rejected_child_better_than_parent"
            if origin_idx is not None or terminal_idx is not None or record_rejected_child_path:
                path_record_index = terminal_hit_budget_count
                for path_index in path_linear[:-1]:
                    pointer_index_map[path_index] = -path_record_index

                terminal_paths_linear.append(path_linear)
                terminal_pairs.append(
                    (
                        -1 if terminal_idx is None else int(terminal_idx),
                        -1 if origin_idx is None else int(origin_idx),
                    )
                )

            current_position = current_coord.astype(np.float64) * microns_per_voxel
            displacement = current_position - origin_position_microns
            displacement_norm_sq = float(np.sum(displacement**2))
            if displacement_norm_sq > 0:
                displacement_vectors.append(displacement / displacement_norm_sq)
            else:
                displacement_vectors.append(np.zeros((3,), dtype=np.float64))

            if origin_idx is not None and terminal_idx is not None:
                diagnostics.setdefault("frontier_per_origin_terminal_accepts", {})
                diagnostics["frontier_per_origin_terminal_accepts"][str(origin_vertex_idx)] = (
                    int(
                        diagnostics["frontier_per_origin_terminal_accepts"].get(
                            str(origin_vertex_idx), 0
                        )
                    )
                    + 1
                )
                valid_terminal_count += 1
                edge_trace = _path_coords_from_linear_indices(path_linear, shape)
                energy_trace = _trace_energy_series(edge_trace, energy)
                scale_trace = _trace_scale_series(edge_trace, scale_indices)
                traces.append(edge_trace)
                connections.append([int(origin_idx), int(terminal_idx)])
                metrics.append(_edge_metric_from_energy_trace(energy_trace))
                energy_traces.append(energy_trace)
                scale_traces.append(scale_trace)
                origin_indices.append(origin_vertex_idx)
                diagnostics["terminal_direct_hit_count"] += 1
                frontier_lifecycle_events.append(
                    _build_frontier_lifecycle_event(
                        seed_origin_idx=origin_vertex_idx,
                        terminal_vertex_idx=terminal_vertex_idx,
                        origin_idx=int(origin_idx),
                        terminal_idx=int(terminal_idx),
                        resolution_reason=resolution_reason,
                        resolution_debug=resolution_debug,
                        terminal_hit_sequence=terminal_hit_sequence,
                        local_candidate_index=len(connections) - 1,
                    )
                )
            else:
                diagnostics.setdefault("frontier_per_origin_terminal_rejections", {})
                diagnostics["frontier_per_origin_terminal_rejections"][str(origin_vertex_idx)] = (
                    int(
                        diagnostics["frontier_per_origin_terminal_rejections"].get(
                            str(origin_vertex_idx), 0
                        )
                    )
                    + 1
                )
                frontier_lifecycle_events.append(
                    _build_frontier_lifecycle_event(
                        seed_origin_idx=origin_vertex_idx,
                        terminal_vertex_idx=terminal_vertex_idx,
                        origin_idx=None,
                        terminal_idx=None,
                        resolution_reason=resolution_reason,
                        resolution_debug=resolution_debug,
                        terminal_hit_sequence=terminal_hit_sequence,
                    )
                )
            if len(new_coords_array):
                for coord in new_coords_array:
                    linear_index = _coord_to_matlab_linear_index(coord, shape)
                    available_map.pop(linear_index, None)
        else:
            if len(new_coords_array):
                selected_seed_moves = _matlab_frontier_select_seed_moves(
                    new_adjusted_energies_array,
                    neighbor_offsets=new_coords_array - current_coord,
                    microns_per_voxel=microns_per_voxel,
                    current_is_source=predecessor_linear is None,
                    edge_budget=max_edges_per_vertex,
                    current_branch_order=current_branch_order,
                )
                for selected_index, selected_branch_order in selected_seed_moves:
                    if selected_branch_order >= max_edges_per_vertex:
                        continue
                    coord = new_coords_array[int(selected_index)]
                    linear_index = _coord_to_matlab_linear_index(coord, shape)
                    branch_order_map[linear_index] = int(selected_branch_order)
                    available_energy = float(new_adjusted_energies_array[int(selected_index)])
                    available_map[linear_index] = available_energy
                    _matlab_frontier_insert_available_location(
                        available_entries,
                        linear_index=linear_index,
                        energy=available_energy,
                    )

        available_map.pop(current_linear, None)
        next_current = None
        stopped_on_nonnegative = False
        while True:
            next_available = _matlab_frontier_pop_best_available_location(
                available_entries,
                available_map,
            )
            if next_available is None:
                break
            candidate_energy, candidate_linear = next_available
            if candidate_energy >= 0:
                available_map.pop(candidate_linear, None)
                diagnostics["stop_reason_counts"]["frontier_exhausted_nonnegative"] += 1
                available_entries.clear()
                stopped_on_nonnegative = True
                next_current = None
                break
            next_current = int(candidate_linear)
            break

        if next_current is None:
            if not available_map and not stopped_on_nonnegative:
                diagnostics["stop_reason_counts"]["frontier_exhausted_nonnegative"] += 1
            break

        current_linear = next_current

    return {
        "origin_index": origin_vertex_idx,
        "candidate_source": "frontier",
        "traces": traces,
        "connections": connections,
        "metrics": metrics,
        "energy_traces": energy_traces,
        "scale_traces": scale_traces,
        "origin_indices": [origin_vertex_idx] * len(traces),
        "connection_sources": ["frontier"] * len(traces),
        "frontier_lifecycle_events": frontier_lifecycle_events,
        "diagnostics": diagnostics,
    }
