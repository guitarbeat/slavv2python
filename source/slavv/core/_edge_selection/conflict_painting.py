"""Conflict-painting chooser for standard edge cleanup."""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from .._edge_payloads import _empty_edges_result
from .cleanup import (
    clean_edges_cycles_python,
    clean_edges_orphans_python,
    clean_edges_vertex_degree_excess_python,
)
from .payloads import (
    build_selected_edges_result,
    initialize_edge_selection_diagnostics,
    normalize_candidate_connection_sources,
    prepare_candidate_indices_for_cleanup,
)


def _construct_structuring_element_offsets_matlab(radii: np.ndarray) -> np.ndarray:
    """Construct MATLAB-shaped ellipsoid offsets using the original radius equation."""
    radii = np.asarray(radii, dtype=np.float32).reshape(3)
    rounded = np.rint(radii).astype(np.int32)
    safe_radii = np.maximum(radii.astype(np.float64), 1e-6)

    offsets = np.array(
        [
            [y, x, z]
            for z in range(-rounded[2], rounded[2] + 1)
            for x in range(-rounded[1], rounded[1] + 1)
            for y in range(-rounded[0], rounded[0] + 1)
        ],
        dtype=np.int32,
    )
    distances = (
        (offsets[:, 0].astype(np.float64) ** 2) / (safe_radii[0] ** 2)
        + (offsets[:, 1].astype(np.float64) ** 2) / (safe_radii[1] ** 2)
        + (offsets[:, 2].astype(np.float64) ** 2) / (safe_radii[2] ** 2)
    )
    kept = offsets[distances <= 1.0]
    if kept.size == 0:
        return np.zeros((1, 3), dtype=np.int32)
    return kept.astype(np.int32, copy=False)


def _offset_coords_matlab(
    position: np.ndarray,
    offsets: np.ndarray,
    image_shape: tuple[int, int, int],
) -> np.ndarray:
    """Apply offsets with MATLAB's edge-handling rule of snapping overflow back to center."""
    base = np.rint(position[:3]).astype(np.int32)
    coords = offsets.astype(np.int32, copy=False) + base
    for axis, size in enumerate(image_shape):
        invalid = (coords[:, axis] < 0) | (coords[:, axis] >= size)
        coords[invalid, axis] = base[axis]
    return coords


def _choose_edges_matlab_style(
    candidates: dict[str, Any],
    vertex_positions: np.ndarray,
    vertex_scales: np.ndarray,
    lumen_radius_pixels_axes: np.ndarray,
    image_shape: tuple[int, int, int],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Choose final edges using MATLAB-shaped filtering and cleanup semantics."""
    traces = candidates["traces"]
    connections = np.asarray(candidates["connections"], dtype=np.int32).reshape(-1, 2)
    metrics = np.asarray(candidates["metrics"], dtype=np.float32).reshape(-1)
    energy_traces = candidates["energy_traces"]
    scale_traces = candidates["scale_traces"]
    diagnostics = initialize_edge_selection_diagnostics(candidates, connections, traces)

    if len(traces) == 0:
        empty = cast("dict[str, Any]", _empty_edges_result(vertex_positions))
        empty["diagnostics"] = diagnostics
        return empty

    filtered_indices = prepare_candidate_indices_for_cleanup(
        connections,
        metrics,
        energy_traces,
        diagnostics,
    )
    if not filtered_indices:
        empty = cast("dict[str, Any]", _empty_edges_result(vertex_positions))
        empty["diagnostics"] = diagnostics
        return empty

    connection_sources = normalize_candidate_connection_sources(
        candidates.get("connection_sources"),
        len(connections),
    )

    sigma_per_influence_vertices = float(params.get("sigma_per_influence_vertices", 1.0))
    sigma_per_influence_edges = float(params.get("sigma_per_influence_edges", 0.5))
    vertex_offset_cache: dict[int, np.ndarray] = {}
    edge_offset_cache: dict[int, np.ndarray] = {}
    painted_image = np.zeros(image_shape, dtype=np.int32)
    painted_source_image = np.zeros(image_shape, dtype=np.uint8)
    source_code_by_label = {"unknown": 0, "frontier": 1, "watershed": 2, "fallback": 3}
    source_label_by_code = {code: label for label, code in source_code_by_label.items()}

    def vertex_offsets(scale: int) -> np.ndarray:
        if scale not in vertex_offset_cache:
            radii = sigma_per_influence_vertices * lumen_radius_pixels_axes[scale]
            vertex_offset_cache[scale] = _construct_structuring_element_offsets_matlab(radii)
        return vertex_offset_cache[scale]

    def edge_offsets(scale: int) -> np.ndarray:
        if scale not in edge_offset_cache:
            radii = sigma_per_influence_edges * lumen_radius_pixels_axes[scale]
            edge_offset_cache[scale] = _construct_structuring_element_offsets_matlab(radii)
        return edge_offset_cache[scale]

    chosen_indices: list[int] = []
    for index in filtered_indices:
        start_vertex, end_vertex = (int(value) for value in connections[index])
        current_source = connection_sources[index] if index < len(connection_sources) else "unknown"
        current_source_code = source_code_by_label.get(current_source, 0)
        endpoint_snapshots: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

        for vertex_index in (start_vertex, end_vertex):
            coords = _offset_coords_matlab(
                vertex_positions[vertex_index],
                vertex_offsets(int(vertex_scales[vertex_index])),
                image_shape,
            )
            snapshot = painted_image[coords[:, 0], coords[:, 1], coords[:, 2]].copy()
            source_snapshot = painted_source_image[coords[:, 0], coords[:, 1], coords[:, 2]].copy()
            painted_image[coords[:, 0], coords[:, 1], coords[:, 2]] = 0
            painted_source_image[coords[:, 0], coords[:, 1], coords[:, 2]] = 0
            endpoint_snapshots.append((coords, snapshot, source_snapshot))

        chosen = True
        trace = np.asarray(traces[index], dtype=np.float32)
        scale_trace = np.asarray(scale_traces[index], dtype=np.int16)
        for point_index, point in enumerate(trace):
            scale_value = int(scale_trace[min(point_index, len(scale_trace) - 1)])
            coords = _offset_coords_matlab(point, edge_offsets(scale_value), image_shape)
            if coords.size == 0:
                continue
            conflicting = {
                int(value)
                for value in painted_image[coords[:, 0], coords[:, 1], coords[:, 2]].tolist()
                if int(value) not in {0, start_vertex + 1, end_vertex + 1}
            }
            if conflicting:
                diagnostics["conflict_rejected_count"] += 1
                blocking_sources = {
                    source_label_by_code.get(int(value), "unknown")
                    for value in painted_source_image[
                        coords[:, 0],
                        coords[:, 1],
                        coords[:, 2],
                    ].tolist()
                    if int(value) != 0
                }
                conflict_rejected_by_source = diagnostics.setdefault(
                    "conflict_rejected_by_source",
                    {},
                )
                conflict_rejected_by_source[current_source] = (
                    int(conflict_rejected_by_source.get(current_source, 0)) + 1
                )
                conflict_blocking_source_counts = diagnostics.setdefault(
                    "conflict_blocking_source_counts",
                    {},
                )
                conflict_source_pairs = diagnostics.setdefault("conflict_source_pairs", {})
                for blocking_source in blocking_sources:
                    conflict_blocking_source_counts[blocking_source] = (
                        int(conflict_blocking_source_counts.get(blocking_source, 0)) + 1
                    )
                    pair_key = f"{current_source}->{blocking_source}"
                    conflict_source_pairs[pair_key] = (
                        int(conflict_source_pairs.get(pair_key, 0)) + 1
                    )
                chosen = False
                break

        if chosen:
            for vertex_index, (coords, _snapshot, _source_snapshot) in zip(
                (start_vertex, end_vertex),
                endpoint_snapshots,
            ):
                painted_image[coords[:, 0], coords[:, 1], coords[:, 2]] = vertex_index + 1
                painted_source_image[coords[:, 0], coords[:, 1], coords[:, 2]] = current_source_code
            chosen_indices.append(index)
        else:
            for coords, snapshot, source_snapshot in endpoint_snapshots:
                painted_image[coords[:, 0], coords[:, 1], coords[:, 2]] = snapshot
                painted_source_image[coords[:, 0], coords[:, 1], coords[:, 2]] = source_snapshot

    if not chosen_indices:
        empty = cast("dict[str, Any]", _empty_edges_result(vertex_positions))
        empty["diagnostics"] = diagnostics
        return empty

    chosen_connections = connections[chosen_indices]
    chosen_metrics = metrics[chosen_indices]
    keep_degree = clean_edges_vertex_degree_excess_python(
        chosen_connections,
        chosen_metrics,
        int(params.get("number_of_edges_per_vertex", 4)),
    )
    diagnostics["degree_pruned_count"] = int(np.sum(~keep_degree))

    after_degree_indices = [
        index for keep, index in zip(keep_degree.tolist(), chosen_indices) if keep
    ]
    if not after_degree_indices:
        empty = cast("dict[str, Any]", _empty_edges_result(vertex_positions))
        empty["diagnostics"] = diagnostics
        return empty

    keep_orphans = clean_edges_orphans_python(
        [traces[index] for index in after_degree_indices],
        image_shape,
        vertex_positions,
    )
    diagnostics["orphan_pruned_count"] = int(np.sum(~keep_orphans))
    after_orphan_indices = [
        index for keep, index in zip(keep_orphans.tolist(), after_degree_indices) if keep
    ]
    if not after_orphan_indices:
        empty = cast("dict[str, Any]", _empty_edges_result(vertex_positions))
        empty["diagnostics"] = diagnostics
        return empty

    keep_cycles = clean_edges_cycles_python(connections[after_orphan_indices])
    diagnostics["cycle_pruned_count"] = int(np.sum(~keep_cycles))
    final_indices = [
        index for keep, index in zip(keep_cycles.tolist(), after_orphan_indices) if keep
    ]

    return build_selected_edges_result(
        final_indices,
        traces,
        connections,
        metrics,
        energy_traces,
        scale_traces,
        connection_sources,
        vertex_positions,
        diagnostics,
    )

