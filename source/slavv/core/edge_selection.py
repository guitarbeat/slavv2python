"""Edge selection and cleanup helpers for SLAVV."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from typing_extensions import TypeAlias

from .edge_primitives import _clip_trace_indices

Int16Array: TypeAlias = "np.ndarray"
Int32Array: TypeAlias = "np.ndarray"
Int64Array: TypeAlias = "np.ndarray"
Float32Array: TypeAlias = "np.ndarray"
Float64Array: TypeAlias = "np.ndarray"
BoolArray: TypeAlias = "np.ndarray"


def _normalize_candidate_connection_sources(
    raw_sources: Any,
    candidate_connection_count: int,
    *,
    default_source: str = "unknown",
) -> list[str]:
    """Return a normalized per-connection source label list."""
    if candidate_connection_count <= 0:
        return []

    if isinstance(raw_sources, np.ndarray):
        source_values = np.asarray(raw_sources).reshape(-1).tolist()
    elif isinstance(raw_sources, (list, tuple)):
        source_values = list(raw_sources)
    else:
        source_values = []

    allowed_sources = {"frontier", "watershed", "geodesic", "fallback", "unknown"}
    default_label = default_source if default_source in allowed_sources else "unknown"
    normalized: list[str] = []
    for index in range(candidate_connection_count):
        if index < len(source_values):
            source_label = str(source_values[index]).strip().lower()
            normalized.append(source_label if source_label in allowed_sources else default_label)
            continue
        normalized.append(default_label)
    return normalized


def _empty_stop_reason_counts() -> dict[str, int]:
    """Return the canonical edge-trace stop-reason counter payload."""
    return {
        "bounds": 0,
        "nan": 0,
        "energy_threshold": 0,
        "energy_rise_step_halving": 0,
        "max_steps": 0,
        "direct_terminal_hit": 0,
        "frontier_exhausted_nonnegative": 0,
        "length_limit": 0,
        "terminal_frontier_hit": 0,
    }


def _empty_edge_diagnostics() -> dict[str, Any]:
    """Return the canonical edge-diagnostics payload."""
    return {
        "candidate_traced_edge_count": 0,
        "terminal_edge_count": 0,
        "self_edge_count": 0,
        "duplicate_directed_pair_count": 0,
        "antiparallel_pair_count": 0,
        "chosen_edge_count": 0,
        "dangling_edge_count": 0,
        "negative_energy_rejected_count": 0,
        "conflict_rejected_count": 0,
        "conflict_rejected_by_source": {},
        "conflict_blocking_source_counts": {},
        "conflict_source_pairs": {},
        "degree_pruned_count": 0,
        "orphan_pruned_count": 0,
        "cycle_pruned_count": 0,
        "watershed_join_supplement_count": 0,
        "watershed_endpoint_degree_rejected": 0,
        "geodesic_join_supplement_count": 0,
        "geodesic_shared_neighborhood_endpoint_relaxed": 0,
        "terminal_direct_hit_count": 0,
        "terminal_reverse_center_hit_count": 0,
        "terminal_reverse_near_hit_count": 0,
        "frontier_terminal_resolution_counts": {},
        "frontier_per_origin_terminal_hits": {},
        "frontier_per_origin_terminal_accepts": {},
        "frontier_per_origin_terminal_rejections": {},
        "stop_reason_counts": _empty_stop_reason_counts(),
    }


def _merge_edge_diagnostics(target: dict[str, Any], source: dict[str, Any]) -> None:
    """Merge additive edge diagnostics from one payload into another."""
    for key, value in source.items():
        if key == "stop_reason_counts":
            target_counts = target.setdefault("stop_reason_counts", _empty_stop_reason_counts())
            for stop_reason, count in value.items():
                target_counts[stop_reason] = int(target_counts.get(stop_reason, 0)) + int(count)
            continue

        if isinstance(value, dict):
            target_map = target.setdefault(key, {})
            if not isinstance(target_map, dict):
                target_map = {}
            for item_key, item_value in value.items():
                target_map[str(item_key)] = int(target_map.get(str(item_key), 0)) + int(item_value)
            target[key] = target_map
            continue

        if isinstance(value, (int, np.integer)):
            target[key] = int(target.get(key, 0)) + int(value)


def _empty_edges_result(vertex_positions: np.ndarray | None = None) -> dict[str, Any]:
    """Return the canonical empty edge payload."""
    positions = (
        np.asarray(vertex_positions, dtype=np.float32)
        if vertex_positions is not None
        else np.empty((0, 3), dtype=np.float32)
    )
    return {
        "traces": [],
        "connections": np.zeros((0, 2), dtype=np.int32),
        "energies": np.zeros((0,), dtype=np.float32),
        "energy_traces": [],
        "scale_traces": [],
        "connection_sources": [],
        "vertex_positions": positions,
        "diagnostics": _empty_edge_diagnostics(),
        "chosen_candidate_indices": np.zeros((0,), dtype=np.int32),
    }


def _initialize_edge_selection_diagnostics(
    candidates: dict[str, Any],
    connections: np.ndarray,
    traces: list[np.ndarray],
) -> dict[str, Any]:
    """Initialize chooser diagnostics from candidate-stage counters and counts."""
    diagnostics = _empty_edge_diagnostics()
    candidate_diagnostics = candidates.get("diagnostics", {})
    for key, value in candidate_diagnostics.items():
        diagnostics[key] = value.copy() if key == "stop_reason_counts" else value
    diagnostics["candidate_traced_edge_count"] = len(traces)
    diagnostics["terminal_edge_count"] = (
        int(np.sum(connections[:, 1] >= 0)) if len(connections) else 0
    )
    diagnostics["self_edge_count"] = (
        int(np.sum(connections[:, 0] == connections[:, 1])) if len(connections) else 0
    )
    diagnostics["dangling_edge_count"] = (
        int(np.sum(connections[:, 1] < 0)) if len(connections) else 0
    )
    return diagnostics


def _prepare_candidate_indices_for_cleanup(
    connections: np.ndarray,
    metrics: np.ndarray,
    energy_traces: list[np.ndarray],
    diagnostics: dict[str, Any],
) -> list[int]:
    """Apply MATLAB-shaped pair filtering before downstream cleanup."""
    valid = (connections[:, 0] != connections[:, 1]) & (connections[:, 1] >= 0)
    filtered_indices = np.flatnonzero(valid)

    if filtered_indices.size:
        nonnegative_max = np.array(
            [
                np.nanmax(np.asarray(energy_traces[index], dtype=np.float32)) >= 0
                for index in filtered_indices
            ],
            dtype=bool,
        )
        diagnostics["negative_energy_rejected_count"] = int(np.sum(nonnegative_max))
        filtered_indices = filtered_indices[~nonnegative_max]
    else:
        diagnostics["negative_energy_rejected_count"] = 0

    if filtered_indices.size == 0:
        return []

    edge_lengths = np.asarray(
        [len(np.asarray(energy_traces[index])) for index in filtered_indices],
        dtype=np.int32,
    )
    length_sorted = filtered_indices[np.argsort(edge_lengths, kind="stable")]
    ordered = length_sorted[np.argsort(metrics[length_sorted], kind="stable")]

    directed_seen: set[tuple[int, int]] = set()
    directed_indices: list[int] = []
    for index in ordered:
        pair_d = (int(connections[index, 0]), int(connections[index, 1]))
        if pair_d in directed_seen:
            diagnostics["duplicate_directed_pair_count"] += 1
            continue
        directed_seen.add(pair_d)
        directed_indices.append(int(index))

    undirected_seen: set[tuple[int, int]] = set()
    filtered_unique_indices: list[int] = []
    for index in directed_indices:
        start_vertex, end_vertex = int(connections[index, 0]), int(connections[index, 1])
        pair_u = (
            (start_vertex, end_vertex) if start_vertex < end_vertex else (end_vertex, start_vertex)
        )
        if pair_u in undirected_seen:
            diagnostics["antiparallel_pair_count"] += 1
            continue
        undirected_seen.add(pair_u)
        filtered_unique_indices.append(int(index))

    return filtered_unique_indices


def _build_selected_edges_result(
    final_indices: list[int],
    traces: list[np.ndarray],
    connections: np.ndarray,
    metrics: np.ndarray,
    energy_traces: list[np.ndarray],
    scale_traces: list[np.ndarray],
    connection_sources: list[str],
    vertex_positions: np.ndarray,
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    """Build the canonical chosen-edge payload from candidate indices."""
    result: dict[str, Any] = _empty_edges_result(vertex_positions)
    result["traces"] = [np.asarray(traces[index], dtype=np.float32) for index in final_indices]
    result["connections"] = np.asarray(connections[final_indices], dtype=np.int32).reshape(-1, 2)
    result["energies"] = np.asarray(metrics[final_indices], dtype=np.float32)
    result["energy_traces"] = [
        np.asarray(energy_traces[index], dtype=np.float32) for index in final_indices
    ]
    result["scale_traces"] = [
        np.asarray(scale_traces[index], dtype=np.int16) for index in final_indices
    ]
    result["connection_sources"] = [
        connection_sources[index] if index < len(connection_sources) else "unknown"
        for index in final_indices
    ]
    result["chosen_candidate_indices"] = np.asarray(final_indices, dtype=np.int32)
    diagnostics["chosen_edge_count"] = len(final_indices)
    result["diagnostics"] = diagnostics
    return result


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
        empty_offsets: np.ndarray = np.zeros((1, 3), dtype=np.int32)
        return empty_offsets
    kept_offsets: Int32Array = kept.astype(np.int32, copy=False)
    return cast("np.ndarray", kept_offsets)


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
    clipped_coords: Int32Array = coords
    return cast("np.ndarray", clipped_coords)


def _clean_edges_vertex_degree_excess_python(
    connections: np.ndarray,
    metrics: np.ndarray,
    max_edges_per_vertex: int,
) -> np.ndarray:
    """Mirror MATLAB's excess-degree cleanup on best-to-worst sorted edges."""
    if connections.size == 0 or max_edges_per_vertex <= 0:
        keep_all: np.ndarray = np.ones((len(connections),), dtype=bool)
        return keep_all

    # Edge cleanup
    keep: np.ndarray = np.ones((len(connections),), dtype=bool)
    n_vertices = int(np.max(connections)) + 1 if connections.size else 0
    if n_vertices <= 0:
        return keep

    adjacency: list[list[int]] = [[] for _ in range(n_vertices)]
    for edge_index, (start_vertex, end_vertex) in enumerate(connections):
        adjacency[int(start_vertex)].append(edge_index)
        adjacency[int(end_vertex)].append(edge_index)

    for edge_indices in adjacency:
        excess = len(edge_indices) - max_edges_per_vertex
        if excess > 0:
            for edge_index in sorted(edge_indices, reverse=True)[:excess]:
                keep[edge_index] = False
    return keep


def _clean_edges_orphans_python(
    traces: list[np.ndarray],
    image_shape: tuple[int, int, int],
    vertex_positions: np.ndarray,
) -> np.ndarray:
    """Remove edges whose endpoints do not touch a vertex or any interior edge voxel."""
    if not traces:
        empty_keep: np.ndarray = np.zeros((0,), dtype=bool)
        return empty_keep

    vertex_coords = np.rint(np.asarray(vertex_positions, dtype=np.float32)).astype(
        np.int32, copy=False
    )
    vertex_coords[:, 0] = np.clip(vertex_coords[:, 0], 0, image_shape[0] - 1)
    vertex_coords[:, 1] = np.clip(vertex_coords[:, 1], 0, image_shape[1] - 1)
    vertex_coords[:, 2] = np.clip(vertex_coords[:, 2], 0, image_shape[2] - 1)
    vertex_locations = {
        int(y + x * image_shape[0] + z * image_shape[0] * image_shape[1])
        for y, x, z in vertex_coords.tolist()
    }

    locations = []
    for trace in traces:
        coords = _clip_trace_indices(np.asarray(trace, dtype=np.float32), image_shape)
        locations.append(
            np.asarray(
                coords[:, 0]
                + coords[:, 1] * image_shape[0]
                + coords[:, 2] * image_shape[0] * image_shape[1],
                dtype=np.int64,
            )
        )

    keep: np.ndarray = np.ones((len(locations),), dtype=bool)
    changed = True
    while changed:
        changed = False
        interior: set[int] = set()
        exterior_locations: list[tuple[int, int]] = []
        for edge_index, edge_locations in enumerate(locations):
            if not keep[edge_index] or edge_locations.size == 0:
                continue
            if edge_locations.size > 2:
                interior.update(int(value) for value in edge_locations[1:-1].tolist())
            exterior_locations.extend(
                (
                    (edge_index, int(edge_locations[0])),
                    (edge_index, int(edge_locations[-1])),
                )
            )
        removable: set[int] = set()
        allowed = interior | vertex_locations
        for edge_index, location in exterior_locations:
            if location not in allowed:
                removable.add(edge_index)

        if removable:
            changed = True
            for edge_index in removable:
                keep[edge_index] = False
    return keep


def _clean_edges_cycles_python(connections: np.ndarray) -> np.ndarray:
    """Remove cycle-closing edges while preserving the best-to-worst order."""
    if connections.size == 0:
        empty_keep: np.ndarray = np.zeros((0,), dtype=bool)
        return empty_keep

    keep: np.ndarray = np.ones((len(connections),), dtype=bool)
    n_vertices = int(np.max(connections)) + 1
    parent = np.arange(n_vertices, dtype=np.int32)

    def find(vertex: int) -> int:
        while parent[vertex] != vertex:
            parent[vertex] = parent[parent[vertex]]
            vertex = parent[vertex]
        return int(vertex)

    for edge_index, (start_vertex, end_vertex) in enumerate(connections):
        root_start = find(int(start_vertex))
        root_end = find(int(end_vertex))
        if root_start == root_end:
            keep[edge_index] = False
            continue
        parent[root_end] = root_start
    return keep


def _choose_edges_matlab_style(
    candidates: dict[str, Any],
    vertex_positions: np.ndarray,
    vertex_scales: np.ndarray,
    lumen_radius_pixels_axes: np.ndarray,
    image_shape: tuple[int, int, int],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Choose final edges using MATLAB-shaped filtering and cleanup semantics.

    This function is a **downstream safety net**. It should not be used to
    compensate for upstream semantic drift in the frontier tracer or watershed
    supplement. If parity is failing here, the root cause is almost certainly
    in ``_trace_origin_edges_matlab_frontier`` or
    ``_finalize_matlab_parity_candidates``.

    MATLAB-parity-critical steps (in order):
        1. **Self-edge / dangling removal** â€” Filters edges where start == end
           or end < 0. MATLAB equivalent: implicit in the edge loop.
        2. **Non-negative energy rejection** â€” Removes edges whose energy trace
           max is >= 0. MATLAB equivalent: ``max(trace) >= 0`` guard.
        3. **Directed dedup, then undirected (antiparallel) dedup** â€” Prefer
           the best-metric copy. MATLAB equivalent: unique-pair filter.
        4. **Conflict painting** â€” Reject edges that overlap with already-chosen
           edges in voxel space. MATLAB equivalent: 3D label image overlap test.
        5. **Degree pruning** â€” Cap edges per vertex to ``number_of_edges_per_vertex``.
           MATLAB equivalent: explicit cap in ``choose_edges_V300.m``.
        6. **Orphan pruning** â€” Remove edges whose endpoints touch neither a
           vertex nor any interior edge voxel. Python-only safety net.
        7. **Cycle pruning** â€” Remove cycle-closing edges via union-find.
           MATLAB equivalent: acyclic spanning-tree construction.
    """
    traces = candidates["traces"]
    connections = np.asarray(candidates["connections"], dtype=np.int32).reshape(-1, 2)
    metrics = np.asarray(candidates["metrics"], dtype=np.float32).reshape(-1)
    energy_traces = candidates["energy_traces"]
    scale_traces = candidates["scale_traces"]
    diagnostics = _initialize_edge_selection_diagnostics(candidates, connections, traces)

    if len(traces) == 0:
        empty = _empty_edges_result(vertex_positions)
        empty["diagnostics"] = diagnostics
        return empty

    filtered_indices = _prepare_candidate_indices_for_cleanup(
        connections,
        metrics,
        energy_traces,
        diagnostics,
    )
    if not filtered_indices:
        empty = _empty_edges_result(vertex_positions)
        empty["diagnostics"] = diagnostics
        return empty

    connection_sources = _normalize_candidate_connection_sources(
        candidates.get("connection_sources"),
        len(connections),
    )

    sigma_per_influence_vertices = float(params.get("sigma_per_influence_vertices", 1.0))
    sigma_per_influence_edges = float(params.get("sigma_per_influence_edges", 0.5))
    vertex_offset_cache: dict[int, np.ndarray] = {}
    edge_offset_cache: dict[int, np.ndarray] = {}
    painted_image: np.ndarray = np.zeros(image_shape, dtype=np.int32)
    painted_source_image: np.ndarray = np.zeros(image_shape, dtype=np.uint8)
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
                        coords[:, 0], coords[:, 1], coords[:, 2]
                    ].tolist()
                    if int(value) != 0
                }
                conflict_rejected_by_source = diagnostics.setdefault(
                    "conflict_rejected_by_source", {}
                )
                conflict_rejected_by_source[current_source] = (
                    int(conflict_rejected_by_source.get(current_source, 0)) + 1
                )
                conflict_blocking_source_counts = diagnostics.setdefault(
                    "conflict_blocking_source_counts", {}
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
                (start_vertex, end_vertex), endpoint_snapshots
            ):
                painted_image[coords[:, 0], coords[:, 1], coords[:, 2]] = vertex_index + 1
                painted_source_image[coords[:, 0], coords[:, 1], coords[:, 2]] = current_source_code
            chosen_indices.append(index)
        else:
            for coords, snapshot, source_snapshot in endpoint_snapshots:
                painted_image[coords[:, 0], coords[:, 1], coords[:, 2]] = snapshot
                painted_source_image[coords[:, 0], coords[:, 1], coords[:, 2]] = source_snapshot

    if not chosen_indices:
        empty = _empty_edges_result(vertex_positions)
        empty["diagnostics"] = diagnostics
        return empty

    chosen_connections = connections[chosen_indices]
    chosen_metrics = metrics[chosen_indices]
    keep_degree: np.ndarray = _clean_edges_vertex_degree_excess_python(
        chosen_connections,
        chosen_metrics,
        int(params.get("number_of_edges_per_vertex", 4)),
    )
    diagnostics["degree_pruned_count"] = int(np.sum(~keep_degree))

    after_degree_indices = [
        index for keep, index in zip(keep_degree.tolist(), chosen_indices) if keep
    ]
    if not after_degree_indices:
        empty = _empty_edges_result(vertex_positions)
        empty["diagnostics"] = diagnostics
        return empty

    keep_orphans: np.ndarray = _clean_edges_orphans_python(
        [traces[index] for index in after_degree_indices],
        image_shape,
        vertex_positions,
    )
    diagnostics["orphan_pruned_count"] = int(np.sum(~keep_orphans))
    after_orphan_indices = [
        index for keep, index in zip(keep_orphans.tolist(), after_degree_indices) if keep
    ]
    if not after_orphan_indices:
        empty = _empty_edges_result(vertex_positions)
        empty["diagnostics"] = diagnostics
        return empty

    keep_cycles: np.ndarray = _clean_edges_cycles_python(connections[after_orphan_indices])
    diagnostics["cycle_pruned_count"] = int(np.sum(~keep_cycles))
    final_indices = [
        index for keep, index in zip(keep_cycles.tolist(), after_orphan_indices) if keep
    ]

    return _build_selected_edges_result(
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


def _choose_edges_matlab_v200_cleanup(
    candidates: dict[str, Any],
    vertex_positions: np.ndarray,
    image_shape: tuple[int, int, int],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Choose final edges using the active MATLAB V200 cleanup chain only."""
    traces = candidates["traces"]
    connections = np.asarray(candidates["connections"], dtype=np.int32).reshape(-1, 2)
    metrics = np.asarray(candidates["metrics"], dtype=np.float32).reshape(-1)
    energy_traces = candidates["energy_traces"]
    scale_traces = candidates["scale_traces"]
    diagnostics = _initialize_edge_selection_diagnostics(candidates, connections, traces)

    if len(traces) == 0:
        empty = _empty_edges_result(vertex_positions)
        empty["diagnostics"] = diagnostics
        return empty

    filtered_indices = _prepare_candidate_indices_for_cleanup(
        connections,
        metrics,
        energy_traces,
        diagnostics,
    )
    if not filtered_indices:
        empty = _empty_edges_result(vertex_positions)
        empty["diagnostics"] = diagnostics
        return empty

    connection_sources = _normalize_candidate_connection_sources(
        candidates.get("connection_sources"),
        len(connections),
    )

    filtered_connections = connections[filtered_indices]
    filtered_metrics = metrics[filtered_indices]
    keep_degree = _clean_edges_vertex_degree_excess_python(
        filtered_connections,
        filtered_metrics,
        int(params.get("number_of_edges_per_vertex", 4)),
    )
    diagnostics["degree_pruned_count"] = int(np.sum(~keep_degree))
    after_degree_indices = [
        index for keep, index in zip(keep_degree.tolist(), filtered_indices) if keep
    ]
    if not after_degree_indices:
        empty = _empty_edges_result(vertex_positions)
        empty["diagnostics"] = diagnostics
        return empty

    keep_orphans = _clean_edges_orphans_python(
        [traces[index] for index in after_degree_indices],
        image_shape,
        vertex_positions,
    )
    diagnostics["orphan_pruned_count"] = int(np.sum(~keep_orphans))
    after_orphan_indices = [
        index for keep, index in zip(keep_orphans.tolist(), after_degree_indices) if keep
    ]
    if not after_orphan_indices:
        empty = _empty_edges_result(vertex_positions)
        empty["diagnostics"] = diagnostics
        return empty

    keep_cycles = _clean_edges_cycles_python(connections[after_orphan_indices])
    diagnostics["cycle_pruned_count"] = int(np.sum(~keep_cycles))
    final_indices = [
        index for keep, index in zip(keep_cycles.tolist(), after_orphan_indices) if keep
    ]
    return _build_selected_edges_result(
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


def choose_edges_for_workflow(
    candidates: dict[str, Any],
    vertex_positions: np.ndarray,
    vertex_scales: np.ndarray,
    lumen_radius_pixels_axes: np.ndarray,
    image_shape: tuple[int, int, int],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Route edge cleanup through the maintained workflow-specific chooser.

    Imported-MATLAB parity runs must use the active MATLAB V200 cleanup chain
    only. Non-parity runs keep the broader MATLAB-shaped chooser that includes
    conflict painting and the richer overlap diagnostics needed outside the
    exact-network parity workflow.
    """
    if bool(params.get("comparison_exact_network", False)):
        return _choose_edges_matlab_v200_cleanup(
            candidates,
            vertex_positions.astype(np.float32, copy=False),
            image_shape,
            params,
        )
    return _choose_edges_matlab_style(
        candidates,
        vertex_positions.astype(np.float32, copy=False),
        vertex_scales,
        lumen_radius_pixels_axes,
        image_shape,
        params,
    )
