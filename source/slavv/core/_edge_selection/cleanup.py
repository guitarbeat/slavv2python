"""Cleanup helpers for edge selection."""

from __future__ import annotations

import numpy as np

from ..edge_primitives import _clip_trace_indices


def clean_edges_vertex_degree_excess_python(
    connections: np.ndarray,
    metrics: np.ndarray,
    max_edges_per_vertex: int,
) -> np.ndarray:
    """Mirror MATLAB's excess-degree cleanup on best-to-worst sorted edges."""
    if connections.size == 0 or max_edges_per_vertex <= 0:
        return np.ones((len(connections),), dtype=bool)

    keep = np.ones((len(connections),), dtype=bool)
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


def clean_edges_orphans_python(
    traces: list[np.ndarray],
    image_shape: tuple[int, int, int],
    vertex_positions: np.ndarray,
) -> np.ndarray:
    """Remove edges whose endpoints do not touch a vertex or any interior edge voxel."""
    if not traces:
        return np.zeros((0,), dtype=bool)

    vertex_coords = np.rint(np.asarray(vertex_positions, dtype=np.float32)).astype(
        np.int32,
        copy=False,
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

    keep = np.ones((len(locations),), dtype=bool)
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


def clean_edges_cycles_python(connections: np.ndarray) -> np.ndarray:
    """Remove cycle-closing edges while preserving the best-to-worst order."""
    if connections.size == 0:
        return np.zeros((0,), dtype=bool)

    keep = np.ones((len(connections),), dtype=bool)
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

