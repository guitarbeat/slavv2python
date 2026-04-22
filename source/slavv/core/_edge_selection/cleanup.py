"""Cleanup helpers for edge selection."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from ..edge_primitives import _clip_trace_indices


def clean_edges_vertex_degree_excess_python(
    connections: np.ndarray,
    metrics: np.ndarray,
    max_edges_per_vertex: int,
) -> np.ndarray:
    """Mirror MATLAB's excess-degree cleanup on best-to-worst sorted edges."""
    if connections.size == 0 or max_edges_per_vertex <= 0:
        return np.ones((len(connections),), dtype=bool)

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


def clean_edges_cycles_python(connections: np.ndarray) -> np.ndarray:
    """Mirror MATLAB's cycle cleanup by removing the worst edge per cycle component."""
    if connections.size == 0:
        return np.zeros((0,), dtype=bool)

    normalized = np.asarray(connections, dtype=np.int32).reshape(-1, 2)
    keep: np.ndarray = np.ones((len(normalized),), dtype=bool)
    n_vertices = int(np.max(normalized)) + 1
    if n_vertices <= 0:
        return keep

    rows = normalized[:, 0].astype(np.int32, copy=False)
    cols = normalized[:, 1].astype(np.int32, copy=False)
    edge_ids = np.arange(1, len(normalized) + 1, dtype=np.int32)

    edge_lookup = sparse.csr_matrix(
        (edge_ids, (rows, cols)),
        shape=(n_vertices, n_vertices),
        dtype=np.int32,
    )
    adjacency = sparse.csr_matrix(
        (np.ones((len(normalized),), dtype=bool), (rows, cols)),
        shape=(n_vertices, n_vertices),
        dtype=bool,
    )
    adjacency = adjacency.maximum(adjacency.transpose()).astype(bool)

    active_vertices = np.flatnonzero(np.asarray(adjacency.sum(axis=0)).ravel() > 1)
    if active_vertices.size == 0:
        return keep

    adjacency = adjacency[active_vertices][:, active_vertices].tocsr()
    edge_lookup = edge_lookup[active_vertices][:, active_vertices].tocsr()

    removed_edge_ids: list[int] = []
    while adjacency.shape[0] > 0:
        two_step = (adjacency @ adjacency).astype(bool)
        cycle_adjacency = two_step.multiply(adjacency).astype(bool)
        if cycle_adjacency.nnz == 0:
            break

        n_components, labels = connected_components(
            cycle_adjacency,
            directed=False,
            connection="weak",
            return_labels=True,
        )
        components = [
            np.flatnonzero(labels == component_index)
            for component_index in range(n_components)
            if np.count_nonzero(labels == component_index) > 1
        ]
        if not components:
            break

        pair_rows: list[int] = []
        pair_cols: list[int] = []
        cycle_vertex_mask = np.zeros((adjacency.shape[0],), dtype=bool)

        for component_vertices in components:
            cycle_vertex_mask[component_vertices] = True
            component_lookup = edge_lookup[component_vertices][:, component_vertices].tocoo()
            if component_lookup.nnz == 0:
                continue

            worst_edge_id = int(np.max(component_lookup.data))
            removed_edge_ids.append(worst_edge_id)
            first_match = int(np.flatnonzero(component_lookup.data == worst_edge_id)[0])
            pair_rows.append(int(component_vertices[component_lookup.row[first_match]]))
            pair_cols.append(int(component_vertices[component_lookup.col[first_match]]))

        if not pair_rows:
            break

        adjacency = adjacency.tolil(copy=True)
        edge_lookup = edge_lookup.tolil(copy=True)
        for row, col in zip(pair_rows, pair_cols):
            adjacency[row, col] = False
            adjacency[col, row] = False
            edge_lookup[row, col] = 0
            edge_lookup[col, row] = 0
        adjacency = adjacency.tocsr()
        edge_lookup = edge_lookup.tocsr()

        adjacency = adjacency[cycle_vertex_mask][:, cycle_vertex_mask].tocsr()
        edge_lookup = edge_lookup[cycle_vertex_mask][:, cycle_vertex_mask].tocsr()

    if removed_edge_ids:
        keep[np.asarray(sorted(set(removed_edge_ids)), dtype=np.int32) - 1] = False
    return keep
