"""Preferred internal name for edge cleanup helpers."""

from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from slavv_python.core.edges.payloads import _clip_trace_indices


def clean_edges_vertex_degree_excess_python(
    connections: np.ndarray,
    metrics: np.ndarray,
    max_edges_per_vertex: int,
) -> np.ndarray:
    """Mirror MATLAB's excess-degree cleanup on best-to-worst sorted edges."""
    del metrics
    if connections.size == 0 or max_edges_per_vertex <= 0:
        return np.ones((len(connections),), dtype=bool)

    normalized = np.asarray(connections, dtype=np.int32).reshape(-1, 2)
    keep: np.ndarray = np.ones((len(normalized),), dtype=bool)
    n_vertices = int(np.max(normalized)) + 1 if normalized.size else 0
    if n_vertices <= 0 or len(normalized) == 0:
        return keep

    rows = normalized[:, 0].astype(np.int32, copy=False)
    cols = normalized[:, 1].astype(np.int32, copy=False)
    edge_ids = np.arange(1, len(normalized) + 1, dtype=np.int32)
    edge_lookup_table = sparse.csr_matrix(
        (edge_ids, (rows, cols)),
        shape=(n_vertices, n_vertices),
        dtype=np.int32,
    )
    adjacency_matrix = sparse.csr_matrix(
        (np.ones((len(normalized),), dtype=bool), (rows, cols)),
        shape=(n_vertices, n_vertices),
        dtype=bool,
    )
    vertex_degrees = (
        np.asarray(adjacency_matrix.sum(axis=0)).ravel()
        + np.asarray(adjacency_matrix.sum(axis=1)).ravel()
    )
    vertex_excess_degrees = vertex_degrees - int(max_edges_per_vertex)
    vertices_of_excess_degree = np.flatnonzero(vertex_excess_degrees > 0)
    if vertices_of_excess_degree.size == 0:
        return keep

    edges_to_remove: list[int] = []
    for vertex_index in vertices_of_excess_degree.tolist():
        incoming_vertices = edge_lookup_table[:, vertex_index].nonzero()[0].astype(np.int32)
        outgoing_vertices = edge_lookup_table[vertex_index, :].nonzero()[1].astype(np.int32)
        edges_at_vertex = np.concatenate(
            (
                edge_lookup_table[incoming_vertices, vertex_index].toarray().ravel(),
                edge_lookup_table[vertex_index, outgoing_vertices].toarray().ravel(),
            )
        )
        if edges_at_vertex.size == 0:
            continue
        edges_at_vertex_descending = np.sort(edges_at_vertex)[::-1]
        excess_degree = int(vertex_excess_degrees[vertex_index])
        edges_to_remove.extend(edges_at_vertex_descending[:excess_degree].astype(int).tolist())

    if edges_to_remove:
        keep[np.asarray(edges_to_remove, dtype=np.int32) - 1] = False
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

    edge_locations_by_original_index: list[np.ndarray] = []
    for trace in traces:
        coords = _clip_trace_indices(np.asarray(trace, dtype=np.float32), image_shape)
        edge_locations_by_original_index.append(
            np.asarray(
                coords[:, 0]
                + coords[:, 1] * image_shape[0]
                + coords[:, 2] * image_shape[0] * image_shape[1],
                dtype=np.int64,
            )
        )

    keep: np.ndarray = np.ones((len(edge_locations_by_original_index),), dtype=bool)
    original_edge_indices = list(range(len(edge_locations_by_original_index)))
    active_edge_locations = list(edge_locations_by_original_index)
    searching_for_orphans = True
    while searching_for_orphans:
        number_of_edges = len(active_edge_locations)
        if number_of_edges == 0:
            break

        edge_index_lut = [
            np.full(edge_locations.shape, edge_index + 1, dtype=np.int32)
            for edge_index, edge_locations in enumerate(active_edge_locations)
        ]
        interior_edge_locations = (
            np.concatenate(
                [
                    edge_locations[1:-1]
                    for edge_locations in active_edge_locations
                    if edge_locations.size > 2
                ],
                axis=0,
            )
            if any(edge_locations.size > 2 for edge_locations in active_edge_locations)
            else np.zeros((0,), dtype=np.int64)
        )
        exterior_edge_locations = np.concatenate(
            [edge_locations[[0, -1]] for edge_locations in active_edge_locations],
            axis=0,
        )
        exterior_edge_location_index2edge_index = np.concatenate(
            [edge_indices[[0, -1]] for edge_indices in edge_index_lut],
            axis=0,
        )
        union_locations = np.union1d(
            interior_edge_locations, np.fromiter(vertex_locations, dtype=np.int64)
        )
        unique_exterior_locations, unique_exterior_indices = np.unique(
            exterior_edge_locations,
            return_index=True,
        )
        orphan_value_mask = ~np.isin(unique_exterior_locations, union_locations)
        orphan_terminal_indices = unique_exterior_indices[orphan_value_mask]
        edge_indices_to_remove = np.unique(
            exterior_edge_location_index2edge_index[orphan_terminal_indices]
        ).astype(np.int32, copy=False)

        if edge_indices_to_remove.size == 0:
            searching_for_orphans = False
            continue

        for current_edge_index in sorted(edge_indices_to_remove.tolist(), reverse=True):
            keep[original_edge_indices[current_edge_index - 1]] = False
            del original_edge_indices[current_edge_index - 1]
            del active_edge_locations[current_edge_index - 1]
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

        vertex_pairs_to_remove: list[tuple[int, int]] = []
        cycle_vertex_mask = np.zeros((adjacency.shape[0],), dtype=bool)

        for component_vertices in components:
            component_lookup = edge_lookup[component_vertices][:, component_vertices].tocoo()
            if component_lookup.nnz == 0:
                continue

            worst_edge_id = int(np.max(component_lookup.data))
            removed_edge_ids.append(worst_edge_id)
            first_match = int(np.flatnonzero(component_lookup.data == worst_edge_id)[0])
            row = int(component_vertices[component_lookup.row[first_match]])
            col = int(component_vertices[component_lookup.col[first_match]])
            vertex_pairs_to_remove.append((row, col))
            cycle_vertex_mask[component_vertices] = True

        if not vertex_pairs_to_remove:
            break

        adjacency = adjacency.tolil(copy=True)
        edge_lookup = edge_lookup.tolil(copy=True)
        for row, col in vertex_pairs_to_remove:
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


__all__ = [
    "clean_edges_cycles_python",
    "clean_edges_orphans_python",
    "clean_edges_vertex_degree_excess_python",
]
