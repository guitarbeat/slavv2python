"""
Core graph data structures and basic normalization helpers for SLAVV.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from scipy import sparse


def _normalize_connections(edge_connections: Any) -> np.ndarray:
    """Normalize edge connections to an ``(N, 2)`` int32 array."""
    connections = np.asarray(edge_connections, dtype=np.int32)
    if connections.size == 0:
        empty_connections: np.ndarray = np.empty((0, 2), dtype=np.int32)
        return empty_connections
    normalized = connections.reshape(-1, 2)
    return cast("np.ndarray", normalized)


def _build_graph_state(
    edge_traces: list[np.ndarray],
    edge_scale_traces: list[np.ndarray],
    edge_energy_traces: list[np.ndarray],
    edge_connections: np.ndarray,
    n_vertices: int,
) -> tuple[
    dict[int, set[int]],
    dict[tuple[int, int], np.ndarray],
    dict[tuple[int, int], np.ndarray],
    dict[tuple[int, int], np.ndarray],
    list[dict[str, Any]],
]:
    """Build sparse adjacency, undirected graph-edge storage, and dangling edge records."""
    adjacency_list: dict[int, set[int]] = {i: set() for i in range(n_vertices)}
    graph_edges: dict[tuple[int, int], np.ndarray] = {}
    graph_edge_scales: dict[tuple[int, int], np.ndarray] = {}
    graph_edge_energies: dict[tuple[int, int], np.ndarray] = {}
    dangling_edges: list[dict[str, Any]] = []

    for trace, scale_trace, energy_trace, (start_vertex, end_vertex) in zip(
        edge_traces,
        edge_scale_traces,
        edge_energy_traces,
        edge_connections,
    ):
        if start_vertex < 0 or end_vertex < 0:
            dangling_edges.append(
                {
                    "start": int(start_vertex) if start_vertex >= 0 else None,
                    "end": int(end_vertex) if end_vertex >= 0 else None,
                    "trace": trace,
                    "scale_trace": scale_trace,
                    "energy_trace": energy_trace,
                }
            )
            continue

        v0, v1 = int(start_vertex), int(end_vertex)
        adjacency_list[v0].add(v1)
        adjacency_list[v1].add(v0)
        key = (v0, v1) if v0 < v1 else (v1, v0)
        graph_edges.setdefault(key, trace)
        graph_edge_scales.setdefault(key, scale_trace)
        graph_edge_energies.setdefault(key, energy_trace)

    return adjacency_list, graph_edges, graph_edge_scales, graph_edge_energies, dangling_edges


def _vertex_degrees(adjacency_list: dict[int, set[int]], n_vertices: int) -> np.ndarray:
    """Return per-vertex degree counts."""
    degrees = np.array([len(adjacency_list[i]) for i in range(n_vertices)], dtype=np.int32)
    return cast("np.ndarray", degrees)


def _matlab_find_nonzero_matrix_entries(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return nonzero matrix coordinates using MATLAB's column-major ``find`` order."""
    matrix = np.asarray(mask, dtype=bool)
    n_rows, _n_cols = matrix.shape
    flat_indices = np.flatnonzero(matrix.reshape(-1, order="F"))
    rows = (flat_indices % n_rows).astype(np.int32, copy=False)
    cols = (flat_indices // n_rows).astype(np.int32, copy=False)
    return rows, cols


def _matlab_lookup_edge_ids(
    edge_lookup_table: sparse.csr_matrix,
    row_vertices: np.ndarray,
    col_vertices: np.ndarray,
) -> np.ndarray:
    """Look up directed edge ids while preserving input pair order."""
    if row_vertices.size == 0:
        return np.zeros((0,), dtype=np.int32)
    edge_ids = np.array(
        [
            int(edge_lookup_table[int(row_vertex), int(col_vertex)])
            for row_vertex, col_vertex in zip(
                row_vertices.tolist(),
                col_vertices.tolist(),
            )
        ],
        dtype=np.int32,
    )
    return cast("np.ndarray", edge_ids[edge_ids > 0] - 1)


def _matlab_network_lookup_tables(
    edge_connections: np.ndarray,
    n_vertices: int,
) -> tuple[sparse.csr_matrix, sparse.csr_matrix]:
    """Build MATLAB-shaped directed edge lookup and symmetric adjacency matrices."""
    if edge_connections.size == 0 or n_vertices <= 0:
        empty_lookup = sparse.csr_matrix((n_vertices, n_vertices), dtype=np.int32)
        empty_adjacency = sparse.csr_matrix((n_vertices, n_vertices), dtype=bool)
        return empty_lookup, empty_adjacency

    normalized = np.asarray(edge_connections, dtype=np.int32).reshape(-1, 2)
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
    adjacency_matrix = adjacency_matrix.maximum(adjacency_matrix.transpose()).astype(bool)
    return edge_lookup_table, adjacency_matrix


def _graph_state_ordered_edges(
    graph_edges: dict[tuple[int, int], np.ndarray],
    graph_edge_scales: dict[tuple[int, int], np.ndarray],
    graph_edge_energies: dict[tuple[int, int], np.ndarray],
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Recover surviving edge connections, traces, and energy traces in insertion order."""
    ordered_pairs = list(graph_edges.keys())
    if not ordered_pairs:
        return np.empty((0, 2), dtype=np.int32), [], [], []
    connections = np.asarray(ordered_pairs, dtype=np.int32).reshape(-1, 2)
    traces = [np.asarray(graph_edges[pair], dtype=np.float32) for pair in ordered_pairs]
    scale_traces = [
        np.asarray(graph_edge_scales[pair], dtype=np.float32).reshape(-1) for pair in ordered_pairs
    ]
    energy_traces = [
        np.asarray(graph_edge_energies[pair], dtype=np.float32) for pair in ordered_pairs
    ]
    return cast("np.ndarray", connections), traces, scale_traces, energy_traces
