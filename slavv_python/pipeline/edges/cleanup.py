"""
Edge Cleanup and Quality Control Engine.

This module provides specialized filters for pruning vascular candidates,
ensuring the final graph adheres to biological topology constraints (e.g.,
maximum branching degrees, orphan removal, and cycle breaking).
"""

from __future__ import annotations

from typing import cast

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from slavv_python.pipeline.edges.payloads import _clip_trace_indices
from slavv_python.utils.matlab_order import yxz_to_matlab_linear_indices

# --- TOPOLOGY FILTERS ---


def remove_excess_vertex_degrees(
    connections: np.ndarray,
    metrics: np.ndarray,
    max_degree: int,
) -> np.ndarray:
    """
    Prunes edges from vertices exceeding the maximum biological degree.

    This filter mirrors MATLAB's greedy pruning: it keeps the best-ranked
    edges (by energy metric) and discards the surplus.
    """
    # 1. Validation & Setup
    if connections.size == 0 or max_degree <= 0:
        return np.ones((len(connections),), dtype=bool)

    edge_connections = np.asarray(connections, dtype=np.int32).reshape(-1, 2)
    keep_mask: np.ndarray = np.ones((len(edge_connections),), dtype=bool)

    num_vertices = int(np.max(edge_connections)) + 1 if edge_connections.size else 0
    if num_vertices <= 0:
        return keep_mask

    # 2. Build Adjacency Structures manually to avoid sparse matrix summation/collapsing
    num_edges = len(edge_connections)
    vertex_indices: np.ndarray = edge_connections.ravel()
    edge_ids: np.ndarray = np.repeat(np.arange(num_edges), 2)

    sort_idx: np.ndarray = np.argsort(vertex_indices)
    sorted_vertices: np.ndarray = vertex_indices[sort_idx]
    sorted_edge_ids: np.ndarray = edge_ids[sort_idx]
    starts = cast(np.ndarray, np.searchsorted(sorted_vertices, np.arange(num_vertices), side="left"))
    ends = cast(np.ndarray, np.searchsorted(sorted_vertices, np.arange(num_vertices), side="right"))

    degrees = ends - starts

    # 3. Identify & Prune Over-connected Vertices
    excess_vertices = np.flatnonzero(degrees > int(max_degree))
    if excess_vertices.size == 0:
        return keep_mask

    pruned_edge_indices: list[int] = []
    for v_idx in excess_vertices.tolist():
        # Collect all edges touching this vertex
        incident_edges = sorted_edge_ids[starts[v_idx]:ends[v_idx]]
        if incident_edges.size == 0:
            continue

        # Sort by quality (MATLAB logic: higher ID = later generated/worse metric)
        # and discard the surplus from the worst end.
        worst_first = np.sort(incident_edges)[::-1]
        surplus_count = int(degrees[v_idx] - max_degree)
        pruned_edge_indices.extend(worst_first[:surplus_count].tolist())

    # 4. Finalize Mask
    if pruned_edge_indices:
        keep_mask[np.asarray(pruned_edge_indices, dtype=np.int32)] = False

    return keep_mask


def prune_orphan_edges(
    traces: list[np.ndarray],
    volume_shape: tuple[int, int, int],
    vertex_positions: np.ndarray,
) -> np.ndarray:
    """
    Removes floating edges that do not connect to a vertex or valid neighbor.

    This filter prevents disconnected 'hairs' or noise traces from entering
    the final network.
    """
    if not traces:
        return np.zeros((0,), dtype=bool)

    # 1. Map Vertices to Linear Index Space
    vertex_locations = _get_linear_voxel_set(vertex_positions, volume_shape)

    # 2. Map Edges to Linear Index Space
    edge_voxels = [_get_linear_trace_voxels(trace, volume_shape) for trace in traces]

    keep_mask: np.ndarray = np.ones((len(edge_voxels),), dtype=bool)
    original_indices = list(range(len(edge_voxels)))
    active_pool = list(edge_voxels)

    # 3. Recursive Pruning
    # Removing one orphan may create a new orphan; loop until convergence.
    while True:
        if not active_pool:
            break

        orphans = _find_orphan_indices(active_pool, vertex_locations)
        if not orphans:
            break

        # Remove found orphans from the active pool and the global keep mask
        for pool_idx in sorted(orphans, reverse=True):
            keep_mask[original_indices[pool_idx]] = False
            del original_indices[pool_idx]
            del active_pool[pool_idx]

    return keep_mask


def break_graph_cycles(connections: np.ndarray) -> np.ndarray:
    """
    Breaks cyclic loops by greedily removing the worst-ranked edge in each component.

    Biologically, small cycles are often artifacts of the discovery logic rather
    than real vascular topology.
    """
    if connections.size == 0:
        return np.zeros((0,), dtype=bool)

    # 1. Build Interaction Graph
    edges = np.asarray(connections, dtype=np.int32).reshape(-1, 2)
    keep_mask: np.ndarray = np.ones((len(edges),), dtype=bool)

    num_vertices = int(np.max(edges)) + 1
    rows, cols = edges[:, 0], edges[:, 1]
    edge_ids: np.ndarray = np.arange(1, len(edges) + 1, dtype=np.int32)

    # Undirected adjacency for cycle detection
    edge_lookup = sparse.csr_matrix((edge_ids, (rows, cols)), shape=(num_vertices, num_vertices))
    adj = sparse.csr_matrix(
        (np.ones(len(edges), dtype=bool), (rows, cols)), shape=(num_vertices, num_vertices)
    )
    undirected_adj = adj.maximum(adj.transpose()).astype(bool)

    # Only process vertices that have more than one connection (potential for cycles)
    active_nodes = np.flatnonzero(np.asarray(undirected_adj.sum(axis=0)).ravel() > 1)
    if active_nodes.size == 0:
        return keep_mask

    # Sub-graph of candidates
    sub_adj = undirected_adj[active_nodes][:, active_nodes].tocsr()
    sub_lookup = edge_lookup[active_nodes][:, active_nodes].tocsr()

    # 2. Greedily Break Cycles
    removed_ids: list[int] = []
    while sub_adj.shape[0] > 0:
        # Detect triangles and larger loops using sparse matrix multiplication
        cycle_candidates = (sub_adj @ sub_adj).astype(bool).multiply(sub_adj).astype(bool)
        if cycle_candidates.nnz == 0:
            break

        # Find connected components within the cycle sub-graph
        n_comp, labels = connected_components(cycle_candidates, directed=False)
        components = [
            np.flatnonzero(labels == i) for i in range(n_comp) if np.count_nonzero(labels == i) > 1
        ]

        if not components:
            break

        dirty_nodes = np.zeros(sub_adj.shape[0], dtype=bool)
        edges_to_kill: list[tuple[int, int]] = []

        for comp_nodes in components:
            comp_edges = sub_lookup[comp_nodes][:, comp_nodes].tocoo()
            if comp_edges.nnz == 0:
                continue

            # Identify the worst edge in this specific cycle component
            worst_id = int(np.max(comp_edges.data))
            removed_ids.append(worst_id)

            # Map back to local subgraph coordinates to update the matrix
            match_idx = int(np.flatnonzero(comp_edges.data == worst_id)[0])
            u_local, v_local = (
                comp_nodes[comp_edges.row[match_idx]],
                comp_nodes[comp_edges.col[match_idx]],
            )
            edges_to_kill.append((u_local, v_local))
            dirty_nodes[comp_nodes] = True

        # Update subgraph state
        sub_adj = sub_adj.tolil()
        sub_lookup = sub_lookup.tolil()
        for u, v in edges_to_kill:
            sub_adj[u, v] = sub_adj[v, u] = False
            sub_lookup[u, v] = sub_lookup[v, u] = 0

        sub_adj = sub_adj.tocsr()[dirty_nodes][:, dirty_nodes]
        sub_lookup = sub_lookup.tocsr()[dirty_nodes][:, dirty_nodes]

    if removed_ids:
        keep_mask[np.asarray(sorted(set(removed_ids))) - 1] = False

    return keep_mask


# --- INTERNAL HELPERS ---


# Removed unused _get_incident_edge_ids helper


def _get_linear_voxel_set(positions: np.ndarray, shape: tuple[int, int, int]) -> set[int]:
    """Converts (N, 3) coordinates to a set of 1D linear voxel indices."""
    coords = np.rint(positions).astype(np.int32)
    coords[:, 0] = np.clip(coords[:, 0], 0, shape[0] - 1)
    coords[:, 1] = np.clip(coords[:, 1], 0, shape[1] - 1)
    coords[:, 2] = np.clip(coords[:, 2], 0, shape[2] - 1)

    return set(yxz_to_matlab_linear_indices(coords, shape).tolist())


def _get_linear_trace_voxels(trace: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    """Maps a trace path to linear voxel indices."""
    c = _clip_trace_indices(np.asarray(trace, dtype=np.float32), shape)
    return cast("np.ndarray", yxz_to_matlab_linear_indices(c, shape))


def _find_orphan_indices(
    active_edge_voxels: list[np.ndarray], vertex_voxels: set[int]
) -> list[int]:
    """Identifies edges whose endpoints are not anchored to any vertex or existing interior edge."""
    # Build reference of all 'grounded' voxels (vertices + interior segments of current edges)
    grounded_voxels = np.fromiter(vertex_voxels, dtype=np.int64)
    interior_voxels = (
        np.concatenate([e[1:-1] for e in active_edge_voxels if e.size > 2], axis=0)
        if any(e.size > 2 for e in active_edge_voxels)
        else np.zeros(0, dtype=np.int64)
    )

    anchor_set = np.union1d(grounded_voxels, interior_voxels)

    orphan_edges: list[int] = []
    for i, voxels in enumerate(active_edge_voxels):
        endpoints = voxels[[0, -1]]
        # If neither end of the edge touches an anchor, it's an orphan
        if not np.any(np.isin(endpoints, anchor_set)):
            orphan_edges.append(i)

    return orphan_edges


__all__ = [
    "break_graph_cycles",
    "prune_orphan_edges",
    "remove_excess_vertex_degrees",
]
