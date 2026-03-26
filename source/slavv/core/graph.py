"""
Network construction and graph theory operations for SLAVV.
Handles the conversion of traced edges into a connected graph (strands, bifurcations).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from slavv.runtime import StageController


def trace_strand_sparse(
    start: int, adjacency_list: dict[int, set[int]], visited: np.ndarray
) -> list[int]:
    """Trace a strand through connected vertices using sparse adjacency list."""
    strand = [start]
    visited[start] = True
    current = start

    while True:
        neighbors = [n for n in adjacency_list[current] if not visited[n]]
        if not neighbors:
            break
        next_vertex = neighbors[0]
        strand.append(next_vertex)
        visited[next_vertex] = True
        current = next_vertex

    return strand


def sort_and_validate_strands_sparse(
    strands: list[list[int]], adjacency_list: dict[int, set[int]]
) -> tuple[list[list[int]], list[list[int]]]:
    """Sort strands and identify mismatched orderings using sparse adjacency."""
    sorted_strands = []
    mismatched = []

    for _i, strand in enumerate(strands):
        # Check if strand endpoints are correctly ordered
        if len(strand) >= 2:
            start, end = strand[0], strand[-1]
            # A strand is valid if start and end have appropriate connectivity
            start_degree = len(adjacency_list[start])
            end_degree = len(adjacency_list[end])

            # Sort strand so lower-degree vertex is at start (for consistency)
            if start_degree > end_degree or (start_degree == end_degree and start > end):
                strand = strand[::-1]

        sorted_strands.append(strand)

    return sorted_strands, mismatched


def construct_network(
    edges: dict[str, Any], vertices: dict[str, Any], params: dict[str, Any]
) -> dict[str, Any]:
    """Construct network from traced edges and detected vertices.

    MATLAB Equivalent: `get_network_V190.m`
    """
    logger.info("Constructing network")

    edge_traces = edges["traces"]
    edge_connections = edges["connections"]
    vertex_positions = vertices["positions"]

    # Parameter for hair removal and physical scaling
    microns_per_voxel = np.array(params.get("microns_per_voxel", [1.0, 1.0, 1.0]), dtype=float)
    comparison_exact_network = bool(params.get("comparison_exact_network", False))
    min_hair_length = (
        0.0 if comparison_exact_network else params.get("min_hair_length_in_microns", 0.0)
    )
    remove_cycles = False if comparison_exact_network else params.get("remove_cycles", True)

    # Build SPARSE adjacency list (instead of dense matrix for memory efficiency)
    n_vertices = len(vertex_positions)
    adjacency_list: dict[int, set[int]] = {i: set() for i in range(n_vertices)}

    # Early return if there are no edges
    if len(edge_traces) == 0:
        vertex_degrees = np.zeros(n_vertices, dtype=np.int32)
        orphans = np.arange(n_vertices, dtype=np.int32)
        logger.info(
            "Constructed network with 0 strands, 0 bifurcations, %d orphans, removed 0 cycles, and 0 mismatched strands",
            len(orphans),
        )
        return {
            "strands": [],
            "bifurcations": np.array([], dtype=np.int32),
            "orphans": orphans,
            "cycles": [],
            "mismatched_strands": [],
            "adjacency_list": adjacency_list,
            "vertex_degrees": vertex_degrees,
            "graph_edges": {},
            "dangling_edges": [],
        }

    # Store actual edges in a dictionary keyed by sorted vertex index pairs
    graph_edges: dict[tuple[int, int], np.ndarray] = {}
    dangling_edges: list[dict[str, Any]] = []

    for trace, (start_vertex, end_vertex) in zip(edge_traces, edge_connections):
        if start_vertex < 0 or end_vertex < 0:
            dangling_edges.append(
                {
                    "start": int(start_vertex) if start_vertex >= 0 else None,
                    "end": int(end_vertex) if end_vertex >= 0 else None,
                    "trace": trace,
                }
            )
            continue

        adjacency_list[start_vertex].add(end_vertex)
        adjacency_list[end_vertex].add(start_vertex)

        key = tuple(sorted((start_vertex, end_vertex)))
        if key not in graph_edges:
            graph_edges[key] = trace

    # Helper function to get degree
    def get_degree(v):
        return len(adjacency_list[v])

    # Optionally remove short hairs connected to degree-1 vertices
    if min_hair_length > 0:
        to_remove = []
        for (v0, v1), trace in graph_edges.items():
            length = np.sum(np.linalg.norm(np.diff(trace, axis=0) * microns_per_voxel, axis=1))
            if length < min_hair_length and (get_degree(v0) == 1 or get_degree(v1) == 1):
                adjacency_list[v0].discard(v1)
                adjacency_list[v1].discard(v0)
                to_remove.append((v0, v1))
        for key in to_remove:
            del graph_edges[key]

    # Optionally remove cycles by building a spanning forest
    cycles: list[tuple[int, int]] = []
    if remove_cycles and graph_edges:
        parent = np.arange(n_vertices)

        # --- EXPLANATION FOR JUNIOR DEVS ---
        # WHY: Vascular networks should ideally be tree-like structures (mostly)
        #      without loops, unless specific biological loops exist.
        # HOW: We use the Union-Find (Disjoint Set) algorithm to detect cycles.
        #      1. We iterate through all edges.
        #      2. For each edge (u, v), we find the "root" parent of u and v.
        #      3. If they accept the same root, adding this edge would create a cycle -> Remove it.
        #      4. If different roots, we "union" them by setting one as the parent of the other.
        # -----------------------------------

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for v0, v1 in list(graph_edges.keys()):
            r0, r1 = find(v0), find(v1)
            if r0 == r1:
                cycles.append((v0, v1))
                adjacency_list[v0].discard(v1)
                adjacency_list[v1].discard(v0)
                del graph_edges[(v0, v1)]
            else:
                parent[r1] = r0

    # Find connected components (strands) using sparse adjacency
    strands = []
    visited = np.zeros(n_vertices, dtype=bool)

    for vertex_idx in range(n_vertices):
        if not visited[vertex_idx] and get_degree(vertex_idx) > 0:
            strand = trace_strand_sparse(vertex_idx, adjacency_list, visited)
            if len(strand) > 1:
                strands.append(strand)

    # Sort strands and flag ordering mismatches
    strands, mismatched = sort_and_validate_strands_sparse(strands, adjacency_list)

    # Find bifurcation vertices (degree > 2) and orphans (degree == 0)
    vertex_degrees = np.array([get_degree(i) for i in range(n_vertices)], dtype=np.int32)
    bifurcations = np.where(vertex_degrees > 2)[0].astype(np.int32)
    orphans = np.where(vertex_degrees == 0)[0].astype(np.int32)

    logger.info(
        "Constructed network with %d strands, %d bifurcations, %d orphans, removed %d cycles, and %d mismatched strands",
        len(strands),
        len(bifurcations),
        len(orphans),
        len(cycles),
        len(mismatched),
    )

    return {
        "strands": strands,
        "bifurcations": bifurcations,
        "orphans": orphans,
        "cycles": cycles,
        "mismatched_strands": mismatched,
        "adjacency_list": adjacency_list,
        "vertex_degrees": vertex_degrees,
        "graph_edges": graph_edges,
        "dangling_edges": dangling_edges,
    }


def construct_network_resumable(
    edges: dict[str, Any],
    vertices: dict[str, Any],
    params: dict[str, Any],
    stage_controller: StageController,
) -> dict[str, Any]:
    """Construct a network while persisting stage-level substeps."""
    from slavv.runtime.run_state import atomic_joblib_dump

    edge_traces = edges["traces"]
    edge_connections = edges["connections"]
    vertex_positions = vertices["positions"]
    microns_per_voxel = np.array(params.get("microns_per_voxel", [1.0, 1.0, 1.0]), dtype=float)
    comparison_exact_network = bool(params.get("comparison_exact_network", False))
    min_hair_length = (
        0.0 if comparison_exact_network else params.get("min_hair_length_in_microns", 0.0)
    )
    remove_cycles = False if comparison_exact_network else params.get("remove_cycles", True)
    n_vertices = len(vertex_positions)

    stage_controller.begin(detail="Building network graph", units_total=5, substage="adjacency")
    adjacency_path = stage_controller.artifact_path("adjacency.pkl")
    pruned_path = stage_controller.artifact_path("hair_pruned.pkl")
    cycle_path = stage_controller.artifact_path("cycle_pruned.pkl")
    strands_path = stage_controller.artifact_path("strands.pkl")

    if adjacency_path.exists():
        adjacency_state = stage_controller.load_state().get("adjacency_loaded", True)
        adjacency_payload = None
        if adjacency_state:
            import joblib

            adjacency_payload = joblib.load(adjacency_path)
        if adjacency_payload is None:
            adjacency_path.unlink(missing_ok=True)
    if not adjacency_path.exists():
        adjacency_list: dict[int, set[int]] = {i: set() for i in range(n_vertices)}
        graph_edges: dict[tuple[int, int], np.ndarray] = {}
        dangling_edges: list[dict[str, Any]] = []
        for trace, (start_vertex, end_vertex) in zip(edge_traces, edge_connections):
            if start_vertex < 0 or end_vertex < 0:
                dangling_edges.append(
                    {
                        "start": int(start_vertex) if start_vertex >= 0 else None,
                        "end": int(end_vertex) if end_vertex >= 0 else None,
                        "trace": trace,
                    }
                )
                continue
            adjacency_list[start_vertex].add(end_vertex)
            adjacency_list[end_vertex].add(start_vertex)
            key = tuple(sorted((start_vertex, end_vertex)))
            graph_edges.setdefault(key, trace)
        atomic_joblib_dump(
            {
                "adjacency_list": adjacency_list,
                "graph_edges": graph_edges,
                "dangling_edges": dangling_edges,
            },
            adjacency_path,
        )
    import joblib

    adjacency_payload = joblib.load(adjacency_path)
    adjacency_list = adjacency_payload["adjacency_list"]
    graph_edges = adjacency_payload["graph_edges"]
    dangling_edges = adjacency_payload["dangling_edges"]
    stage_controller.update(units_total=5, units_completed=1, substage="adjacency")

    if min_hair_length > 0 and not pruned_path.exists():
        to_remove = []
        for (v0, v1), trace in graph_edges.items():
            length = np.sum(np.linalg.norm(np.diff(trace, axis=0) * microns_per_voxel, axis=1))
            if length < min_hair_length and (
                len(adjacency_list[v0]) == 1 or len(adjacency_list[v1]) == 1
            ):
                adjacency_list[v0].discard(v1)
                adjacency_list[v1].discard(v0)
                to_remove.append((v0, v1))
        for key in to_remove:
            del graph_edges[key]
        atomic_joblib_dump(
            {"adjacency_list": adjacency_list, "graph_edges": graph_edges},
            pruned_path,
        )
    if pruned_path.exists():
        pruned_payload = joblib.load(pruned_path)
        adjacency_list = pruned_payload["adjacency_list"]
        graph_edges = pruned_payload["graph_edges"]
    stage_controller.update(units_total=5, units_completed=2, substage="hair_prune")

    cycles: list[tuple[int, int]] = []
    if remove_cycles and graph_edges and not cycle_path.exists():
        parent = np.arange(n_vertices)

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        for v0, v1 in list(graph_edges.keys()):
            r0, r1 = find(v0), find(v1)
            if r0 == r1:
                cycles.append((v0, v1))
                adjacency_list[v0].discard(v1)
                adjacency_list[v1].discard(v0)
                del graph_edges[(v0, v1)]
            else:
                parent[r1] = r0
        atomic_joblib_dump(
            {"adjacency_list": adjacency_list, "graph_edges": graph_edges, "cycles": cycles},
            cycle_path,
        )
    if cycle_path.exists():
        cycle_payload = joblib.load(cycle_path)
        adjacency_list = cycle_payload["adjacency_list"]
        graph_edges = cycle_payload["graph_edges"]
        cycles = cycle_payload["cycles"]
    stage_controller.update(units_total=5, units_completed=3, substage="cycle_prune")

    if not strands_path.exists():
        strands = []
        visited = np.zeros(n_vertices, dtype=bool)
        for vertex_idx in range(n_vertices):
            if not visited[vertex_idx] and len(adjacency_list[vertex_idx]) > 0:
                strand = trace_strand_sparse(vertex_idx, adjacency_list, visited)
                if len(strand) > 1:
                    strands.append(strand)
        strands, mismatched = sort_and_validate_strands_sparse(strands, adjacency_list)
        atomic_joblib_dump(
            {"strands": strands, "mismatched": mismatched},
            strands_path,
        )
    strands_payload = joblib.load(strands_path)
    strands = strands_payload["strands"]
    mismatched = strands_payload["mismatched"]
    stage_controller.update(units_total=5, units_completed=4, substage="strand_trace")

    vertex_degrees = np.array([len(adjacency_list[i]) for i in range(n_vertices)], dtype=np.int32)
    bifurcations = np.where(vertex_degrees > 2)[0].astype(np.int32)
    orphans = np.where(vertex_degrees == 0)[0].astype(np.int32)
    return {
        "strands": strands,
        "bifurcations": bifurcations,
        "orphans": orphans,
        "cycles": cycles,
        "mismatched_strands": mismatched,
        "adjacency_list": adjacency_list,
        "vertex_degrees": vertex_degrees,
        "graph_edges": graph_edges,
        "dangling_edges": dangling_edges,
    }


__all__ = [
    "construct_network",
    "construct_network_resumable",
    "sort_and_validate_strands_sparse",
    "trace_strand_sparse",
]
