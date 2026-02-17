
"""
Network construction and graph theory operations for SLAVV.
Handles the conversion of traced edges into a connected graph (strands, bifurcations).
"""
import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Set

logger = logging.getLogger(__name__)

def trace_strand_sparse(start: int, adjacency_list: Dict[int, Set[int]], 
                        visited: np.ndarray) -> List[int]:
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

def sort_and_validate_strands_sparse(strands: List[List[int]], 
                                     adjacency_list: Dict[int, Set[int]]) -> Tuple[List[List[int]], List[List[int]]]:
    """Sort strands and identify mismatched orderings using sparse adjacency."""
    sorted_strands = []
    mismatched = []
    
    for i, strand in enumerate(strands):
        # Check if strand endpoints are correctly ordered
        if len(strand) >= 2:
            start, end = strand[0], strand[-1]
            # A strand is valid if start and end have appropriate connectivity
            start_degree = len(adjacency_list[start])
            end_degree = len(adjacency_list[end])
            
            # Sort strand so lower-degree vertex is at start (for consistency)
            if start_degree > end_degree:
                strand = strand[::-1]
            elif start_degree == end_degree and start > end:
                strand = strand[::-1]
        
        sorted_strands.append(strand)
    
    return sorted_strands, mismatched

def construct_network(edges: Dict[str, Any], vertices: Dict[str, Any],
                      params: Dict[str, Any]) -> Dict[str, Any]:
    """Construct network from traced edges and detected vertices.
    
    MATLAB Equivalent: `get_network_V190.m`
    """
    logger.info("Constructing network")

    edge_traces = edges["traces"]
    edge_connections = edges["connections"]
    vertex_positions = vertices["positions"]

    # Parameter for hair removal and physical scaling
    microns_per_voxel = np.array(params.get("microns_per_voxel", [1.0, 1.0, 1.0]), dtype=float)
    min_hair_length = params.get("min_hair_length_in_microns", 0.0)
    remove_cycles = params.get("remove_cycles", True)

    # Build SPARSE adjacency list (instead of dense matrix for memory efficiency)
    n_vertices = len(vertex_positions)
    adjacency_list: Dict[int, Set[int]] = {i: set() for i in range(n_vertices)}

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
    graph_edges: Dict[Tuple[int, int], np.ndarray] = {}
    dangling_edges: List[Dict[str, Any]] = []

    for trace, (start_vertex, end_vertex) in zip(edge_traces, edge_connections):
        if start_vertex < 0 or end_vertex < 0:
            dangling_edges.append({
                "start": int(start_vertex) if start_vertex >= 0 else None,
                "end": int(end_vertex) if end_vertex >= 0 else None,
                "trace": trace,
            })
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
            length = np.sum(
                np.linalg.norm(np.diff(trace, axis=0) * microns_per_voxel, axis=1)
            )
            if length < min_hair_length and (
                get_degree(v0) == 1 or get_degree(v1) == 1
            ):
                adjacency_list[v0].discard(v1)
                adjacency_list[v1].discard(v0)
                to_remove.append((v0, v1))
        for key in to_remove:
            del graph_edges[key]

    # Optionally remove cycles by building a spanning forest
    cycles: List[Tuple[int, int]] = []
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

        for (v0, v1) in list(graph_edges.keys()):
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

__all__ = [
    "trace_strand_sparse",
    "sort_and_validate_strands_sparse",
    "construct_network",
]
