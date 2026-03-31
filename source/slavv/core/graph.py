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
    mismatched: list[list[int]] = []

    for strand in strands:
        if len(strand) >= 2:
            start, end = strand[0], strand[-1]
            start_degree = len(adjacency_list[start])
            end_degree = len(adjacency_list[end])
            if start_degree > end_degree or (start_degree == end_degree and start > end):
                strand = strand[::-1]

        sorted_strands.append(strand)

    return sorted_strands, mismatched


def _normalize_connections(edge_connections: Any) -> np.ndarray:
    """Normalize edge connections to an ``(N, 2)`` int32 array."""
    connections = np.asarray(edge_connections, dtype=np.int32)
    if connections.size == 0:
        return np.empty((0, 2), dtype=np.int32)
    return connections.reshape(-1, 2)  # type: ignore[no-any-return]


def _build_graph_state(
    edge_traces: list[np.ndarray],
    edge_connections: np.ndarray,
    n_vertices: int,
) -> tuple[dict[int, set[int]], dict[tuple[int, int], np.ndarray], list[dict[str, Any]]]:
    """Build sparse adjacency, undirected graph-edge storage, and dangling edge records."""
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

        v0, v1 = int(start_vertex), int(end_vertex)
        adjacency_list[v0].add(v1)
        adjacency_list[v1].add(v0)
        key = (v0, v1) if v0 < v1 else (v1, v0)
        graph_edges.setdefault(key, trace)

    return adjacency_list, graph_edges, dangling_edges


def _remove_short_hairs(
    graph_edges: dict[tuple[int, int], np.ndarray],
    adjacency_list: dict[int, set[int]],
    microns_per_voxel: np.ndarray,
    min_hair_length: float,
) -> None:
    """Remove short terminal hairs in-place."""
    if min_hair_length <= 0:
        return

    to_remove: list[tuple[int, int]] = []
    for (v0, v1), trace in graph_edges.items():
        length: float = np.sum(np.linalg.norm(np.diff(trace, axis=0) * microns_per_voxel, axis=1))
        if length < min_hair_length and (
            len(adjacency_list[v0]) == 1 or len(adjacency_list[v1]) == 1
        ):
            adjacency_list[v0].discard(v1)
            adjacency_list[v1].discard(v0)
            to_remove.append((v0, v1))

    for key in to_remove:
        del graph_edges[key]


def _remove_cycles(
    graph_edges: dict[tuple[int, int], np.ndarray],
    adjacency_list: dict[int, set[int]],
    n_vertices: int,
) -> list[tuple[int, int]]:
    """Remove cycle-closing edges by building a spanning forest in best-to-worst order."""
    cycles: list[tuple[int, int]] = []
    if not graph_edges:
        return cycles

    parent: np.ndarray = np.arange(n_vertices, dtype=np.int32)

    def find(vertex: int) -> int:
        while parent[vertex] != vertex:
            parent[vertex] = parent[parent[vertex]]
            vertex = int(parent[vertex])
        return int(vertex)

    for v0, v1 in list(graph_edges.keys()):
        root0 = find(v0)
        root1 = find(v1)
        if root0 == root1:
            cycles.append((v0, v1))
            adjacency_list[v0].discard(v1)
            adjacency_list[v1].discard(v0)
            del graph_edges[(v0, v1)]
        else:
            parent[root1] = root0

    return cycles


def _vertex_degrees(adjacency_list: dict[int, set[int]], n_vertices: int) -> np.ndarray:
    """Return per-vertex degree counts."""
    return np.array([len(adjacency_list[i]) for i in range(n_vertices)], dtype=np.int32)  # type: ignore[no-any-return]


def _default_network_topology(
    adjacency_list: dict[int, set[int]],
    n_vertices: int,
) -> dict[str, Any]:
    """Construct the default sparse-walk network topology."""
    strands = []
    visited: np.ndarray = np.zeros(n_vertices, dtype=bool)

    for vertex_idx in range(n_vertices):
        if not visited[vertex_idx] and len(adjacency_list[vertex_idx]) > 0:
            strand = trace_strand_sparse(vertex_idx, adjacency_list, visited)
            if len(strand) > 1:
                strands.append(strand)

    strands, mismatched = sort_and_validate_strands_sparse(strands, adjacency_list)
    return {
        "strands": strands,
        "mismatched_strands": mismatched,
    }


def _interior_components(
    interior_vertices: set[int],
    adjacency_list: dict[int, set[int]],
) -> list[list[int]]:
    """Return connected components of the degree-2 interior-vertex subgraph."""
    components: list[list[int]] = []
    visited: set[int] = set()

    for start in sorted(interior_vertices):
        if start in visited:
            continue
        stack = [start]
        component: list[int] = []
        while stack:
            vertex = stack.pop()
            if vertex in visited:
                continue
            visited.add(vertex)
            component.append(vertex)
            for neighbor in sorted(adjacency_list[vertex], reverse=True):
                if neighbor in interior_vertices and neighbor not in visited:
                    stack.append(neighbor)
        components.append(sorted(component))

    return components


def _component_edge_indices(
    connections: np.ndarray,
    component_set: set[int],
    end_vertices: set[int],
) -> list[int]:
    """Return edge indices that belong to a MATLAB-style strand component."""
    indices: list[int] = []
    for index, (start_vertex, end_vertex) in enumerate(connections):
        start = int(start_vertex)
        end = int(end_vertex)
        if (
            (start in component_set and end in component_set)
            or (start in component_set and end in end_vertices)
            or (end in component_set and start in end_vertices)
        ):
            indices.append(index)
    return indices


def _order_strand_vertices(
    edge_pairs: list[tuple[int, int]],
    preferred_start: int | None,
) -> list[int]:
    """Order strand vertices end-to-end from an undirected edge set."""
    if not edge_pairs:
        return []

    local_adjacency: dict[int, set[int]] = {}
    for start_vertex, end_vertex in edge_pairs:
        local_adjacency.setdefault(int(start_vertex), set()).add(int(end_vertex))
        local_adjacency.setdefault(int(end_vertex), set()).add(int(start_vertex))

    start_vertex = preferred_start if preferred_start in local_adjacency else min(local_adjacency)
    path = [int(start_vertex)]
    current = int(start_vertex)
    used_edges: set[tuple[int, int]] = set()

    while len(used_edges) < len(edge_pairs):
        next_vertex = None
        for neighbor in sorted(local_adjacency[current]):
            v0, v1 = current, int(neighbor)
            edge_key = (v0, v1) if v0 < v1 else (v1, v0)
            if edge_key not in used_edges:
                next_vertex = int(neighbor)
                used_edges.add(edge_key)
                break
        if next_vertex is None:
            break
        path.append(next_vertex)
        current = next_vertex

    return path


def _matlab_parity_network_topology(
    edge_connections: np.ndarray,
    adjacency_list: dict[int, set[int]],
    n_vertices: int,
) -> dict[str, Any]:
    """Construct strand topology following MATLAB's degree-2-interior model."""
    if edge_connections.size == 0:
        return {
            "strands": [],
            "strands_to_vertices": [],
            "mismatched_strands": [],
        }

    vertex_degrees = _vertex_degrees(adjacency_list, n_vertices)
    interior_vertices = {
        int(vertex_index)
        for vertex_index, degree in enumerate(vertex_degrees.tolist())
        if degree == 2
    }

    strands: list[list[int]] = []
    strands_to_vertices: list[list[int]] = []

    for component in _interior_components(interior_vertices, adjacency_list):
        active_component = component
        edge_indices: list[int] = []
        end_vertices: list[int] = []

        while active_component:
            component_set = set(active_component)
            end_vertices = sorted(
                {
                    int(neighbor)
                    for vertex in active_component
                    for neighbor in adjacency_list[vertex]
                    if neighbor not in component_set
                }
            )
            edge_indices = _component_edge_indices(
                edge_connections, component_set, set(end_vertices)
            )
            if end_vertices or not edge_indices:
                break

            worst_edge_index = max(edge_indices)
            worst_start, worst_end = (int(value) for value in edge_connections[worst_edge_index])
            active_component = [
                vertex for vertex in active_component if vertex not in {worst_start, worst_end}
            ]

        if not active_component or not edge_indices:
            continue

        ordered_vertices = _order_strand_vertices(
            [
                (int(edge_connections[index, 0]), int(edge_connections[index, 1]))
                for index in edge_indices
            ],
            end_vertices[0] if end_vertices else None,
        )
        if len(ordered_vertices) > 1:
            strands.append(ordered_vertices)
            strands_to_vertices.append([ordered_vertices[0], ordered_vertices[-1]])

    for start_vertex, end_vertex in edge_connections.tolist():
        if int(start_vertex) not in interior_vertices and int(end_vertex) not in interior_vertices:
            strand = [int(start_vertex), int(end_vertex)]
            strands.append(strand)
            strands_to_vertices.append(strand.copy())

    return {
        "strands": strands,
        "strands_to_vertices": strands_to_vertices,
        "mismatched_strands": [],
    }


def _network_payload(
    adjacency_list: dict[int, set[int]],
    graph_edges: dict[tuple[int, int], np.ndarray],
    dangling_edges: list[dict[str, Any]],
    cycles: list[tuple[int, int]],
    edge_connections: np.ndarray,
    n_vertices: int,
    comparison_exact_network: bool,
) -> dict[str, Any]:
    """Build the final network payload from shared graph state."""
    if comparison_exact_network:
        topology = _matlab_parity_network_topology(
            edge_connections[(edge_connections[:, 0] >= 0) & (edge_connections[:, 1] >= 0)],
            adjacency_list,
            n_vertices,
        )
    else:
        topology = _default_network_topology(adjacency_list, n_vertices)

    vertex_degrees = _vertex_degrees(adjacency_list, n_vertices)
    bifurcations = np.where(vertex_degrees > 2)[0].astype(np.int32)
    orphans = np.where(vertex_degrees == 0)[0].astype(np.int32)

    payload = {
        "strands": topology["strands"],
        "bifurcations": bifurcations,
        "orphans": orphans,
        "cycles": cycles,
        "mismatched_strands": topology["mismatched_strands"],
        "adjacency_list": adjacency_list,
        "vertex_degrees": vertex_degrees,
        "graph_edges": graph_edges,
        "dangling_edges": dangling_edges,
    }
    if "strands_to_vertices" in topology:
        payload["strands_to_vertices"] = topology["strands_to_vertices"]
    return payload


def construct_network(
    edges: dict[str, Any], vertices: dict[str, Any], params: dict[str, Any]
) -> dict[str, Any]:
    """Construct network from traced edges and detected vertices."""
    logger.info("Constructing network")

    edge_traces = edges["traces"]
    edge_connections = _normalize_connections(edges["connections"])
    vertex_positions = vertices["positions"]
    n_vertices = len(vertex_positions)

    microns_per_voxel = np.array(params.get("microns_per_voxel", [1.0, 1.0, 1.0]), dtype=float)
    comparison_exact_network = bool(params.get("comparison_exact_network", False))
    min_hair_length = (
        0.0 if comparison_exact_network else params.get("min_hair_length_in_microns", 0.0)
    )
    remove_cycles = False if comparison_exact_network else params.get("remove_cycles", True)

    adjacency_list, graph_edges, dangling_edges = _build_graph_state(
        edge_traces,
        edge_connections,
        n_vertices,
    )

    _remove_short_hairs(graph_edges, adjacency_list, microns_per_voxel, float(min_hair_length))
    cycles = _remove_cycles(graph_edges, adjacency_list, n_vertices) if remove_cycles else []

    network = _network_payload(
        adjacency_list,
        graph_edges,
        dangling_edges,
        cycles,
        edge_connections,
        n_vertices,
        comparison_exact_network,
    )

    logger.info(
        "Constructed network with %d strands, %d bifurcations, %d orphans, removed %d cycles, and %d mismatched strands",
        len(network["strands"]),
        len(network["bifurcations"]),
        len(network["orphans"]),
        len(network["cycles"]),
        len(network["mismatched_strands"]),
    )
    return network


def construct_network_resumable(
    edges: dict[str, Any],
    vertices: dict[str, Any],
    params: dict[str, Any],
    stage_controller: StageController,
) -> dict[str, Any]:
    """Construct a network while persisting stage-level substeps."""
    from slavv.runtime.run_state import atomic_joblib_dump

    edge_traces = edges["traces"]
    edge_connections = _normalize_connections(edges["connections"])
    vertex_positions = vertices["positions"]
    n_vertices = len(vertex_positions)

    microns_per_voxel = np.array(params.get("microns_per_voxel", [1.0, 1.0, 1.0]), dtype=float)
    comparison_exact_network = bool(params.get("comparison_exact_network", False))
    min_hair_length = (
        0.0 if comparison_exact_network else params.get("min_hair_length_in_microns", 0.0)
    )
    remove_cycles = False if comparison_exact_network else params.get("remove_cycles", True)

    stage_controller.begin(detail="Building network graph", units_total=5, substage="adjacency")
    adjacency_path = stage_controller.artifact_path("adjacency.pkl")
    pruned_path = stage_controller.artifact_path("hair_pruned.pkl")
    cycle_path = stage_controller.artifact_path("cycle_pruned.pkl")
    strands_path = stage_controller.artifact_path("strands.pkl")

    if not adjacency_path.exists():
        adjacency_list, graph_edges, dangling_edges = _build_graph_state(
            edge_traces,
            edge_connections,
            n_vertices,
        )
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
        _remove_short_hairs(graph_edges, adjacency_list, microns_per_voxel, float(min_hair_length))
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
        cycles = _remove_cycles(graph_edges, adjacency_list, n_vertices)
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
        topology = (
            _matlab_parity_network_topology(
                edge_connections[(edge_connections[:, 0] >= 0) & (edge_connections[:, 1] >= 0)],
                adjacency_list,
                n_vertices,
            )
            if comparison_exact_network
            else _default_network_topology(adjacency_list, n_vertices)
        )
        atomic_joblib_dump(topology, strands_path)
    topology = joblib.load(strands_path)
    stage_controller.update(units_total=5, units_completed=4, substage="strand_trace")

    network = _network_payload(
        adjacency_list,
        graph_edges,
        dangling_edges,
        cycles,
        edge_connections,
        n_vertices,
        comparison_exact_network,
    )
    network["strands"] = topology["strands"]
    network["mismatched_strands"] = topology["mismatched_strands"]
    if "strands_to_vertices" in topology:
        network["strands_to_vertices"] = topology["strands_to_vertices"]
    stage_controller.update(units_total=5, units_completed=5, substage="finalize")
    return network


__all__ = [
    "construct_network",
    "construct_network_resumable",
    "sort_and_validate_strands_sparse",
    "trace_strand_sparse",
]
