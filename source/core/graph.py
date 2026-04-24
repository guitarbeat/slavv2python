"""
Network construction and graph theory operations for source.
Handles the conversion of traced edges into a connected graph (strands, bifurcations).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from ..utils.safe_unpickle import safe_load

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from source.runtime import StageController


def trace_strand_sparse(
    start: int, adjacency_list: dict[int, set[int]], visited: np.ndarray
) -> list[int]:
    """Trace a strand through connected vertices using sparse adjacency list."""
    strand = [start]
    visited[start] = True
    current = start

    while True:
        neighbors = sorted(n for n in adjacency_list[current] if not visited[n])
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


def _remove_short_hairs(
    graph_edges: dict[tuple[int, int], np.ndarray],
    adjacency_list: dict[int, set[int]],
    microns_per_voxel: np.ndarray,
    min_hair_length: float,
    graph_edge_scales: dict[tuple[int, int], np.ndarray] | None = None,
    graph_edge_energies: dict[tuple[int, int], np.ndarray] | None = None,
) -> None:
    """Remove short terminal hairs in-place."""
    if min_hair_length <= 0:
        return

    while True:
        to_remove: list[tuple[int, int]] = []
        for (v0, v1), trace in list(graph_edges.items()):
            length: float = np.sum(
                np.linalg.norm(np.diff(trace, axis=0) * microns_per_voxel, axis=1)
            )
            if length < min_hair_length and (
                len(adjacency_list[v0]) == 1 or len(adjacency_list[v1]) == 1
            ):
                to_remove.append((v0, v1))

        if not to_remove:
            return

        for v0, v1 in to_remove:
            adjacency_list[v0].discard(v1)
            adjacency_list[v1].discard(v0)
            del graph_edges[(v0, v1)]
            if graph_edge_scales is not None:
                del graph_edge_scales[(v0, v1)]
            if graph_edge_energies is not None:
                del graph_edge_energies[(v0, v1)]


def _remove_cycles(
    graph_edges: dict[tuple[int, int], np.ndarray],
    adjacency_list: dict[int, set[int]],
    n_vertices: int,
    graph_edge_scales: dict[tuple[int, int], np.ndarray] | None = None,
    graph_edge_energies: dict[tuple[int, int], np.ndarray] | None = None,
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
        return vertex

    for v0, v1 in list(graph_edges.keys()):
        root0 = find(v0)
        root1 = find(v1)
        if root0 == root1:
            cycles.append((v0, v1))
            adjacency_list[v0].discard(v1)
            adjacency_list[v1].discard(v0)
            del graph_edges[(v0, v1)]
            if graph_edge_scales is not None:
                del graph_edge_scales[(v0, v1)]
            if graph_edge_energies is not None:
                del graph_edge_energies[(v0, v1)]
        else:
            parent[root1] = root0

    return cycles


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


def _matlab_get_network_v190(
    edge_connections: np.ndarray,
    n_vertices: int,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Mirror MATLAB's ``get_network_V190`` strand and bifurcation decomposition."""
    normalized = _normalize_connections(edge_connections)
    if normalized.size == 0 or n_vertices <= 0:
        return [], [], [], np.zeros((0,), dtype=np.int32)

    edge_lookup_table, adjacency_matrix = _matlab_network_lookup_tables(normalized, n_vertices)
    degrees = np.asarray(adjacency_matrix.sum(axis=1)).ravel().astype(np.int32, copy=False)
    vertex_is_interior = degrees == 2
    vertex_is_bifurcation = degrees >= 3

    bifurcation_vertices = np.flatnonzero(vertex_is_bifurcation).astype(np.int32, copy=False)
    interior_vertices = np.flatnonzero(vertex_is_interior).astype(np.int32, copy=False)

    vertices_in_strands: list[np.ndarray] = []
    edge_indices_in_strands: list[np.ndarray] = []
    end_vertices_in_strands: list[np.ndarray] = []

    if interior_vertices.size:
        interior_adjacency = adjacency_matrix[interior_vertices][:, interior_vertices].tocsr()
        n_components, labels = connected_components(
            interior_adjacency,
            directed=False,
            connection="weak",
            return_labels=True,
        )
        for component_index in range(n_components):
            component_vertices = interior_vertices[np.flatnonzero(labels == component_index)]
            if component_vertices.size == 0:
                continue

            current_interior = component_vertices.astype(np.int32, copy=True)
            adjacency_for_strand = adjacency_matrix[current_interior, :].toarray().astype(bool)
            adjacency_for_strand_exterior = adjacency_for_strand.copy()
            adjacency_for_strand_exterior[:, current_interior] = False
            exterior_neighbor_counts = adjacency_for_strand_exterior.sum(axis=0).astype(np.int32)

            if np.all(exterior_neighbor_counts == 0):
                adjacency_interior = adjacency_matrix[current_interior][
                    :, current_interior
                ].toarray()
                interior_rows, interior_cols = _matlab_find_nonzero_matrix_entries(
                    adjacency_interior
                )
                cyclical_edge_ids = _matlab_lookup_edge_ids(
                    edge_lookup_table,
                    current_interior[interior_rows],
                    current_interior[interior_cols],
                )
                if cyclical_edge_ids.size:
                    worst_edge_vertices = normalized[int(np.max(cyclical_edge_ids))]
                    current_interior = current_interior[
                        ~np.isin(current_interior, worst_edge_vertices.astype(np.int32))
                    ]
                    adjacency_for_strand = (
                        adjacency_matrix[current_interior, :].toarray().astype(bool)
                    )
                    adjacency_for_strand_exterior = adjacency_for_strand.copy()
                    adjacency_for_strand_exterior[:, current_interior] = False
                    exterior_neighbor_counts = adjacency_for_strand_exterior.sum(axis=0).astype(
                        np.int32
                    )

            end_vertices = np.flatnonzero(exterior_neighbor_counts > 0).astype(np.int32, copy=False)
            adjacency_interior = adjacency_matrix[current_interior][:, current_interior].toarray()
            interior_rows, interior_cols = _matlab_find_nonzero_matrix_entries(adjacency_interior)
            interior_edge_ids = _matlab_lookup_edge_ids(
                edge_lookup_table,
                current_interior[interior_rows],
                current_interior[interior_cols],
            )

            endpoint_local_rows: list[int] = []
            repeated_end_vertices: list[int] = []
            adjacency_for_strand_work = adjacency_for_strand.copy()
            for end_vertex in end_vertices.tolist():
                for _ in range(int(exterior_neighbor_counts[end_vertex])):
                    local_rows = np.flatnonzero(adjacency_for_strand_work[:, end_vertex])
                    if local_rows.size == 0:
                        continue
                    local_row = int(local_rows[0])
                    endpoint_local_rows.append(local_row)
                    repeated_end_vertices.append(int(end_vertex))
                    adjacency_for_strand_work[local_row, end_vertex] = False

            if endpoint_local_rows:
                endpoint_local_rows_array = np.asarray(endpoint_local_rows, dtype=np.int32)
                repeated_end_vertices_array = np.asarray(repeated_end_vertices, dtype=np.int32)
                endpoint_edge_ids = _matlab_lookup_edge_ids(
                    edge_lookup_table,
                    repeated_end_vertices_array,
                    current_interior[endpoint_local_rows_array],
                )
                reverse_endpoint_edge_ids = _matlab_lookup_edge_ids(
                    edge_lookup_table,
                    current_interior[endpoint_local_rows_array],
                    repeated_end_vertices_array,
                )
                strand_edge_ids = np.concatenate(
                    [interior_edge_ids, endpoint_edge_ids, reverse_endpoint_edge_ids],
                )
            else:
                strand_edge_ids = interior_edge_ids

            vertices_in_strands.append(
                np.concatenate([current_interior, end_vertices]).astype(np.int32, copy=False)
            )
            edge_indices_in_strands.append(strand_edge_ids.astype(np.int32, copy=False))
            end_vertices_in_strands.append(end_vertices.astype(np.int32, copy=False))

    non_interior_vertices = np.flatnonzero(~vertex_is_interior).astype(np.int32, copy=False)
    if non_interior_vertices.size:
        adjacency_without_interiors = adjacency_matrix.toarray().astype(bool)
        adjacency_without_interiors[vertex_is_interior, :] = False
        adjacency_without_interiors[:, vertex_is_interior] = False
        extra_rows, extra_cols = _matlab_find_nonzero_matrix_entries(adjacency_without_interiors)
        extra_edge_ids = _matlab_lookup_edge_ids(edge_lookup_table, extra_rows, extra_cols)
        for edge_index in extra_edge_ids.tolist():
            vertices: np.ndarray = normalized[int(edge_index)].astype(np.int32, copy=False)
            vertices_in_strands.append(vertices.copy())
            edge_indices_in_strands.append(np.asarray([edge_index], dtype=np.int32))
            end_vertices_in_strands.append(vertices.copy())

    return (
        vertices_in_strands,
        edge_indices_in_strands,
        end_vertices_in_strands,
        bifurcation_vertices,
    )


def _matlab_sort_network_v180(
    edge_connections: np.ndarray,
    end_vertices_in_strands: list[np.ndarray],
    edge_indices_in_strands: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Mirror MATLAB's ``sort_network_V180`` strand ordering."""
    normalized = _normalize_connections(edge_connections)
    vertex_indices_in_strands: list[np.ndarray] = []
    edge_indices_in_strands_out: list[np.ndarray] = []
    edge_backwards_in_strands: list[np.ndarray] = []

    for strand_end_vertices, strand_edge_indices in zip(
        end_vertices_in_strands,
        edge_indices_in_strands,
    ):
        strand_edge_indices_array = np.asarray(strand_edge_indices, dtype=np.int32).reshape(-1)
        if strand_edge_indices_array.size == 0:
            vertex_indices_in_strands.append(np.zeros((0,), dtype=np.int32))
            edge_indices_in_strands_out.append(np.zeros((0,), dtype=np.int32))
            edge_backwards_in_strands.append(np.zeros((0,), dtype=bool))
            continue

        vertices_of_edges_at_strand = np.asarray(
            normalized[strand_edge_indices_array],
            dtype=np.int32,
        ).copy()
        number_of_edges_in_strand = len(strand_edge_indices_array)
        ordered_vertices: np.ndarray = np.zeros(
            (number_of_edges_in_strand + 1,),
            dtype=np.int32,
        )
        ordered_edge_indices: np.ndarray = np.zeros(
            (number_of_edges_in_strand,),
            dtype=np.int32,
        )
        edge_backwards: np.ndarray = np.zeros((number_of_edges_in_strand,), dtype=bool)

        if len(strand_end_vertices) > 0:
            flat_vertices = vertices_of_edges_at_strand.reshape(-1, order="F")
            matches = np.flatnonzero(flat_vertices == int(strand_end_vertices[0]))
            next_index_of_strand = int(matches[0] + 1) if matches.size else 1
        else:
            next_index_of_strand = 1

        ending_vertex_of_next_index = int(vertices_of_edges_at_strand[0, 0])
        for strand_edge_index in range(number_of_edges_in_strand):
            if next_index_of_strand > number_of_edges_in_strand:
                next_edge_origin = 1
                next_edge_terminus = 0
                next_index_of_strand -= number_of_edges_in_strand
            else:
                next_edge_origin = 0
                next_edge_terminus = 1

            row_index = int(next_index_of_strand - 1)
            ordered_vertices[strand_edge_index] = int(
                vertices_of_edges_at_strand[row_index, next_edge_origin]
            )
            ordered_edge_indices[strand_edge_index] = int(strand_edge_indices_array[row_index])
            edge_backwards[strand_edge_index] = bool(next_edge_terminus == 0)
            ending_vertex_of_next_index = int(
                vertices_of_edges_at_strand[row_index, next_edge_terminus]
            )
            vertices_of_edges_at_strand[row_index, :] = 0

            flat_vertices = vertices_of_edges_at_strand.reshape(-1, order="F")
            matches = np.flatnonzero(flat_vertices == ending_vertex_of_next_index)
            if matches.size:
                next_index_of_strand = int(matches[0] + 1)

        ordered_vertices[number_of_edges_in_strand] = ending_vertex_of_next_index
        vertex_indices_in_strands.append(ordered_vertices)
        edge_indices_in_strands_out.append(ordered_edge_indices)
        edge_backwards_in_strands.append(edge_backwards)

    return vertex_indices_in_strands, edge_indices_in_strands_out, edge_backwards_in_strands


def _matlab_get_strand_objects(
    edge_traces: list[np.ndarray],
    edge_scale_traces: list[np.ndarray],
    edge_energy_traces: list[np.ndarray],
    edge_indices_in_strands: list[np.ndarray],
    edge_backwards_in_strands: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Mirror MATLAB's ``get_strand_objects`` strand assembly from ordered edges."""
    strand_space_traces: list[np.ndarray] = []
    strand_scale_traces: list[np.ndarray] = []
    strand_energy_traces: list[np.ndarray] = []

    for strand_edge_indices, strand_edge_backwards in zip(
        edge_indices_in_strands,
        edge_backwards_in_strands,
    ):
        ordered_edge_indices = np.asarray(strand_edge_indices, dtype=np.int32).reshape(-1)
        backwards_flags = np.asarray(strand_edge_backwards, dtype=bool).reshape(-1)
        edge_traces_at_strand = [
            np.asarray(edge_traces[int(edge_index)], dtype=np.float32).copy()
            for edge_index in ordered_edge_indices.tolist()
        ]
        edge_scales_at_strand = [
            np.asarray(edge_scale_traces[int(edge_index)], dtype=np.float32).reshape(-1).copy()
            for edge_index in ordered_edge_indices.tolist()
        ]
        edge_energies_at_strand = [
            np.asarray(edge_energy_traces[int(edge_index)], dtype=np.float32).copy()
            for edge_index in ordered_edge_indices.tolist()
        ]

        for edge_ordinal in range(len(ordered_edge_indices)):
            if backwards_flags[edge_ordinal]:
                edge_traces_at_strand[edge_ordinal] = np.flipud(edge_traces_at_strand[edge_ordinal])
                edge_scales_at_strand[edge_ordinal] = np.flipud(edge_scales_at_strand[edge_ordinal])
                edge_energies_at_strand[edge_ordinal] = np.flipud(
                    edge_energies_at_strand[edge_ordinal],
                )
            if edge_ordinal < len(ordered_edge_indices) - 1:
                edge_traces_at_strand[edge_ordinal] = edge_traces_at_strand[edge_ordinal][:-1]
                edge_scales_at_strand[edge_ordinal] = edge_scales_at_strand[edge_ordinal][:-1]
                edge_energies_at_strand[edge_ordinal] = edge_energies_at_strand[edge_ordinal][:-1]

        strand_trace = np.concatenate(edge_traces_at_strand, axis=0)
        strand_scale = np.concatenate(edge_scales_at_strand, axis=0)
        strand_energy = np.concatenate(edge_energies_at_strand, axis=0)
        rounded_positions = np.rint(10.0 * strand_trace[:, :3]).astype(np.int32, copy=False)
        _unique_rows, unique_indices = np.unique(
            rounded_positions,
            axis=0,
            return_index=True,
        )
        stable_unique_indices = np.asarray(sorted(unique_indices.tolist()), dtype=np.int32)
        strand_space_traces.append(strand_trace[stable_unique_indices])
        strand_scale_traces.append(strand_scale[stable_unique_indices])
        strand_energy_traces.append(strand_energy[stable_unique_indices])

    return strand_space_traces, strand_scale_traces, strand_energy_traces


def _matlab_edge_metrics(energy_traces: list[np.ndarray]) -> np.ndarray:
    """Mirror MATLAB ``get_edge_metric(..., 'max')`` over a list of traces."""
    if not energy_traces:
        return cast("np.ndarray", np.zeros((0,), dtype=np.float32))
    metrics = np.asarray(
        [
            float(np.max(np.asarray(trace, dtype=np.float32)))
            if np.asarray(trace).size
            else float("nan")
            for trace in energy_traces
        ],
        dtype=np.float32,
    )
    metrics[np.isnan(metrics)] = np.float32(-1000.0)
    return cast("np.ndarray", metrics)


def _matlab_interp1_implicit_axis(values: np.ndarray, query_points: np.ndarray) -> np.ndarray:
    """Mirror MATLAB ``interp1(values, xq)`` where the x-axis is ``1:numel(values)``."""
    ordinate_axis = np.arange(1, len(values) + 1, dtype=np.float64)
    queries = np.asarray(query_points, dtype=np.float64)
    result = np.interp(
        queries,
        ordinate_axis,
        np.asarray(values, dtype=np.float64),
        left=np.nan,
        right=np.nan,
    )
    return cast("np.ndarray", np.asarray(result, dtype=np.float64))


def _matlab_smooth_edges_v2(
    edge_space_subscripts: list[np.ndarray],
    edge_scale_subscripts: list[np.ndarray],
    edge_energies: list[np.ndarray],
    smoothing_kernel_sigma_to_lumen_radius_ratio: float,
    lumen_radius_in_microns_range: np.ndarray,
    microns_per_voxel: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Mirror the active local-neighbor branch of MATLAB ``smooth_edges_V2``."""
    if not edge_space_subscripts:
        return [], [], []
    if np.asarray(lumen_radius_in_microns_range).size == 0:
        return (
            [
                np.asarray(space_trace, dtype=np.float32).copy()
                for space_trace in edge_space_subscripts
            ],
            [
                np.asarray(scale_trace, dtype=np.float32).reshape(-1).copy() + np.float32(1.0)
                for scale_trace in edge_scale_subscripts
            ],
            [
                np.asarray(energy_trace, dtype=np.float32).reshape(-1).copy()
                for energy_trace in edge_energies
            ],
        )

    smoothed_space_subscripts = [
        np.asarray(space_trace, dtype=np.float32).copy() for space_trace in edge_space_subscripts
    ]
    smoothed_scale_subscripts = [
        np.asarray(scale_trace, dtype=np.float32).reshape(-1).copy() + np.float32(1.0)
        for scale_trace in edge_scale_subscripts
    ]
    smoothed_energies = [
        np.asarray(energy_trace, dtype=np.float32).reshape(-1).copy()
        for energy_trace in edge_energies
    ]

    for edge_index in range(len(smoothed_space_subscripts)):
        is_inf_position = np.isneginf(smoothed_energies[edge_index])
        if np.any(~is_inf_position):
            smoothed_energies[edge_index][is_inf_position] = np.min(
                smoothed_energies[edge_index][~is_inf_position]
            )

    scale_subscript_averages = np.asarray(
        [
            float(
                np.sum(scale_trace.astype(np.float64) * energy_trace.astype(np.float64))
                / np.sum(energy_trace.astype(np.float64))
            )
            for scale_trace, energy_trace in zip(
                smoothed_scale_subscripts,
                smoothed_energies,
            )
        ],
        dtype=np.float64,
    )
    average_lumen_radii = np.exp(
        _matlab_interp1_implicit_axis(
            np.log(np.asarray(lumen_radius_in_microns_range, dtype=np.float64)),
            scale_subscript_averages,
        )
    )
    microns_per_sigma = (
        np.asarray(smoothing_kernel_sigma_to_lumen_radius_ratio, dtype=np.float64)
        * average_lumen_radii
    )

    for edge_index in range(len(smoothed_space_subscripts)):
        if not np.any(smoothed_energies[edge_index] > -np.inf):
            continue

        edge_subscripts_at_edge = np.column_stack(
            (
                np.asarray(smoothed_space_subscripts[edge_index], dtype=np.float64),
                np.asarray(smoothed_scale_subscripts[edge_index], dtype=np.float64),
            )
        )
        edge_energies_at_edge = np.asarray(smoothed_energies[edge_index], dtype=np.float64)
        edge_microns_at_edge = edge_subscripts_at_edge[:, :3] * np.asarray(
            microns_per_voxel,
            dtype=np.float64,
        )
        edge_cumulative_length = np.concatenate(
            (
                np.zeros((1,), dtype=np.float64),
                np.cumsum(
                    np.sqrt(
                        np.sum(
                            np.diff(edge_microns_at_edge[:, :3], axis=0) ** 2,
                            axis=1,
                        )
                    )
                ),
            )
        )
        kernel_micron_domains = edge_cumulative_length[:, None] - edge_cumulative_length[None, :]
        sigma = float(microns_per_sigma[edge_index])
        kernel_sigma_domains = kernel_micron_domains / sigma
        gaussian_kernels = np.exp(-(kernel_sigma_domains**2) / 2.0)
        energy_conv_kernel = np.sum(edge_energies_at_edge[:, None] * gaussian_kernels, axis=0)
        energy_conv_energy_conv_kernel = np.sum(
            (edge_energies_at_edge[:, None] ** 2) * gaussian_kernels,
            axis=0,
        )
        subscript_conv_energy_conv_kernel = np.sum(
            edge_subscripts_at_edge[:, None, :]
            * edge_energies_at_edge[:, None, None]
            * gaussian_kernels[:, :, None],
            axis=0,
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            edge_subscripts_smoothed = (
                subscript_conv_energy_conv_kernel / energy_conv_kernel[:, None]
            )
            edge_energies_smoothed = energy_conv_energy_conv_kernel / energy_conv_kernel

        edge_subscripts_smoothed[0, :] = edge_subscripts_at_edge[0, :]
        edge_subscripts_smoothed[-1, :] = edge_subscripts_at_edge[-1, :]
        edge_energies_smoothed[0] = edge_energies_at_edge[0]
        edge_energies_smoothed[-1] = edge_energies_at_edge[-1]

        edge_cumulative_lengths = np.concatenate(
            (
                np.zeros((1,), dtype=np.float64),
                np.cumsum(
                    np.max(
                        np.abs(np.diff(edge_subscripts_smoothed[:, :3], axis=0)),
                        axis=1,
                    )
                ),
            )
        )
        edge_sample_lengths = np.linspace(
            0.0,
            float(edge_cumulative_lengths[-1]),
            num=len(edge_cumulative_lengths),
            dtype=np.float64,
        )

        sampled_subscripts = np.column_stack(
            [
                np.interp(
                    edge_sample_lengths,
                    edge_cumulative_lengths,
                    edge_subscripts_smoothed[:, dimension],
                )
                for dimension in range(edge_subscripts_smoothed.shape[1])
            ]
        )
        sampled_energies = np.interp(
            edge_sample_lengths,
            edge_cumulative_lengths,
            edge_energies_smoothed,
        )

        smoothed_space_subscripts[edge_index] = sampled_subscripts[:, :3].astype(
            np.float32,
            copy=False,
        )
        smoothed_scale_subscripts[edge_index] = sampled_subscripts[:, 3].astype(
            np.float32,
            copy=False,
        )
        smoothed_energies[edge_index] = sampled_energies.astype(np.float32, copy=False)

    return (
        smoothed_space_subscripts,
        smoothed_scale_subscripts,
        smoothed_energies,
    )


def _matlab_get_vessel_directions_v3(
    strand_space_subscripts: list[np.ndarray],
    microns_per_voxel: np.ndarray,
) -> list[np.ndarray]:
    """Mirror MATLAB ``get_vessel_directions_V3`` over strand objects."""
    vessel_directions: list[np.ndarray] = []
    microns_per_voxel_arr = np.asarray(microns_per_voxel, dtype=np.float32)

    for strand in strand_space_subscripts:
        strand_coords = np.asarray(strand, dtype=np.float32)
        if len(strand_coords) == 0:
            vessel_directions.append(np.zeros((0, 3), dtype=np.float32))
            continue

        strand_coords = strand_coords * microns_per_voxel_arr
        if len(strand_coords) > 2:
            cropped = strand_coords[2:, :] - strand_coords[:-2, :]
            directions = np.vstack((cropped[0:1, :], cropped, cropped[-1:, :]))
        elif len(strand_coords) == 2:
            cropped = strand_coords[1:2, :] - strand_coords[0:1, :]
            directions = np.vstack((cropped, cropped))
        else:
            directions = np.zeros((1, 3), dtype=np.float32)

        norms = np.sqrt(np.sum(directions**2, axis=1, keepdims=True))
        with np.errstate(divide="ignore", invalid="ignore"):
            unit_directions = directions / norms
        vessel_directions.append(np.asarray(unit_directions, dtype=np.float32))

    return vessel_directions


def _matlab_network_topology(
    edge_connections: np.ndarray,
    edge_traces: list[np.ndarray],
    edge_scale_traces: list[np.ndarray],
    edge_energy_traces: list[np.ndarray],
    n_vertices: int,
) -> dict[str, Any]:
    """Construct MATLAB-shaped strand topology and strand objects."""
    (
        _vertices_in_strands_unsorted,
        edge_indices_in_strands_unsorted,
        end_vertices_in_strands,
        bifurcation_vertices,
    ) = _matlab_get_network_v190(edge_connections, n_vertices)
    (
        vertex_indices_in_strands,
        edge_indices_in_strands,
        edge_backwards_in_strands,
    ) = _matlab_sort_network_v180(
        edge_connections,
        end_vertices_in_strands,
        edge_indices_in_strands_unsorted,
    )
    strand_space_traces, strand_scale_traces, strand_energy_traces = _matlab_get_strand_objects(
        edge_traces,
        edge_scale_traces,
        edge_energy_traces,
        edge_indices_in_strands,
        edge_backwards_in_strands,
    )
    return {
        "strands": [strand.tolist() for strand in vertex_indices_in_strands],
        "mismatched_strands": [],
        "bifurcations": bifurcation_vertices.astype(np.int32, copy=False),
        "edge_indices_in_strands": edge_indices_in_strands,
        "edge_backwards_in_strands": edge_backwards_in_strands,
        "end_vertices_in_strands": end_vertices_in_strands,
        "strand_traces": strand_space_traces,
        "strand_space_traces": strand_space_traces,
        "strand_scale_traces": strand_scale_traces,
        "strand_energy_traces": strand_energy_traces,
    }


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


def _network_payload(
    adjacency_list: dict[int, set[int]],
    graph_edges: dict[tuple[int, int], np.ndarray],
    graph_edge_scales: dict[tuple[int, int], np.ndarray],
    graph_edge_energies: dict[tuple[int, int], np.ndarray],
    dangling_edges: list[dict[str, Any]],
    cycles: list[tuple[int, int]],
    n_vertices: int,
    *,
    lumen_radius_microns: np.ndarray,
    microns_per_voxel: np.ndarray,
) -> dict[str, Any]:
    """Build the final network payload from shared graph state."""
    (
        pruned_connections,
        pruned_traces,
        pruned_scale_traces,
        pruned_energy_traces,
    ) = _graph_state_ordered_edges(
        graph_edges,
        graph_edge_scales,
        graph_edge_energies,
    )
    topology = _matlab_network_topology(
        pruned_connections,
        pruned_traces,
        pruned_scale_traces,
        pruned_energy_traces,
        n_vertices,
    )
    sigma_strand_smoothing = float(np.sqrt(2.0) / 2.0)
    strand_space_traces = cast("list[np.ndarray]", topology["strand_space_traces"])
    strand_scale_traces = cast("list[np.ndarray]", topology["strand_scale_traces"])
    strand_energy_traces = cast("list[np.ndarray]", topology["strand_energy_traces"])
    if sigma_strand_smoothing and np.asarray(lumen_radius_microns).size > 0:
        (
            strand_space_traces,
            strand_scale_traces,
            strand_energy_traces,
        ) = _matlab_smooth_edges_v2(
            strand_space_traces,
            strand_scale_traces,
            strand_energy_traces,
            sigma_strand_smoothing,
            lumen_radius_microns,
            microns_per_voxel,
        )
    vessel_directions = _matlab_get_vessel_directions_v3(
        strand_space_traces,
        microns_per_voxel,
    )
    mean_strand_energies = _matlab_edge_metrics(strand_energy_traces)
    strand_subscripts = [
        np.column_stack((space_trace, scale_trace))
        for space_trace, scale_trace in zip(
            strand_space_traces,
            strand_scale_traces,
        )
    ]
    vertex_degrees = _vertex_degrees(adjacency_list, n_vertices)
    orphans = np.where(vertex_degrees == 0)[0].astype(np.int32)

    payload = {
        "strands": topology["strands"],
        "bifurcations": topology["bifurcations"],
        "orphans": orphans,
        "cycles": cycles,
        "mismatched_strands": topology["mismatched_strands"],
        "adjacency_list": adjacency_list,
        "vertex_degrees": vertex_degrees,
        "graph_edges": graph_edges,
        "graph_edge_scales": graph_edge_scales,
        "graph_edge_energies": graph_edge_energies,
        "dangling_edges": dangling_edges,
        "edge_indices_in_strands": topology["edge_indices_in_strands"],
        "edge_backwards_in_strands": topology["edge_backwards_in_strands"],
        "end_vertices_in_strands": topology["end_vertices_in_strands"],
        "strand_subscripts": strand_subscripts,
        "strand_traces": strand_space_traces,
        "strand_space_traces": strand_space_traces,
        "strand_scale_traces": strand_scale_traces,
        "strand_energy_traces": strand_energy_traces,
        "mean_strand_energies": mean_strand_energies,
        "vessel_directions": vessel_directions,
    }
    return payload


def construct_network(
    edges: dict[str, Any], vertices: dict[str, Any], params: dict[str, Any]
) -> dict[str, Any]:
    """Construct network from traced edges and detected vertices."""
    logger.info("Constructing network")

    edge_traces = edges["traces"]
    edge_scale_traces = edges.get(
        "scale_traces",
        [np.zeros((len(np.asarray(trace)),), dtype=np.float32) for trace in edge_traces],
    )
    edge_energy_traces = edges.get(
        "energy_traces",
        [np.zeros((len(np.asarray(trace)),), dtype=np.float32) for trace in edge_traces],
    )
    edge_connections = _normalize_connections(edges["connections"])
    vertex_positions = np.asarray(vertices["positions"], dtype=np.float32)
    bridge_vertex_positions = np.asarray(
        edges.get("bridge_vertex_positions", np.empty((0, 3), dtype=np.float32)),
        dtype=np.float32,
    ).reshape(-1, 3)
    if bridge_vertex_positions.size:
        vertex_positions = np.vstack([vertex_positions, bridge_vertex_positions]).astype(
            np.float32,
            copy=False,
        )
    n_vertices = len(vertex_positions)

    microns_per_voxel = np.array(params.get("microns_per_voxel", [1.0, 1.0, 1.0]), dtype=float)
    lumen_radius_microns = np.asarray(
        edges.get("lumen_radius_microns", params.get("lumen_radius_microns", [])),
        dtype=np.float32,
    ).reshape(-1)
    min_hair_length = params.get("min_hair_length_in_microns", 0.0)
    remove_cycles = bool(params.get("remove_cycles", False))

    adjacency_list, graph_edges, graph_edge_scales, graph_edge_energies, dangling_edges = (
        _build_graph_state(
            edge_traces,
            edge_scale_traces,
            edge_energy_traces,
            edge_connections,
            n_vertices,
        )
    )

    _remove_short_hairs(
        graph_edges,
        adjacency_list,
        microns_per_voxel,
        float(min_hair_length),
        graph_edge_scales,
        graph_edge_energies,
    )
    cycles = (
        _remove_cycles(
            graph_edges,
            adjacency_list,
            n_vertices,
            graph_edge_scales,
            graph_edge_energies,
        )
        if remove_cycles
        else []
    )

    network = _network_payload(
        adjacency_list,
        graph_edges,
        graph_edge_scales,
        graph_edge_energies,
        dangling_edges,
        cycles,
        n_vertices,
        lumen_radius_microns=lumen_radius_microns,
        microns_per_voxel=microns_per_voxel,
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
    from source.runtime.run_state import atomic_joblib_dump

    edge_traces = edges["traces"]
    edge_scale_traces = edges.get(
        "scale_traces",
        [np.zeros((len(np.asarray(trace)),), dtype=np.float32) for trace in edge_traces],
    )
    edge_energy_traces = edges.get(
        "energy_traces",
        [np.zeros((len(np.asarray(trace)),), dtype=np.float32) for trace in edge_traces],
    )
    edge_connections = _normalize_connections(edges["connections"])
    vertex_positions = np.asarray(vertices["positions"], dtype=np.float32)
    bridge_vertex_positions = np.asarray(
        edges.get("bridge_vertex_positions", np.empty((0, 3), dtype=np.float32)),
        dtype=np.float32,
    ).reshape(-1, 3)
    if bridge_vertex_positions.size:
        vertex_positions = np.vstack([vertex_positions, bridge_vertex_positions]).astype(
            np.float32,
            copy=False,
        )
    n_vertices = len(vertex_positions)

    microns_per_voxel = np.array(params.get("microns_per_voxel", [1.0, 1.0, 1.0]), dtype=float)
    lumen_radius_microns = np.asarray(
        edges.get("lumen_radius_microns", params.get("lumen_radius_microns", [])),
        dtype=np.float32,
    ).reshape(-1)
    min_hair_length = params.get("min_hair_length_in_microns", 0.0)
    remove_cycles = bool(params.get("remove_cycles", False))

    stage_controller.begin(detail="Building network graph", units_total=5, substage="adjacency")
    adjacency_path = stage_controller.artifact_path("adjacency.pkl")
    pruned_path = stage_controller.artifact_path("hair_pruned.pkl")
    cycle_path = stage_controller.artifact_path("cycle_pruned.pkl")
    strands_path = stage_controller.artifact_path("strands.pkl")

    if not adjacency_path.exists():
        adjacency_list, graph_edges, graph_edge_scales, graph_edge_energies, dangling_edges = (
            _build_graph_state(
                edge_traces,
                edge_scale_traces,
                edge_energy_traces,
                edge_connections,
                n_vertices,
            )
        )
        atomic_joblib_dump(
            {
                "adjacency_list": adjacency_list,
                "graph_edges": graph_edges,
                "graph_edge_scales": graph_edge_scales,
                "graph_edge_energies": graph_edge_energies,
                "dangling_edges": dangling_edges,
            },
            adjacency_path,
        )

    adjacency_payload = safe_load(adjacency_path)
    adjacency_list = adjacency_payload["adjacency_list"]
    graph_edges = adjacency_payload["graph_edges"]
    graph_edge_scales = adjacency_payload["graph_edge_scales"]
    graph_edge_energies = adjacency_payload["graph_edge_energies"]
    dangling_edges = adjacency_payload["dangling_edges"]
    stage_controller.update(units_total=5, units_completed=1, substage="adjacency")

    if min_hair_length > 0 and not pruned_path.exists():
        _remove_short_hairs(
            graph_edges,
            adjacency_list,
            microns_per_voxel,
            float(min_hair_length),
            graph_edge_scales,
            graph_edge_energies,
        )
        atomic_joblib_dump(
            {
                "adjacency_list": adjacency_list,
                "graph_edges": graph_edges,
                "graph_edge_scales": graph_edge_scales,
                "graph_edge_energies": graph_edge_energies,
            },
            pruned_path,
        )
    if pruned_path.exists():
        pruned_payload = safe_load(pruned_path)
        adjacency_list = pruned_payload["adjacency_list"]
        graph_edges = pruned_payload["graph_edges"]
        graph_edge_scales = pruned_payload["graph_edge_scales"]
        graph_edge_energies = pruned_payload["graph_edge_energies"]
    stage_controller.update(units_total=5, units_completed=2, substage="hair_prune")

    cycles: list[tuple[int, int]] = []
    if remove_cycles and graph_edges and not cycle_path.exists():
        cycles = _remove_cycles(
            graph_edges,
            adjacency_list,
            n_vertices,
            graph_edge_scales,
            graph_edge_energies,
        )
        atomic_joblib_dump(
            {
                "adjacency_list": adjacency_list,
                "graph_edges": graph_edges,
                "graph_edge_scales": graph_edge_scales,
                "graph_edge_energies": graph_edge_energies,
                "cycles": cycles,
            },
            cycle_path,
        )
    if cycle_path.exists():
        cycle_payload = safe_load(cycle_path)
        adjacency_list = cycle_payload["adjacency_list"]
        graph_edges = cycle_payload["graph_edges"]
        graph_edge_scales = cycle_payload["graph_edge_scales"]
        graph_edge_energies = cycle_payload["graph_edge_energies"]
        cycles = cycle_payload["cycles"]
    stage_controller.update(units_total=5, units_completed=3, substage="cycle_prune")

    if not strands_path.exists():
        (
            pruned_connections,
            pruned_traces,
            pruned_scale_traces,
            pruned_energy_traces,
        ) = _graph_state_ordered_edges(
            graph_edges,
            graph_edge_scales,
            graph_edge_energies,
        )
        topology = _matlab_network_topology(
            pruned_connections,
            pruned_traces,
            pruned_scale_traces,
            pruned_energy_traces,
            n_vertices,
        )
        atomic_joblib_dump(topology, strands_path)
    topology = safe_load(strands_path)
    stage_controller.update(units_total=5, units_completed=4, substage="strand_trace")

    network = _network_payload(
        adjacency_list,
        graph_edges,
        graph_edge_scales,
        graph_edge_energies,
        dangling_edges,
        cycles,
        n_vertices,
        lumen_radius_microns=lumen_radius_microns,
        microns_per_voxel=microns_per_voxel,
    )
    network["strands"] = topology["strands"]
    network["mismatched_strands"] = topology["mismatched_strands"]
    stage_controller.update(units_total=5, units_completed=5, substage="finalize")
    return network


__all__ = [
    "_matlab_edge_metrics",
    "_matlab_get_network_v190",
    "_matlab_get_strand_objects",
    "_matlab_get_vessel_directions_v3",
    "_matlab_network_topology",
    "_matlab_smooth_edges_v2",
    "_matlab_sort_network_v180",
    "construct_network",
    "construct_network_resumable",
    "sort_and_validate_strands_sparse",
    "trace_strand_sparse",
]



