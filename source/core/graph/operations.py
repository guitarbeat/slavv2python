"""
Graph manipulation and traversal logic for SLAVV.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from scipy.sparse.csgraph import connected_components

from .base import (
    _matlab_find_nonzero_matrix_entries,
    _matlab_lookup_edge_ids,
    _matlab_network_lookup_tables,
    _normalize_connections,
)


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
        gaussian_kernels = np.exp(-(kernel_sigma_domains ** 2) / 2.0)
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

        norms = np.sqrt(np.sum(directions ** 2, axis=1, keepdims=True))
        with np.errstate(divide="ignore", invalid="ignore"):
            unit_directions = directions / norms
        vessel_directions.append(np.asarray(unit_directions, dtype=np.float32))

    return vessel_directions
