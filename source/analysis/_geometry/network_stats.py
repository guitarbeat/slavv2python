from __future__ import annotations

from typing import TYPE_CHECKING, Any

import networkx as nx
import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator


def calculate_branching_angles(
    strands: list[list[int]],
    vertex_positions: np.ndarray,
    microns_per_voxel: list[float],
    bifurcations: np.ndarray,
) -> list[float]:
    """Compute pairwise angles at each bifurcation."""
    vertex_positions_arr = np.asarray(vertex_positions, dtype=float)
    scale = np.asarray(microns_per_voxel, dtype=float)
    bifurcation_ids = {int(b) for b in np.asarray(bifurcations, dtype=int)}
    directions: dict[int, list[np.ndarray]] = {}

    for strand in strands:
        if len(strand) < 2:
            continue
        start, next_idx = strand[0], strand[1]
        end, prev_idx = strand[-1], strand[-2]

        vec_start = (vertex_positions_arr[next_idx] - vertex_positions_arr[start]) * scale
        vec_end = (vertex_positions_arr[prev_idx] - vertex_positions_arr[end]) * scale

        if start in bifurcation_ids:
            directions.setdefault(start, []).append(vec_start)
        if end in bifurcation_ids:
            directions.setdefault(end, []).append(vec_end)

    angles: list[float] = []
    for direction_vecs in directions.values():
        valid_vectors = [vec for vec in direction_vecs if np.linalg.norm(vec) > 0]
        if len(valid_vectors) < 2:
            continue
        for i in range(len(valid_vectors)):
            for j in range(i + 1, len(valid_vectors)):
                direction_vec_a = valid_vectors[i]
                direction_vec_b = valid_vectors[j]
                norm_a = np.linalg.norm(direction_vec_a)
                norm_b = np.linalg.norm(direction_vec_b)
                if norm_a == 0 or norm_b == 0:
                    continue
                cosang = np.clip(
                    np.dot(direction_vec_a, direction_vec_b) / (norm_a * norm_b),
                    -1.0,
                    1.0,
                )
                angles.append(float(np.degrees(np.arccos(cosang))))

    return angles


def calculate_surface_area(
    strands: list[list[int]],
    vertex_positions: np.ndarray,
    radii: np.ndarray,
    microns_per_voxel: list[float],
) -> float:
    """Compute total vessel surface area."""
    vertex_positions_arr = np.asarray(vertex_positions, dtype=float)
    radii_arr = np.asarray(radii, dtype=float)
    scale = np.asarray(microns_per_voxel, dtype=float)
    return float(
        sum(
            2 * np.pi * radius * length
            for length, radius in _iter_segment_lengths_and_radii(
                strands,
                vertex_positions_arr,
                radii_arr,
                scale,
            )
        )
    )


def calculate_vessel_volume(
    strands: list[list[int]],
    vertex_positions: np.ndarray,
    radii: np.ndarray,
    microns_per_voxel: list[float],
) -> float:
    """Compute total vessel volume."""
    vertex_positions_arr = np.asarray(vertex_positions, dtype=float)
    radii_arr = np.asarray(radii, dtype=float)
    scale = np.asarray(microns_per_voxel, dtype=float)
    return float(
        sum(
            np.pi * radius**2 * length
            for length, radius in _iter_segment_lengths_and_radii(
                strands,
                vertex_positions_arr,
                radii_arr,
                scale,
            )
        )
    )


def _iter_segment_lengths_and_radii(
    strands: list[list[int]],
    vertex_positions: np.ndarray,
    radii: np.ndarray,
    scale: np.ndarray,
) -> Iterator[tuple[float, float]]:
    """Yield scaled segment lengths with their midpoint radii."""
    for strand in strands:
        if len(strand) < 2:
            continue
        for start_vertex, end_vertex in zip(strand, strand[1:]):
            pos1 = vertex_positions[start_vertex] * scale
            pos2 = vertex_positions[end_vertex] * scale
            length = float(np.linalg.norm(pos2 - pos1))
            radius = float(0.5 * (radii[start_vertex] + radii[end_vertex]))
            yield length, radius


def calculate_network_statistics(
    strands: list[list[int]],
    bifurcations: np.ndarray,
    vertex_positions: np.ndarray,
    radii: np.ndarray,
    microns_per_voxel: list[float],
    image_shape: tuple[int, ...],
    edge_energies: np.ndarray | None = None,
) -> dict[str, Any]:
    """Calculate aggregate metrics for a traced vascular network."""
    stats: dict[str, Any] = {
        "num_strands": len(strands),
        "num_bifurcations": len(bifurcations),
        "num_vertices": len(vertex_positions),
    }
    graph_obj, strand_lengths, edge_lengths, edge_radii, tortuosities = _network_graph_components(
        strands,
        vertex_positions,
        radii,
        microns_per_voxel,
    )

    _update_mean_std_stats(stats, strand_lengths, "mean_strand_length", "strand_length_std")
    stats["total_length"] = float(np.sum(strand_lengths)) if strand_lengths else 0.0
    _update_mean_std_stats(stats, tortuosities, "mean_tortuosity", "tortuosity_std")
    _update_graph_stats(stats, graph_obj)
    _update_mean_std_stats(stats, edge_lengths, "mean_edge_length", "edge_length_std")
    _update_mean_std_stats(stats, edge_radii, "mean_edge_radius", "edge_radius_std")
    _update_branch_angle_stats(stats, strands, vertex_positions, microns_per_voxel, bifurcations)

    if len(radii) > 0:
        stats["mean_radius"] = np.mean(radii)
        stats["radius_std"] = np.std(radii)
        stats["min_radius"] = np.min(radii)
        stats["max_radius"] = np.max(radii)

    if edge_energies is not None and len(edge_energies) > 0:
        stats["mean_edge_energy"] = float(np.mean(edge_energies))
        stats["edge_energy_std"] = float(np.std(edge_energies))
    else:
        stats["mean_edge_energy"] = 0.0
        stats["edge_energy_std"] = 0.0

    total_volume = calculate_vessel_volume(strands, vertex_positions, radii, microns_per_voxel)
    image_volume = np.prod(image_shape) * np.prod(microns_per_voxel)
    stats["total_volume"] = total_volume
    stats["volume_fraction"] = total_volume / image_volume if image_volume > 0 else 0

    total_surface_area = calculate_surface_area(strands, vertex_positions, radii, microns_per_voxel)
    stats["total_surface_area"] = total_surface_area
    stats["surface_area_density"] = total_surface_area / image_volume if image_volume > 0 else 0
    stats["length_density"] = stats.get("total_length", 0) / image_volume if image_volume > 0 else 0
    stats["bifurcation_density"] = len(bifurcations) / image_volume if image_volume > 0 else 0
    stats["vertex_density"] = stats["num_vertices"] / image_volume if image_volume > 0 else 0
    stats["edge_density"] = stats["num_edges"] / image_volume if image_volume > 0 else 0
    return stats


def _network_graph_components(
    strands: list[list[int]],
    vertex_positions: np.ndarray,
    radii: np.ndarray,
    microns_per_voxel: list[float],
) -> tuple[nx.Graph, list[float], list[float], list[float], list[float]]:
    graph_obj = nx.Graph()
    graph_obj.add_nodes_from(range(len(vertex_positions)))
    strand_lengths: list[float] = []
    edge_lengths: list[float] = []
    edge_radii: list[float] = []
    tortuosities: list[float] = []
    scale = np.asarray(microns_per_voxel, dtype=float)
    vertex_positions_arr = np.asarray(vertex_positions, dtype=float)
    radii_arr = np.asarray(radii, dtype=float)
    for strand in strands:
        if len(strand) <= 1:
            continue
        strand_length = 0.0
        for idx in range(len(strand) - 1):
            start_idx = strand[idx]
            end_idx = strand[idx + 1]
            pos1 = vertex_positions_arr[start_idx] * scale
            pos2 = vertex_positions_arr[end_idx] * scale
            seg_length = np.linalg.norm(pos2 - pos1)
            strand_length += seg_length
            edge_lengths.append(float(seg_length))
            edge_radii.append(float((radii_arr[start_idx] + radii_arr[end_idx]) / 2.0))
            graph_obj.add_edge(start_idx, end_idx, weight=seg_length)
        strand_lengths.append(float(strand_length))
        euclidean = np.linalg.norm(
            (vertex_positions_arr[strand[-1]] - vertex_positions_arr[strand[0]]) * scale
        )
        if euclidean > 0:
            tortuosities.append(float(strand_length / euclidean))
    return graph_obj, strand_lengths, edge_lengths, edge_radii, tortuosities


def _update_mean_std_stats(
    stats: dict[str, Any],
    values: list[float],
    mean_key: str,
    std_key: str,
) -> None:
    if values:
        stats[mean_key] = float(np.mean(values))
        stats[std_key] = float(np.std(values))
        return
    stats[mean_key] = 0.0
    stats[std_key] = 0.0


def _pairwise_path_lengths(graph_obj: nx.Graph) -> list[float]:
    pairwise_lengths: list[float] = []
    for component in nx.connected_components(graph_obj):
        subgraph = graph_obj.subgraph(component)
        if subgraph.number_of_nodes() < 2:
            continue
        for start_node, dist_map in nx.all_pairs_dijkstra_path_length(subgraph, weight="weight"):
            pairwise_lengths.extend(
                distance for end_node, distance in dist_map.items() if start_node < end_node
            )
    return pairwise_lengths


def _update_centrality_stats(stats: dict[str, Any], graph_obj: nx.Graph) -> None:
    if betweenness := nx.betweenness_centrality(graph_obj, weight="weight"):
        values = np.fromiter(betweenness.values(), dtype=float)
        stats["betweenness_mean"] = float(np.mean(values))
        stats["betweenness_std"] = float(np.std(values))
    else:
        stats["betweenness_mean"] = 0.0
        stats["betweenness_std"] = 0.0

    if closeness := nx.closeness_centrality(graph_obj, distance="weight"):
        values = np.fromiter(closeness.values(), dtype=float)
        stats["closeness_mean"] = float(np.mean(values))
        stats["closeness_std"] = float(np.std(values))
    else:
        stats["closeness_mean"] = 0.0
        stats["closeness_std"] = 0.0

    eigen = (
        nx.eigenvector_centrality(graph_obj, weight="weight", max_iter=1000)
        if graph_obj.number_of_nodes() > 0
        else {}
    )
    if eigen:
        values = np.fromiter(eigen.values(), dtype=float)
        stats["eigenvector_mean"] = float(np.mean(values))
        stats["eigenvector_std"] = float(np.std(values))
    else:
        stats["eigenvector_mean"] = 0.0
        stats["eigenvector_std"] = 0.0


def _update_graph_stats(stats: dict[str, Any], graph_obj: nx.Graph) -> None:
    stats["num_edges"] = graph_obj.number_of_edges()
    if degrees := [degree for _, degree in graph_obj.degree()]:
        stats["mean_degree"] = float(np.mean(degrees))
        stats["degree_std"] = float(np.std(degrees))
    else:
        stats["mean_degree"] = 0.0
        stats["degree_std"] = 0.0

    stats["num_connected_components"] = nx.number_connected_components(graph_obj)
    stats["num_endpoints"] = sum(degree == 1 for _, degree in graph_obj.degree())
    if pairwise_lengths := _pairwise_path_lengths(graph_obj):
        stats["avg_path_length"] = float(np.mean(pairwise_lengths))
        stats["network_diameter"] = float(np.max(pairwise_lengths))
    else:
        stats["avg_path_length"] = 0.0
        stats["network_diameter"] = 0.0
    stats["clustering_coefficient"] = (
        float(nx.average_clustering(graph_obj, weight="weight"))
        if graph_obj.number_of_nodes() > 1
        else 0.0
    )
    stats["graph_density"] = float(nx.density(graph_obj))
    _update_centrality_stats(stats, graph_obj)


def _update_branch_angle_stats(
    stats: dict[str, Any],
    strands: list[list[int]],
    vertex_positions: np.ndarray,
    microns_per_voxel: list[float],
    bifurcations: np.ndarray,
) -> None:
    if angles := calculate_branching_angles(
        strands,
        vertex_positions,
        microns_per_voxel,
        bifurcations,
    ):
        stats["mean_branch_angle"] = float(np.mean(angles))
        stats["branch_angle_std"] = float(np.std(angles))
        return
    stats["mean_branch_angle"] = 0.0
    stats["branch_angle_std"] = 0.0
