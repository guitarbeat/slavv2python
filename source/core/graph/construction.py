"""
Main network construction logic for SLAVV.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from ...utils.safe_unpickle import safe_load
from .base import (
    _build_graph_state,
    _graph_state_ordered_edges,
    _normalize_connections,
    _vertex_degrees,
)
from .metrics import _matlab_edge_metrics
from .operations import (
    _matlab_get_vessel_directions_v3,
    _matlab_network_topology,
    _matlab_smooth_edges_v2,
    _remove_cycles,
    _remove_short_hairs,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from source.runtime import StageController


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
