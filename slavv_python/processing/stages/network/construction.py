"""
Main network construction logic for SLAVV.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from slavv_python.schema.results import EdgeSet, NetworkResult, VertexSet
from .base import (
    _graph_state_ordered_edges,
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
    from slavv_python.engine.state import StageController


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
) -> NetworkResult:
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

    return NetworkResult.create(
        strands=topology["strands"],
        bifurcations=topology["bifurcations"],
        vertex_degrees=vertex_degrees,
        orphans=orphans,
        cycles=cycles,
        mismatched_strands=topology["mismatched_strands"],
        adjacency_list=adjacency_list,
        graph_edges=graph_edges,
        graph_edge_scales=graph_edge_scales,
        graph_edge_energies=graph_edge_energies,
        dangling_edges=dangling_edges,
        edge_indices_in_strands=topology["edge_indices_in_strands"],
        edge_backwards_in_strands=topology["edge_backwards_in_strands"],
        end_vertices_in_strands=topology["end_vertices_in_strands"],
        strand_subscripts=strand_subscripts,
        strand_traces=strand_space_traces,
        strand_space_traces=strand_space_traces,
        strand_scale_traces=strand_scale_traces,
        strand_energy_traces=strand_energy_traces,
        mean_strand_energies=mean_strand_energies,
        vessel_directions=vessel_directions,
    )


def construct_network(edges: EdgeSet, vertices: VertexSet, params: dict[str, Any]) -> NetworkResult:
    """Construct network from traced edges and detected vertices."""
    from slavv_python.processing.stages.network.manager import NetworkManager

    return NetworkManager.run(edges, vertices, params)


def construct_network_resumable(
    edges: EdgeSet,
    vertices: VertexSet,
    params: dict[str, Any],
    stage_controller: StageController,
) -> NetworkResult:
    """Construct a network while persisting stage-level substeps."""
    from slavv_python.processing.stages.network.manager import NetworkManager

    return NetworkManager.run_resumable(edges, vertices, params, stage_controller)
