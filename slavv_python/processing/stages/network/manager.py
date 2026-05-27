"""Consolidated network construction manager."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from slavv_python.schema.results import (  # noqa: TC001
    EdgeSet,
    NetworkResult,
    VertexSet,
)
from slavv_python.utils.safe_unpickle import safe_load

from .base import _build_graph_state, _graph_state_ordered_edges, _normalize_connections
from .construction import _network_payload
from .operations import _matlab_network_topology, _remove_cycles, _remove_short_hairs

if TYPE_CHECKING:
    from slavv_python.engine.state import StageController

logger = logging.getLogger(__name__)


@dataclass
class _GraphBuildInputs:
    edge_traces: list[np.ndarray]
    edge_scale_traces: list[np.ndarray]
    edge_energy_traces: list[np.ndarray]
    edge_connections: np.ndarray
    n_vertices: int
    microns_per_voxel: np.ndarray
    lumen_radius_microns: np.ndarray
    min_hair_length: float
    remove_cycles: bool


class NetworkManager:
    """Deep facade for network graph construction (ephemeral and resumable)."""

    @classmethod
    def _inputs_from_stages(
        cls, edges: EdgeSet, vertices: VertexSet, params: dict[str, Any]
    ) -> _GraphBuildInputs:
        edge_traces = edges.traces
        edge_scale_traces = edges.extra.get(
            "scale_traces",
            [np.zeros((len(np.asarray(trace)),), dtype=np.float32) for trace in edge_traces],
        )
        edge_energy_traces = edges.extra.get(
            "energy_traces",
            [np.zeros((len(np.asarray(trace)),), dtype=np.float32) for trace in edge_traces],
        )
        edge_connections = _normalize_connections(edges.connections)
        vertex_positions = np.asarray(vertices.positions, dtype=np.float32)
        bridge_vertex_positions = np.asarray(
            edges.extra.get("bridge_vertex_positions", np.empty((0, 3), dtype=np.float32)),
            dtype=np.float32,
        ).reshape(-1, 3)
        if bridge_vertex_positions.size:
            vertex_positions = np.vstack([vertex_positions, bridge_vertex_positions]).astype(
                np.float32,
                copy=False,
            )
        return _GraphBuildInputs(
            edge_traces=edge_traces,
            edge_scale_traces=edge_scale_traces,
            edge_energy_traces=edge_energy_traces,
            edge_connections=edge_connections,
            n_vertices=len(vertex_positions),
            microns_per_voxel=np.array(
                params.get("microns_per_voxel", [1.0, 1.0, 1.0]), dtype=float
            ),
            lumen_radius_microns=np.asarray(
                edges.extra.get("lumen_radius_microns", params.get("lumen_radius_microns", [])),
                dtype=np.float32,
            ).reshape(-1),
            min_hair_length=float(params.get("min_hair_length_in_microns", 0.0)),
            remove_cycles=bool(params.get("remove_cycles", False)),
        )

    @classmethod
    def run(cls, edges: EdgeSet, vertices: VertexSet, params: dict[str, Any]) -> NetworkResult:
        """Construct network from traced edges and detected vertices."""
        logger.info("Constructing network")
        inputs = cls._inputs_from_stages(edges, vertices, params)

        adjacency_list, graph_edges, graph_edge_scales, graph_edge_energies, dangling_edges = (
            _build_graph_state(
                inputs.edge_traces,
                inputs.edge_scale_traces,
                inputs.edge_energy_traces,
                inputs.edge_connections,
                inputs.n_vertices,
            )
        )

        _remove_short_hairs(
            graph_edges,
            adjacency_list,
            inputs.microns_per_voxel,
            inputs.min_hair_length,
            graph_edge_scales,
            graph_edge_energies,
        )
        cycles = (
            _remove_cycles(
                graph_edges,
                adjacency_list,
                inputs.n_vertices,
                graph_edge_scales,
                graph_edge_energies,
            )
            if inputs.remove_cycles
            else []
        )

        return _network_payload(
            adjacency_list,
            graph_edges,
            graph_edge_scales,
            graph_edge_energies,
            dangling_edges,
            cycles,
            inputs.n_vertices,
            lumen_radius_microns=inputs.lumen_radius_microns,
            microns_per_voxel=inputs.microns_per_voxel,
        )

    @classmethod
    def run_resumable(
        cls,
        edges: EdgeSet,
        vertices: VertexSet,
        params: dict[str, Any],
        stage_controller: StageController,
    ) -> NetworkResult:
        """Construct a network while persisting stage-level substeps."""
        from slavv_python.engine.state.tracker import atomic_joblib_dump

        inputs = cls._inputs_from_stages(edges, vertices, params)

        stage_controller.begin(detail="Building network graph", units_total=5, substage="adjacency")
        adjacency_path = stage_controller.artifact_path("adjacency.pkl")
        pruned_path = stage_controller.artifact_path("hair_pruned.pkl")
        cycle_path = stage_controller.artifact_path("cycle_pruned.pkl")
        strands_path = stage_controller.artifact_path("strands.pkl")

        if not adjacency_path.exists():
            adjacency_list, graph_edges, graph_edge_scales, graph_edge_energies, dangling_edges = (
                _build_graph_state(
                    inputs.edge_traces,
                    inputs.edge_scale_traces,
                    inputs.edge_energy_traces,
                    inputs.edge_connections,
                    inputs.n_vertices,
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

        if inputs.min_hair_length > 0 and not pruned_path.exists():
            _remove_short_hairs(
                graph_edges,
                adjacency_list,
                inputs.microns_per_voxel,
                inputs.min_hair_length,
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
        if inputs.remove_cycles and graph_edges and not cycle_path.exists():
            cycles = _remove_cycles(
                graph_edges,
                adjacency_list,
                inputs.n_vertices,
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
                inputs.n_vertices,
            )
            atomic_joblib_dump(topology, strands_path)
        safe_load(strands_path)
        stage_controller.update(units_total=5, units_completed=4, substage="strand_trace")

        network = _network_payload(
            adjacency_list,
            graph_edges,
            graph_edge_scales,
            graph_edge_energies,
            dangling_edges,
            cycles,
            inputs.n_vertices,
            lumen_radius_microns=inputs.lumen_radius_microns,
            microns_per_voxel=inputs.microns_per_voxel,
        )
        stage_controller.update(units_total=5, units_completed=5, substage="finalize")
        return network


__all__ = ["NetworkManager"]
