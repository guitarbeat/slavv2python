"""Top-level edge-candidate generation entrypoints."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from .._edge_payloads import _empty_edge_diagnostics
from .._radius_utils import _scalar_radius
from ..edge_primitives import (
    TraceMetadata,
    _edge_metric_from_energy_trace,
    _record_trace_diagnostics,
    _trace_energy_series,
    _trace_scale_series,
)
from .global_watershed import _generate_edge_candidates_matlab_global_watershed

if TYPE_CHECKING:
    from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)


def _edge_candidates_facade() -> Any:
    """Return the facade module to preserve patchable edge-candidate hooks in tests."""
    from .. import edge_candidates as edge_candidates_facade

    return edge_candidates_facade


def _generate_fallback_directions(
    *,
    energy: np.ndarray,
    start_pos: np.ndarray,
    start_radius: float,
    microns_per_voxel: np.ndarray,
    max_edges_per_vertex: int,
    direction_method: str,
    vertex_idx: int,
) -> np.ndarray:
    """Generate the direction set for one fallback tracing origin."""
    edge_candidates_facade = _edge_candidates_facade()
    if direction_method == "hessian":
        directions = cast(
            "np.ndarray",
            edge_candidates_facade.estimate_vessel_directions(
                energy,
                start_pos,
                start_radius,
                microns_per_voxel,
            ),
        )
        if directions.shape[0] < max_edges_per_vertex:
            extra = cast(
                "np.ndarray",
                edge_candidates_facade.generate_edge_directions(
                    max_edges_per_vertex - directions.shape[0],
                    seed=vertex_idx,
                ),
            )
            return cast("np.ndarray", np.vstack([directions, extra]))
        return cast("np.ndarray", directions[:max_edges_per_vertex])
    return cast(
        "np.ndarray",
        edge_candidates_facade.generate_edge_directions(
            max_edges_per_vertex,
            seed=vertex_idx,
        ),
    )


def _finalize_matlab_parity_candidates(
    candidates: dict[str, Any],
    energy: np.ndarray,
    scale_indices: np.ndarray | None,
    vertex_positions: np.ndarray,
    energy_sign: float,
    params: dict[str, Any],
    microns_per_voxel: np.ndarray | None = None,
) -> dict[str, Any]:
    """Finalize MATLAB-parity candidates. Currently a pass-through as non-parity supplements are removed."""
    del energy, scale_indices, vertex_positions, energy_sign, params, microns_per_voxel
    return candidates


def _generate_edge_candidates_matlab_frontier(
    energy: np.ndarray,
    scale_indices: np.ndarray | None,
    vertex_positions: np.ndarray,
    vertex_scales: np.ndarray,
    lumen_radius_microns: np.ndarray,
    microns_per_voxel: np.ndarray,
    vertex_center_image: np.ndarray,
    params: dict[str, Any],
    *,
    heartbeat: Any | None = None,
) -> dict[str, Any]:
    """Generate edge candidates using MATLAB's exact global shared-state watershed search."""
    candidates = _generate_edge_candidates_matlab_global_watershed(
        energy,
        scale_indices,
        vertex_positions,
        vertex_scales,
        lumen_radius_microns,
        microns_per_voxel,
        vertex_center_image,
        params,
        heartbeat=heartbeat,
    )
    per_origin_candidate_counts = candidates["diagnostics"].get(
        "frontier_per_origin_candidate_counts",
        {},
    )
    logger.info(
        "Global watershed candidates: %d origins produced candidates, %d did not",
        len(per_origin_candidate_counts),
        len(vertex_positions) - len(per_origin_candidate_counts),
    )
    return cast("dict[str, Any]", candidates)


def _trace_fallback_origin_candidates(
    *,
    energy: np.ndarray,
    scale_indices: np.ndarray | None,
    vertex_positions: np.ndarray,
    vertex_scales: np.ndarray,
    lumen_radius_pixels: np.ndarray,
    lumen_radius_microns: np.ndarray,
    microns_per_voxel: np.ndarray,
    vertex_center_image: np.ndarray | None,
    vertex_image: np.ndarray | None,
    tree: cKDTree,
    max_search_radius: float,
    energy_sign: float,
    direction_method: str,
    max_edges_per_vertex: int,
    step_size_ratio: float,
    max_edge_energy: float,
    max_length_ratio: float,
    discrete_tracing: bool,
    energy_prepared: np.ndarray,
    mpv_prepared: np.ndarray,
    diagnostics: dict[str, Any],
    vertex_idx: int,
    start_pos: np.ndarray,
    start_scale: np.ndarray | np.generic,
) -> tuple[
    list[np.ndarray],
    list[list[int]],
    list[float],
    list[np.ndarray],
    list[np.ndarray],
    list[int],
    list[str],
]:
    """Trace all fallback candidate directions for a single origin vertex."""
    edge_candidates_facade = _edge_candidates_facade()
    start_radius = _scalar_radius(lumen_radius_pixels[start_scale])
    step_size = start_radius * step_size_ratio
    max_length = start_radius * max_length_ratio
    max_steps = max(1, int(np.ceil(max_length / max(step_size, 1e-12))))
    directions = _generate_fallback_directions(
        energy=energy,
        start_pos=start_pos,
        start_radius=start_radius,
        microns_per_voxel=microns_per_voxel,
        max_edges_per_vertex=max_edges_per_vertex,
        direction_method=direction_method,
        vertex_idx=vertex_idx,
    )

    traces: list[np.ndarray] = []
    connections: list[list[int]] = []
    metrics: list[float] = []
    energy_traces: list[np.ndarray] = []
    scale_traces: list[np.ndarray] = []
    origin_indices: list[int] = []
    connection_sources: list[str] = []

    for direction in directions:
        trace_result = edge_candidates_facade.trace_edge(
            energy_prepared,
            start_pos,
            direction,
            step_size,
            max_edge_energy,
            vertex_positions,
            vertex_scales,
            lumen_radius_pixels,
            lumen_radius_microns,
            max_steps,
            mpv_prepared,
            energy_sign,
            discrete_steps=discrete_tracing,
            vertex_center_image=vertex_center_image,
            vertex_image=vertex_image,
            tree=tree,
            max_search_radius=max_search_radius,
            origin_vertex_idx=vertex_idx,
            return_metadata=True,
        )
        edge_trace, trace_metadata = cast(
            "tuple[list[np.ndarray], TraceMetadata]",
            trace_result,
        )
        if len(edge_trace) <= 1:
            continue

        edge_arr = np.asarray(edge_trace, dtype=np.float32)
        terminal_vertex = trace_metadata["terminal_vertex"]
        energy_trace = _trace_energy_series(edge_arr, energy)
        scale_trace = _trace_scale_series(edge_arr, scale_indices)
        _record_trace_diagnostics(diagnostics, trace_metadata)

        traces.append(edge_arr)
        connections.append([vertex_idx, terminal_vertex if terminal_vertex is not None else -1])
        metrics.append(_edge_metric_from_energy_trace(energy_trace))
        energy_traces.append(energy_trace)
        scale_traces.append(scale_trace)
        origin_indices.append(vertex_idx)
        connection_sources.append("fallback")

    return (
        traces,
        connections,
        metrics,
        energy_traces,
        scale_traces,
        origin_indices,
        connection_sources,
    )


def _generate_edge_candidates(
    energy: np.ndarray,
    scale_indices: np.ndarray | None,
    vertex_positions: np.ndarray,
    vertex_scales: np.ndarray,
    lumen_radius_pixels: np.ndarray,
    lumen_radius_microns: np.ndarray,
    microns_per_voxel: np.ndarray,
    vertex_center_image: np.ndarray | None,
    vertex_image: np.ndarray | None,
    tree: cKDTree,
    max_search_radius: float,
    params: dict[str, Any],
    energy_sign: float,
) -> dict[str, Any]:
    """Generate directed edge candidates without final dedupe or degree pruning."""
    max_edges_per_vertex = params.get("number_of_edges_per_vertex", 4)
    step_size_ratio = params.get("step_size_per_origin_radius", 1.0)
    max_edge_energy = params.get("max_edge_energy", 0.0)
    max_length_ratio = params.get("max_edge_length_per_origin_radius", 60.0)
    discrete_tracing = params.get("discrete_tracing", False)
    direction_method = params.get("direction_method", "hessian")

    traces: list[np.ndarray] = []
    connections: list[list[int]] = []
    metrics: list[float] = []
    energy_traces: list[np.ndarray] = []
    scale_traces: list[np.ndarray] = []
    origin_indices: list[int] = []
    connection_sources: list[str] = []
    diagnostics = _empty_edge_diagnostics()

    energy_prepared = np.ascontiguousarray(energy, dtype=np.float64)
    mpv_prepared = np.asarray(microns_per_voxel, dtype=np.float64)

    for vertex_idx, (start_pos, start_scale) in enumerate(zip(vertex_positions, vertex_scales)):
        (
            unit_traces,
            unit_connections,
            unit_metrics,
            unit_energy_traces,
            unit_scale_traces,
            unit_origin_indices,
            unit_connection_sources,
        ) = _trace_fallback_origin_candidates(
            energy=energy,
            scale_indices=scale_indices,
            vertex_positions=vertex_positions,
            vertex_scales=vertex_scales,
            lumen_radius_pixels=lumen_radius_pixels,
            lumen_radius_microns=lumen_radius_microns,
            microns_per_voxel=microns_per_voxel,
            vertex_center_image=vertex_center_image,
            vertex_image=vertex_image,
            tree=tree,
            max_search_radius=max_search_radius,
            energy_sign=energy_sign,
            direction_method=direction_method,
            max_edges_per_vertex=max_edges_per_vertex,
            step_size_ratio=step_size_ratio,
            max_edge_energy=max_edge_energy,
            max_length_ratio=max_length_ratio,
            discrete_tracing=discrete_tracing,
            energy_prepared=energy_prepared,
            mpv_prepared=mpv_prepared,
            diagnostics=diagnostics,
            vertex_idx=vertex_idx,
            start_pos=start_pos,
            start_scale=start_scale,
        )
        traces.extend(unit_traces)
        connections.extend(unit_connections)
        metrics.extend(unit_metrics)
        energy_traces.extend(unit_energy_traces)
        scale_traces.extend(unit_scale_traces)
        origin_indices.extend(unit_origin_indices)
        connection_sources.extend(unit_connection_sources)

    candidates = {
        "traces": traces,
        "connections": np.asarray(connections, dtype=np.int32).reshape(-1, 2),
        "metrics": np.asarray(metrics, dtype=np.float32),
        "energy_traces": energy_traces,
        "scale_traces": scale_traces,
        "origin_indices": np.asarray(origin_indices, dtype=np.int32),
        "candidate_source": "fallback",
        "connection_sources": connection_sources,
        "diagnostics": diagnostics,
    }
    return candidates
