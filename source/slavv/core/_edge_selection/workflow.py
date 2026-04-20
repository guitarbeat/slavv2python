"""Workflow routing and parity cleanup for edge selection."""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from .._edge_payloads import _empty_edges_result
from .cleanup import (
    clean_edges_cycles_python,
    clean_edges_orphans_python,
    clean_edges_vertex_degree_excess_python,
)
from .conflict_painting import _choose_edges_matlab_style
from .payloads import (
    build_selected_edges_result,
    initialize_edge_selection_diagnostics,
    normalize_candidate_connection_sources,
    prepare_candidate_indices_for_cleanup,
)


def _choose_edges_matlab_v200_cleanup(
    candidates: dict[str, Any],
    vertex_positions: np.ndarray,
    image_shape: tuple[int, int, int],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Choose final edges using the active MATLAB V200 cleanup chain only."""
    traces = candidates["traces"]
    connections = np.asarray(candidates["connections"], dtype=np.int32).reshape(-1, 2)
    metrics = np.asarray(candidates["metrics"], dtype=np.float32).reshape(-1)
    energy_traces = candidates["energy_traces"]
    scale_traces = candidates["scale_traces"]
    diagnostics = initialize_edge_selection_diagnostics(candidates, connections, traces)

    if len(traces) == 0:
        empty = cast("dict[str, Any]", _empty_edges_result(vertex_positions))
        empty["diagnostics"] = diagnostics
        return empty

    filtered_indices = prepare_candidate_indices_for_cleanup(
        connections,
        metrics,
        energy_traces,
        diagnostics,
    )
    if not filtered_indices:
        empty = cast("dict[str, Any]", _empty_edges_result(vertex_positions))
        empty["diagnostics"] = diagnostics
        return empty

    connection_sources = normalize_candidate_connection_sources(
        candidates.get("connection_sources"),
        len(connections),
    )

    filtered_connections = connections[filtered_indices]
    filtered_metrics = metrics[filtered_indices]
    keep_degree = clean_edges_vertex_degree_excess_python(
        filtered_connections,
        filtered_metrics,
        int(params.get("number_of_edges_per_vertex", 4)),
    )
    diagnostics["degree_pruned_count"] = int(np.sum(~keep_degree))
    after_degree_indices = [
        index for keep, index in zip(keep_degree.tolist(), filtered_indices) if keep
    ]
    if not after_degree_indices:
        empty = cast("dict[str, Any]", _empty_edges_result(vertex_positions))
        empty["diagnostics"] = diagnostics
        return empty

    keep_orphans = clean_edges_orphans_python(
        [traces[index] for index in after_degree_indices],
        image_shape,
        vertex_positions,
    )
    diagnostics["orphan_pruned_count"] = int(np.sum(~keep_orphans))
    after_orphan_indices = [
        index for keep, index in zip(keep_orphans.tolist(), after_degree_indices) if keep
    ]
    if not after_orphan_indices:
        empty = cast("dict[str, Any]", _empty_edges_result(vertex_positions))
        empty["diagnostics"] = diagnostics
        return empty

    keep_cycles = clean_edges_cycles_python(connections[after_orphan_indices])
    diagnostics["cycle_pruned_count"] = int(np.sum(~keep_cycles))
    final_indices = [
        index for keep, index in zip(keep_cycles.tolist(), after_orphan_indices) if keep
    ]
    return build_selected_edges_result(
        final_indices,
        traces,
        connections,
        metrics,
        energy_traces,
        scale_traces,
        connection_sources,
        vertex_positions,
        diagnostics,
    )


def choose_edges_for_workflow(
    candidates: dict[str, Any],
    vertex_positions: np.ndarray,
    vertex_scales: np.ndarray,
    lumen_radius_pixels_axes: np.ndarray,
    image_shape: tuple[int, int, int],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Route edge cleanup through the maintained workflow-specific chooser."""
    if bool(params.get("comparison_exact_network", False)):
        return _choose_edges_matlab_v200_cleanup(
            candidates,
            vertex_positions.astype(np.float32, copy=False),
            image_shape,
            params,
        )
    return _choose_edges_matlab_style(
        candidates,
        vertex_positions.astype(np.float32, copy=False),
        vertex_scales,
        lumen_radius_pixels_axes,
        image_shape,
        params,
    )
