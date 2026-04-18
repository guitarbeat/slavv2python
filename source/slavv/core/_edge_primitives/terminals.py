"""Trace terminal resolution helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .lookup import near_vertex, vertex_at_position

if TYPE_CHECKING:
    from scipy.spatial import cKDTree


def _resolve_trace_terminal_vertex(
    edge_trace: list[np.ndarray] | np.ndarray,
    vertex_center_image: np.ndarray | None,
    vertex_positions: np.ndarray,
    vertex_scales: np.ndarray,
    lumen_radius_microns: np.ndarray,
    microns_per_voxel: np.ndarray,
    origin_vertex: int,
    tree: cKDTree | None = None,
    max_search_radius: float = 0.0,
    direct_terminal_vertex: int | None = None,
) -> tuple[int | None, str | None]:
    """Resolve a terminal vertex using MATLAB-style center hits plus tolerant fallback."""
    trace_array = np.asarray(edge_trace, dtype=np.float32).reshape(-1, 3)

    if direct_terminal_vertex is not None and direct_terminal_vertex != origin_vertex:
        return int(direct_terminal_vertex), "direct_hit"

    if len(trace_array) == 0:
        return None, None

    if vertex_center_image is not None:
        terminal_vertex = vertex_at_position(trace_array[-1], vertex_center_image)
        if terminal_vertex is not None and terminal_vertex != origin_vertex:
            return int(terminal_vertex), "direct_hit"

        for point in trace_array[-2::-1]:
            terminal_vertex = vertex_at_position(point, vertex_center_image)
            if terminal_vertex is not None and terminal_vertex != origin_vertex:
                return int(terminal_vertex), "reverse_center_hit"

    for point in trace_array[::-1]:
        terminal_vertex = near_vertex(
            point,
            vertex_positions,
            vertex_scales,
            lumen_radius_microns,
            microns_per_voxel,
            tree=tree,
            max_search_radius=max_search_radius,
        )
        if terminal_vertex is not None and terminal_vertex != origin_vertex:
            return int(terminal_vertex), "reverse_near_hit"

    return None, None


def _finalize_traced_edge(
    edge_trace: list[np.ndarray] | np.ndarray,
    *,
    stop_reason: str,
    direct_terminal_vertex: int | None,
    vertex_center_image: np.ndarray | None,
    vertex_positions: np.ndarray,
    vertex_scales: np.ndarray,
    lumen_radius_microns: np.ndarray,
    microns_per_voxel: np.ndarray,
    origin_vertex: int,
    tree: cKDTree | None = None,
    max_search_radius: float = 0.0,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Finalize a raw trace by resolving its terminal vertex and normalizing metadata."""
    trace_array = np.asarray(edge_trace, dtype=np.float32).reshape(-1, 3)
    final_trace = [point.copy() for point in trace_array]
    terminal_vertex, terminal_resolution = _resolve_trace_terminal_vertex(
        trace_array,
        vertex_center_image,
        vertex_positions,
        vertex_scales,
        lumen_radius_microns,
        microns_per_voxel,
        origin_vertex,
        tree=tree,
        max_search_radius=max_search_radius,
        direct_terminal_vertex=direct_terminal_vertex,
    )

    if terminal_vertex is not None:
        final_trace.append(np.asarray(vertex_positions[terminal_vertex], dtype=np.float32).copy())

    return final_trace, {
        "stop_reason": stop_reason,
        "terminal_vertex": terminal_vertex,
        "terminal_resolution": terminal_resolution,
    }
