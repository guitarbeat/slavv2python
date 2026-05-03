"""Lookup and terminal-resolution helpers for edge tracing."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from scipy.spatial import cKDTree


def in_bounds(pos: np.ndarray, shape: tuple[int, ...]) -> bool:
    """Check if the floored position lies within array bounds."""
    if len(shape) == 3:
        res_3d: bool = 0 <= pos[0] < shape[0] and 0 <= pos[1] < shape[1] and 0 <= pos[2] < shape[2]
        return res_3d

    pos_int = np.floor(pos).astype(int)
    res: bool = np.all((pos_int >= 0) & (pos_int < np.array(shape)))  # type: ignore[assignment]
    return res


def vertex_at_position(pos: np.ndarray, vertex_image: np.ndarray) -> int | None:
    """Return the 0-indexed vertex id at ``pos`` if present."""
    pos_int = np.floor(pos).astype(int)
    if not np.all((pos_int >= 0) & (pos_int < np.array(vertex_image.shape))):
        return None

    vertex_id = vertex_image[pos_int[0], pos_int[1], pos_int[2]]
    return int(vertex_id - 1) if vertex_id > 0 else None


def near_vertex(
        pos: np.ndarray,
        vertex_positions: np.ndarray,
        vertex_scales: np.ndarray,
        lumen_radius_microns: np.ndarray,
        microns_per_voxel: np.ndarray,
        tree: cKDTree | None = None,
        max_search_radius: float = 0.0,
        exclude_vertex: int | None = None,
) -> int | None:
    """Return the index of a nearby vertex if within its physical radius."""
    tolerance_microns = 0.5 * np.mean(microns_per_voxel)

    if tree is not None:
        pos_microns = np.asarray(pos * microns_per_voxel, dtype=np.float64)
        candidates = tree.query_ball_point(pos_microns, max_search_radius)
        ranked_candidates: list[tuple[float, int]] = []
        for i in candidates:
            if exclude_vertex is not None and int(i) == exclude_vertex:
                continue
            vertex_pos = vertex_positions[i]
            vertex_scale = vertex_scales[i]
            radius = lumen_radius_microns[vertex_scale]
            diff = pos_microns - (vertex_pos * microns_per_voxel)
            distance = float(np.linalg.norm(diff))
            if distance <= radius + tolerance_microns:
                ranked_candidates.append((distance, int(i)))
        if ranked_candidates:
            ranked_candidates.sort(key=lambda item: item[0])
            return int(ranked_candidates[0][1])
        return None

    ranked_candidates = []
    for i, (vertex_pos, vertex_scale) in enumerate(zip(vertex_positions, vertex_scales)):
        if exclude_vertex is not None and i == exclude_vertex:
            continue
        radius = lumen_radius_microns[vertex_scale]
        diff = (pos - vertex_pos) * microns_per_voxel
        distance = float(np.linalg.norm(diff))
        if distance <= radius + tolerance_microns:
            ranked_candidates.append((distance, int(i)))
    if ranked_candidates:
        ranked_candidates.sort(key=lambda item: item[0])
        return int(ranked_candidates[0][1])
    return None


def find_terminal_vertex(
        pos: np.ndarray,
        vertex_positions: np.ndarray,
        vertex_scales: np.ndarray,
        lumen_radius_microns: np.ndarray,
        microns_per_voxel: np.ndarray,
        tree: cKDTree | None = None,
        max_search_radius: float = 0.0,
        exclude_vertex: int | None = None,
) -> int | None:
    """Find the index of a terminal vertex near a given position, if any."""
    return near_vertex(
        pos,
        vertex_positions,
        vertex_scales,
        lumen_radius_microns,
        microns_per_voxel,
        tree=tree,
        max_search_radius=max_search_radius,
        exclude_vertex=exclude_vertex,
    )


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
        vertex_image: np.ndarray | None = None,
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

    if vertex_image is not None:
        terminal_vertex = vertex_at_position(trace_array[-1], vertex_image)
        if terminal_vertex is not None and terminal_vertex != origin_vertex:
            return int(terminal_vertex), "reverse_volume_hit"

        for point in trace_array[-2::-1]:
            terminal_vertex = vertex_at_position(point, vertex_image)
            if terminal_vertex is not None and terminal_vertex != origin_vertex:
                return int(terminal_vertex), "reverse_volume_hit"

    for point in trace_array[::-1]:
        terminal_vertex = near_vertex(
            point,
            vertex_positions,
            vertex_scales,
            lumen_radius_microns,
            microns_per_voxel,
            tree=tree,
            max_search_radius=max_search_radius,
            exclude_vertex=origin_vertex,
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
        vertex_image: np.ndarray | None = None,
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
        vertex_image=vertex_image,
    )

    if terminal_vertex is not None:
        final_trace.append(np.asarray(vertex_positions[terminal_vertex], dtype=np.float32).copy())

    return final_trace, {
        "stop_reason": stop_reason,
        "terminal_vertex": terminal_vertex,
        "terminal_resolution": terminal_resolution,
    }


__all__ = [
    "_finalize_traced_edge",
    "_resolve_trace_terminal_vertex",
    "find_terminal_vertex",
    "in_bounds",
    "near_vertex",
    "vertex_at_position",
]
