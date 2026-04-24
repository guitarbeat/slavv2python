"""Lookup helpers for edge tracing."""

from __future__ import annotations

from typing import TYPE_CHECKING

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
) -> int | None:
    """Return the index of a nearby vertex if within its physical radius."""
    tolerance_microns = 0.5 * np.mean(microns_per_voxel)

    if tree is not None:
        pos_microns = np.asarray(pos * microns_per_voxel, dtype=np.float64)
        candidates = tree.query_ball_point(pos_microns, max_search_radius)
        for i in candidates:
            vertex_pos = vertex_positions[i]
            vertex_scale = vertex_scales[i]
            radius = lumen_radius_microns[vertex_scale]
            diff = pos_microns - (vertex_pos * microns_per_voxel)
            if np.linalg.norm(diff) <= radius + tolerance_microns:
                return int(i)
        return None

    for i, (vertex_pos, vertex_scale) in enumerate(zip(vertex_positions, vertex_scales)):
        radius = lumen_radius_microns[vertex_scale]
        diff = (pos - vertex_pos) * microns_per_voxel
        if np.linalg.norm(diff) <= radius + tolerance_microns:
            return int(i)
    return None


def find_terminal_vertex(
    pos: np.ndarray,
    vertex_positions: np.ndarray,
    vertex_scales: np.ndarray,
    lumen_radius_microns: np.ndarray,
    microns_per_voxel: np.ndarray,
    tree: cKDTree | None = None,
    max_search_radius: float = 0.0,
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
    )
