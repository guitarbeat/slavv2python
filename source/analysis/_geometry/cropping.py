from __future__ import annotations

from typing import cast

import numpy as np


def crop_vertices(
    vertex_positions: np.ndarray,
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
) -> tuple[np.ndarray, np.ndarray]:
    """Crop vertices to an axis-aligned bounding box."""
    vertex_positions_arr = np.asarray(vertex_positions)
    bounds_arr = np.asarray(bounds, dtype=float)

    mask = (
        (vertex_positions_arr[:, 0] >= bounds_arr[0, 0])
        & (vertex_positions_arr[:, 0] <= bounds_arr[0, 1])
        & (vertex_positions_arr[:, 1] >= bounds_arr[1, 0])
        & (vertex_positions_arr[:, 1] <= bounds_arr[1, 1])
        & (vertex_positions_arr[:, 2] >= bounds_arr[2, 0])
        & (vertex_positions_arr[:, 2] <= bounds_arr[2, 1])
    )
    return cast("np.ndarray", vertex_positions_arr[mask]), mask


def crop_edges(edge_indices: np.ndarray, vertex_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Remove edges whose endpoints are not both retained in ``vertex_mask``."""
    edge_indices_arr = np.asarray(edge_indices)
    vertex_mask_arr = np.asarray(vertex_mask, dtype=bool)
    keep = vertex_mask_arr[edge_indices_arr[:, 0]] & vertex_mask_arr[edge_indices_arr[:, 1]]
    return cast("np.ndarray", edge_indices_arr[keep]), keep


def crop_vertices_by_mask(
    vertex_positions: np.ndarray,
    mask_volume: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Crop vertices by a 3D binary mask."""
    vertex_positions_arr = np.asarray(vertex_positions)
    mask_volume_arr = np.asarray(mask_volume, dtype=bool)

    coords = np.floor(vertex_positions_arr).astype(int)
    in_bounds = (
        (coords[:, 0] >= 0)
        & (coords[:, 0] < mask_volume_arr.shape[0])
        & (coords[:, 1] >= 0)
        & (coords[:, 1] < mask_volume_arr.shape[1])
        & (coords[:, 2] >= 0)
        & (coords[:, 2] < mask_volume_arr.shape[2])
    )

    mask: np.ndarray = np.zeros(len(vertex_positions_arr), dtype=bool)
    valid_indices = np.where(in_bounds)[0]
    valid_coords = coords[in_bounds]
    mask[valid_indices] = mask_volume_arr[
        valid_coords[:, 0],
        valid_coords[:, 1],
        valid_coords[:, 2],
    ]
    return cast("np.ndarray", vertex_positions_arr[mask]), mask
