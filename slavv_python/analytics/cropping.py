from __future__ import annotations

import numpy as np


def crop_vertices(
    vertices: np.ndarray, bounds: tuple[tuple[float, float], ...]
) -> tuple[np.ndarray, np.ndarray]:
    """Crop vertices to those within the specified [min, max] bounds per axis."""
    vertices_arr = np.asarray(vertices, dtype=float)
    mask = np.ones(vertices_arr.shape[0], dtype=bool)
    for i, (vmin, vmax) in enumerate(bounds):
        mask &= (vertices_arr[:, i] >= vmin) & (vertices_arr[:, i] <= vmax)
    return vertices_arr[mask], mask


def crop_edges(edges: np.ndarray, vertex_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep edges where both endpoint vertices are present in the vertex mask."""
    edges_arr = np.asarray(edges, dtype=int)
    keep = vertex_mask[edges_arr[:, 0]] & vertex_mask[edges_arr[:, 1]]
    return edges_arr[keep], keep


def crop_vertices_by_mask(
    vertices: np.ndarray, mask_volume: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Keep vertices where the corresponding floored voxel in mask_volume is True."""
    vertices_arr = np.asarray(vertices, dtype=float)
    v_int = np.floor(vertices_arr).astype(int)
    mask = np.zeros(vertices_arr.shape[0], dtype=bool)
    shape = mask_volume.shape
    for i, pos in enumerate(v_int):
        if (
            0 <= pos[0] < shape[0]
            and 0 <= pos[1] < shape[1]
            and 0 <= pos[2] < shape[2]
            and mask_volume[pos[0], pos[1], pos[2]]
        ):
            mask[i] = True
    return vertices_arr[mask], mask
