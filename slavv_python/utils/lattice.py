"""Unified chunking lattice and iteration utilities."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from slavv_python.pipeline.policy import PipelinePolicy


def compute_chunking_lattice(
    image_shape: tuple[int, int, int],
    strel_size_pixels: np.ndarray,
    max_voxels_per_node: float,
    policy: PipelinePolicy,
) -> tuple[np.ndarray, int]:
    """Compute 3D chunk lattice dimensions consistently across paths."""
    target_voxels = max(max_voxels_per_node, 1.0)
    target_char_len = target_voxels ** (1.0 / 3.0)
    strel = np.asarray(strel_size_pixels, dtype=np.float64)
    aspect_ratio = strel / max(np.prod(strel) ** (1.0 / 3.0), 1e-12)
    target_dims = np.maximum(target_char_len * aspect_ratio, 1.0)

    exact_lattice = np.asarray(image_shape, dtype=np.float64) / target_dims
    lattice_dims = policy.round(np.maximum(exact_lattice, 1.0)).astype(np.int32)

    return lattice_dims, int(np.prod(lattice_dims))


def iter_chunk_slices(
    image_shape: tuple[int, int, int],
    lattice_dims: np.ndarray,
    overlap: tuple[int, int, int],
    policy: PipelinePolicy,
):
    """Yield overlapped 3D chunk slices."""
    overlap_arr = np.asarray(overlap, dtype=np.int64)

    borders = [
        policy.round(np.linspace(0, float(image_shape[axis]), int(lattice_dims[axis]) + 1)).astype(
            np.int64
        )
        for axis in range(3)
    ]

    for y_idx in range(lattice_dims[0]):
        y_start = max(int(borders[0][y_idx] - overlap_arr[0]), 0)
        y_end = min(int(borders[0][y_idx + 1] + overlap_arr[0]), image_shape[0])
        for x_idx in range(lattice_dims[1]):
            x_start = max(int(borders[1][x_idx] - overlap_arr[1]), 0)
            x_end = min(int(borders[1][x_idx + 1] + overlap_arr[1]), image_shape[1])
            for z_idx in range(lattice_dims[2]):
                z_start = max(int(borders[2][z_idx] - overlap_arr[2]), 0)
                z_end = min(int(borders[2][z_idx + 1] + overlap_arr[2]), image_shape[2])
                yield (
                    slice(y_start, y_end),
                    slice(x_start, x_end),
                    slice(z_start, z_end),
                )


__all__ = ["compute_chunking_lattice", "iter_chunk_slices"]
