"""MATLAB port: ``calculate_linear_strel_range.m`` strel LUT for Watershed Discovery."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from slavv_python.pipeline.edges.edge_types import Int32Array


def build_matlab_local_strel_geometry(
    scale_index: int,
    lumen_radius_microns: np.ndarray,
    microns_per_voxel: np.ndarray,
    *,
    step_size_per_origin_radius: float,
) -> dict[str, np.ndarray]:
    """Port MATLAB ``calculate_linear_strel_range`` local geometry for one scale.

    The universe is realigned to [Z, X, Y] for exact parity. We preserve MATLAB's
    logical tie-breaking (smallest Y changes fastest) by making the Y dimension
    (index 2) the innermost loop.
    """
    radii_microns = float(
        np.asarray(lumen_radius_microns, dtype=np.float64).reshape(-1)[int(scale_index)]
    ) * float(step_size_per_origin_radius)
    radii_pixels = np.maximum(radii_microns / np.asarray(microns_per_voxel, dtype=np.float64), 1.0)
    # Replicate MATLAB round() which uses round-half-up
    rounded_radii = np.floor(radii_pixels + 0.5).astype(np.int32, copy=False)

    r0, r1, r2 = int(rounded_radii[0]), int(rounded_radii[1]), int(rounded_radii[2])
    px0, px1, px2 = float(radii_pixels[0]), float(radii_pixels[1]), float(radii_pixels[2])

    # Preserve MATLAB's logical tie-breaking (smallest Y changes fastest) by making the Y dimension
    # (index 0) the innermost loop. The internal grid is [Y, X, Z].
    d2_grid, d1_grid, d0_grid = np.ogrid[-r2 : r2 + 1, -r1 : r1 + 1, -r0 : r0 + 1]

    linf = np.maximum(np.maximum(np.abs(d0_grid), np.abs(d1_grid)), np.abs(d2_grid))
    radial_l2 = (d0_grid / px0) ** 2 + (d1_grid / px1) ** 2 + (d2_grid / px2) ** 2

    mask = (radial_l2 <= 1.0) | (linf <= 1)

    b_d2, b_d1, b_d0 = np.broadcast_arrays(d2_grid, d1_grid, d0_grid)
    offsets_array: Int32Array = np.column_stack([b_d0[mask], b_d1[mask], b_d2[mask]]).astype(
        np.int32
    )

    relative_distances = offsets_array.astype(np.float64, copy=False) * np.asarray(
        microns_per_voxel,
        dtype=np.float64,
    )
    distance_lut = np.sqrt(np.sum(relative_distances**2, axis=1))
    unit_vectors = np.zeros_like(relative_distances, dtype=np.float64)
    valid = distance_lut > 1e-12
    unit_vectors[valid] = relative_distances[valid] / distance_lut[valid, None]
    safe_radius = max(
        float(np.asarray(lumen_radius_microns, dtype=np.float64).reshape(-1)[int(scale_index)]),
        1e-6,
    )
    r_over_r_lut = distance_lut / safe_radius
    return {
        "local_subscripts": offsets_array,
        "distance_lut": distance_lut.astype(np.float64, copy=False),
        "unit_vectors": unit_vectors.astype(np.float64, copy=False),
        "r_over_R": r_over_r_lut.astype(np.float64, copy=False),
    }


# Parity and legacy call sites still use the underscore-prefixed name.
_build_matlab_local_strel_geometry = build_matlab_local_strel_geometry


@functools.lru_cache(maxsize=128)
def _build_matlab_global_watershed_lut_cached(
    scale_index: int,
    *,
    size_of_image: tuple[int, int, int],
    lumen_radius_microns_tuple: tuple[float, ...],
    microns_per_voxel_tuple: tuple[float, ...],
    step_size_per_origin_radius: float,
) -> dict[str, np.ndarray]:
    """Internal cached implementation of LUT generation."""
    local_geometry = build_matlab_local_strel_geometry(
        scale_index,
        np.asarray(lumen_radius_microns_tuple, dtype=np.float64),
        np.asarray(microns_per_voxel_tuple, dtype=np.float64),
        step_size_per_origin_radius=step_size_per_origin_radius,
    )
    local_subscripts = np.asarray(local_geometry["local_subscripts"], dtype=np.int32)
    pointer_indices: np.ndarray = np.arange(1, len(local_subscripts) + 1, dtype=np.uint64)

    cum_prod_image_dims = np.cumprod(np.asarray(size_of_image, dtype=np.int64))
    linear_offsets = (
        local_subscripts[:, 0].astype(np.int64, copy=False)
        + local_subscripts[:, 1].astype(np.int64, copy=False) * int(cum_prod_image_dims[0])
        + local_subscripts[:, 2].astype(np.int64, copy=False) * int(cum_prod_image_dims[1])
    )
    min_offsets = (
        (
            int(np.min(local_subscripts[:, 0])),
            int(np.min(local_subscripts[:, 1])),
            int(np.min(local_subscripts[:, 2])),
        )
        if len(local_subscripts) > 0
        else (0, 0, 0)
    )
    max_offsets = (
        (
            int(np.max(local_subscripts[:, 0])),
            int(np.max(local_subscripts[:, 1])),
            int(np.max(local_subscripts[:, 2])),
        )
        if len(local_subscripts) > 0
        else (0, 0, 0)
    )
    return {
        "linear_offsets": linear_offsets.astype(np.int64, copy=False),
        "local_subscripts": local_subscripts,
        "pointer_indices": pointer_indices,
        "distance_lut": np.asarray(local_geometry["distance_lut"], dtype=np.float64),
        "r_over_R": np.asarray(local_geometry["r_over_R"], dtype=np.float64),
        "unit_vectors": np.asarray(local_geometry["unit_vectors"], dtype=np.float64),
        "min_offsets": min_offsets,
        "max_offsets": max_offsets,
    }


_WATERSHED_LUT_FAST_CACHE: dict[tuple[Any, ...], dict[str, np.ndarray]] = {}


def clear_matlab_global_watershed_lut_cache() -> None:
    """Clear internal LUT caches."""
    _WATERSHED_LUT_FAST_CACHE.clear()
    _build_matlab_global_watershed_lut_cached.cache_clear()


def build_matlab_global_watershed_lut(
    scale_index: int,
    *,
    size_of_image: tuple[int, int, int],
    lumen_radius_microns: np.ndarray,
    microns_per_voxel: np.ndarray,
    step_size_per_origin_radius: float,
) -> dict[str, np.ndarray]:
    """Build MATLAB watershed LUT fields for one scale exactly enough for parity checks."""
    s_idx = int(scale_index)
    radii_arr = np.asarray(lumen_radius_microns, dtype=np.float64).reshape(-1)
    mpv_arr = np.asarray(microns_per_voxel, dtype=np.float64).reshape(-1)
    radius_val = float(radii_arr[s_idx])
    step_val = float(step_size_per_origin_radius)
    cache_key = (
        s_idx,
        tuple(size_of_image),
        radius_val,
        (float(mpv_arr[0]), float(mpv_arr[1]), float(mpv_arr[2])),
        step_val,
    )
    cached = _WATERSHED_LUT_FAST_CACHE.get(cache_key)
    if cached is not None:
        return cached

    result = _build_matlab_global_watershed_lut_cached(
        s_idx,
        size_of_image=size_of_image,
        lumen_radius_microns_tuple=tuple(radii_arr.tolist()),
        microns_per_voxel_tuple=tuple(mpv_arr.tolist()),
        step_size_per_origin_radius=step_val,
    )
    _WATERSHED_LUT_FAST_CACHE[cache_key] = result
    return result


_build_matlab_global_watershed_lut = build_matlab_global_watershed_lut
