"""
Vertex and Edge tracing logic for SLAVV.
Handles vertex extraction (local maxima/minima) and edge tracing through the energy field.
"""

from __future__ import annotations

import logging
import math
from heapq import heappop, heappush
from typing import TYPE_CHECKING, Any, cast

import joblib
import numpy as np
import scipy.ndimage as ndi
from scipy.spatial import cKDTree
from skimage import feature  # Needed for Hessian
from skimage.draw import ellipsoid
from skimage.segmentation import watershed

# Imports from sibling modules
from .energy import compute_gradient_impl

if TYPE_CHECKING:
    from pathlib import Path

    from slavv.runtime import StageController

logger = logging.getLogger(__name__)


def _vertex_window_apothem(space_strel_apothem: int) -> int:
    """Normalize the MATLAB vertex neighborhood radius in voxel units."""
    return max(int(space_strel_apothem), 0)


def _vertex_neighborhood_slices(
    pos: np.ndarray, apothem: int, shape: tuple[int, int, int]
) -> tuple[slice, slice, slice]:
    """Return clipped cube slices centered on ``pos``."""
    y, x, z = (int(coord) for coord in pos)
    res_v: tuple[slice, slice, slice] = (
        slice(max(0, y - apothem), min(shape[0], y + apothem + 1)),
        slice(max(0, x - apothem), min(shape[1], x + apothem + 1)),
        slice(max(0, z - apothem), min(shape[2], z + apothem + 1)),
    )
    return res_v


def _matlab_linear_indices(coords: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    """Return MATLAB-style column-major linear indices for 0-based coordinates."""
    coords = np.asarray(coords, dtype=np.int64)
    return coords[:, 0] + coords[:, 1] * shape[0] + coords[:, 2] * shape[0] * shape[1]  # type: ignore[no-any-return]


def _sort_vertex_order(
    vertex_positions: np.ndarray,
    vertex_energies: np.ndarray,
    image_shape: tuple[int, int, int],
    energy_sign: float,
) -> np.ndarray:
    """Sort vertices like MATLAB: by energy, then by column-major linear index for ties."""
    if len(vertex_positions) == 0:
        return np.array([], dtype=np.int64)  # type: ignore[no-any-return]

    linear_indices = _matlab_linear_indices(vertex_positions, image_shape)
    if energy_sign < 0:
        return np.lexsort((linear_indices, vertex_energies))  # type: ignore[no-any-return]
    return np.lexsort((linear_indices, -vertex_energies))  # type: ignore[no-any-return]


def _empty_vertices_result() -> dict[str, Any]:
    """Return the canonical empty vertex payload."""
    return {
        "positions": np.empty((0, 3), dtype=np.float32),
        "scales": np.empty((0,), dtype=np.int16),
        "energies": np.empty((0,), dtype=np.float32),
        "radii_pixels": np.empty((0,), dtype=np.float32),
        "radii_microns": np.empty((0,), dtype=np.float32),
        "radii": np.empty((0,), dtype=np.float32),
    }


def _empty_stop_reason_counts() -> dict[str, int]:
    """Return the canonical edge-trace stop-reason counter payload."""
    return {
        "bounds": 0,
        "nan": 0,
        "energy_threshold": 0,
        "energy_rise_step_halving": 0,
        "max_steps": 0,
        "direct_terminal_hit": 0,
        "frontier_exhausted_nonnegative": 0,
        "length_limit": 0,
        "terminal_frontier_hit": 0,
    }


def _empty_edge_diagnostics() -> dict[str, Any]:
    """Return the canonical edge-diagnostics payload."""
    return {
        "candidate_traced_edge_count": 0,
        "terminal_edge_count": 0,
        "self_edge_count": 0,
        "duplicate_directed_pair_count": 0,
        "antiparallel_pair_count": 0,
        "chosen_edge_count": 0,
        "dangling_edge_count": 0,
        "negative_energy_rejected_count": 0,
        "conflict_rejected_count": 0,
        "degree_pruned_count": 0,
        "orphan_pruned_count": 0,
        "cycle_pruned_count": 0,
        "watershed_join_supplement_count": 0,
        "watershed_endpoint_degree_rejected": 0,
        "terminal_direct_hit_count": 0,
        "terminal_reverse_center_hit_count": 0,
        "terminal_reverse_near_hit_count": 0,
        "stop_reason_counts": _empty_stop_reason_counts(),
    }


def _merge_edge_diagnostics(target: dict[str, Any], source: dict[str, Any]) -> None:
    """Merge additive edge diagnostics from one payload into another."""
    for key, value in source.items():
        if key == "stop_reason_counts":
            target_counts = target.setdefault("stop_reason_counts", _empty_stop_reason_counts())
            for stop_reason, count in value.items():
                target_counts[stop_reason] = int(target_counts.get(stop_reason, 0)) + int(count)
            continue

        if isinstance(value, dict):
            target_map = target.setdefault(key, {})
            if not isinstance(target_map, dict):
                target_map = {}
            for item_key, item_value in value.items():
                target_map[str(item_key)] = int(target_map.get(str(item_key), 0)) + int(item_value)
            target[key] = target_map
            continue

        if isinstance(value, (int, np.integer)):
            target[key] = int(target.get(key, 0)) + int(value)


def _empty_edges_result(vertex_positions: np.ndarray | None = None) -> dict[str, Any]:
    """Return the canonical empty edge payload."""
    positions = (
        np.asarray(vertex_positions, dtype=np.float32)
        if vertex_positions is not None
        else np.empty((0, 3), dtype=np.float32)
    )
    return {
        "traces": [],
        "connections": np.zeros((0, 2), dtype=np.int32),
        "energies": np.zeros((0,), dtype=np.float32),
        "energy_traces": [],
        "scale_traces": [],
        "connection_sources": [],
        "vertex_positions": positions,
        "diagnostics": _empty_edge_diagnostics(),
        "chosen_candidate_indices": np.zeros((0,), dtype=np.int32),
    }


def _coerce_radius_axes(
    lumen_radius_pixels: np.ndarray,
    lumen_radius_pixels_axes: np.ndarray | None,
) -> np.ndarray:
    """Normalize scale radii into a `(num_scales, 3)` axis-aware array."""
    if lumen_radius_pixels_axes is not None:
        axes = np.asarray(lumen_radius_pixels_axes, dtype=np.float32)
        if axes.ndim == 2 and axes.shape[1] == 3:
            return axes  # type: ignore[no-any-return]

    radii = np.asarray(lumen_radius_pixels, dtype=np.float32).reshape(-1, 1)
    return np.repeat(radii, 3, axis=1)  # type: ignore[no-any-return]


def _scalar_radius(radius_value: np.ndarray | float | int) -> float:
    """Convert isotropic or axis-aware radii into a single tracing radius."""
    radius_array = np.asarray(radius_value, dtype=np.float32).reshape(-1)
    if radius_array.size == 0:
        return 0.0
    if radius_array.size == 1:
        return float(radius_array[0])
    return float(np.cbrt(np.prod(radius_array)))


def _chunk_lattice_dimensions(
    image_shape: tuple[int, int, int],
    strel_size_pixels: np.ndarray,
    max_voxels_per_node: float,
) -> tuple[int, int, int]:
    """Approximate MATLAB's 3D chunk lattice sizing."""
    target_voxels = max(float(max_voxels_per_node), 1.0)
    target_char_len = target_voxels ** (1.0 / 3.0)
    strel = np.asarray(strel_size_pixels, dtype=np.float64)
    aspect_ratio = strel / max(np.prod(strel) ** (1.0 / 3.0), 1e-12)
    target_dims = np.maximum(target_char_len * aspect_ratio, 1.0)
    lattice = np.maximum(np.rint(np.asarray(image_shape, dtype=np.float64) / target_dims), 1)
    val = lattice.tolist()
    return (int(val[0]), int(val[1]), int(val[2]))


def _iter_overlapping_chunks(
    image_shape: tuple[int, int, int],
    lattice_dims: tuple[int, int, int],
    overlap: tuple[int, int, int],
):
    """Yield padded 3D chunk slices using MATLAB-like lattice borders."""
    overlap = np.asarray(overlap, dtype=np.int64)
    borders = [
        np.rint(np.linspace(0, image_shape[axis], lattice_dims[axis] + 1)).astype(np.int64)
        for axis in range(3)
    ]

    for y_index in range(lattice_dims[0]):
        y_start = max(int(borders[0][y_index] - overlap[0]), 0)
        y_end = min(int(borders[0][y_index + 1] + overlap[0]), image_shape[0])
        for x_index in range(lattice_dims[1]):
            x_start = max(int(borders[1][x_index] - overlap[1]), 0)
            x_end = min(int(borders[1][x_index + 1] + overlap[1]), image_shape[1])
            for z_index in range(lattice_dims[2]):
                z_start = max(int(borders[2][z_index] - overlap[2]), 0)
                z_end = min(int(borders[2][z_index + 1] + overlap[2]), image_shape[2])
                yield (
                    slice(y_start, y_end),
                    slice(x_start, x_end),
                    slice(z_start, z_end),
                )


def _matlab_vertex_candidates_in_chunk(
    energy: np.ndarray,
    scale_indices: np.ndarray,
    energy_sign: float,
    energy_upper_bound: float,
    space_strel_apothem: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run MATLAB-style candidate scanning within one overlapped chunk."""
    apothem = _vertex_window_apothem(space_strel_apothem)
    interior_mask = np.zeros(energy.shape, dtype=bool)
    if apothem == 0:
        interior_mask[:] = True
    else:
        interior_mask[
            apothem : energy.shape[0] - apothem,
            apothem : energy.shape[1] - apothem,
            apothem : energy.shape[2] - apothem,
        ] = True

    finite_mask = np.isfinite(energy)
    if energy_sign < 0:
        active_mask = interior_mask & finite_mask & (energy < energy_upper_bound)
    else:
        active_mask = interior_mask & finite_mask & (energy > energy_upper_bound)

    candidate_positions = np.argwhere(active_mask)
    if len(candidate_positions) == 0:
        return (
            np.empty((0, 3), dtype=np.int32),
            np.empty((0,), dtype=np.int16),
            np.empty((0,), dtype=np.float32),
        )

    candidate_energies = energy[active_mask]
    order = _sort_vertex_order(candidate_positions, candidate_energies, energy.shape, energy_sign)
    candidate_positions = candidate_positions[order]

    accepted_positions: list[np.ndarray] = []
    accepted_scales: list[int] = []
    accepted_energies: list[float] = []

    for pos in candidate_positions:
        y, x, z = (int(coord) for coord in pos)
        if not active_mask[y, x, z]:
            continue

        slices = _vertex_neighborhood_slices(pos, apothem, energy.shape)
        window = energy[slices]
        if energy_sign < 0:
            window_extreme: float = np.nanmin(window)
            is_vertex = np.isfinite(window_extreme) and energy[y, x, z] <= window_extreme
        else:
            window_extreme = np.nanmax(window)
            is_vertex = np.isfinite(window_extreme) and energy[y, x, z] >= window_extreme

        if is_vertex:
            accepted_positions.append(pos.astype(np.int32, copy=False))
            accepted_scales.append(int(scale_indices[y, x, z]))
            accepted_energies.append(float(energy[y, x, z]))

        active_mask[slices] = False

    return (
        np.asarray(accepted_positions, dtype=np.int32).reshape(-1, 3),
        np.asarray(accepted_scales, dtype=np.int16),
        np.asarray(accepted_energies, dtype=np.float32),
    )


def _matlab_vertex_candidates(
    energy: np.ndarray,
    scale_indices: np.ndarray,
    energy_sign: float,
    energy_upper_bound: float,
    space_strel_apothem: int,
    strel_size_pixels: np.ndarray,
    max_voxels_per_node: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find MATLAB-style candidate vertices on the projected energy volume."""
    apothem = _vertex_window_apothem(space_strel_apothem)
    overlap = (apothem, apothem, apothem)
    lattice_dims = _chunk_lattice_dimensions(energy.shape, strel_size_pixels, max_voxels_per_node)

    accepted_positions: list[np.ndarray] = []
    accepted_scales: list[np.ndarray] = []
    accepted_energies: list[np.ndarray] = []
    for chunk_slice in _iter_overlapping_chunks(energy.shape, lattice_dims, overlap):
        chunk_positions, chunk_scales, chunk_energies = _matlab_vertex_candidates_in_chunk(
            energy[chunk_slice],
            scale_indices[chunk_slice],
            energy_sign,
            energy_upper_bound,
            space_strel_apothem,
        )
        if len(chunk_positions) == 0:
            continue

        chunk_offset = np.array(
            [chunk_slice[0].start, chunk_slice[1].start, chunk_slice[2].start],
            dtype=np.int32,
        )
        accepted_positions.append(chunk_positions + chunk_offset)
        accepted_scales.append(chunk_scales)
        accepted_energies.append(chunk_energies)

    if not accepted_positions:
        return (
            np.empty((0, 3), dtype=np.int32),
            np.empty((0,), dtype=np.int16),
            np.empty((0,), dtype=np.float32),
        )

    return (
        np.vstack(accepted_positions).astype(np.int32, copy=False),
        np.concatenate(accepted_scales).astype(np.int16, copy=False),
        np.concatenate(accepted_energies).astype(np.float32, copy=False),
    )


def _crop_vertices_matlab_style(
    vertex_positions: np.ndarray,
    vertex_scales: np.ndarray,
    vertex_energies: np.ndarray,
    image_shape: tuple[int, int, int],
    lumen_radius_pixels_axes: np.ndarray,
    length_dilation_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Crop MATLAB candidate vertices against image bounds and extreme scales."""
    if len(vertex_positions) == 0:
        return (
            np.empty((0, 3), dtype=np.int32),
            np.empty((0,), dtype=np.int16),
            np.empty((0,), dtype=np.float32),
        )

    scale_indices = np.rint(vertex_scales).astype(np.int64)
    scaled_radii = np.rint(length_dilation_ratio * lumen_radius_pixels_axes[scale_indices]).astype(
        np.int64
    )
    positions = np.rint(vertex_positions).astype(np.int64)

    mins = positions - scaled_radii
    maxs = positions + scaled_radii
    scale_is_min = scale_indices <= 0
    scale_is_max = scale_indices >= (len(lumen_radius_pixels_axes) - 1)
    excluded = (
        (mins[:, 0] < 0)
        | (mins[:, 1] < 0)
        | (mins[:, 2] < 0)
        | (maxs[:, 0] >= image_shape[0])
        | (maxs[:, 1] >= image_shape[1])
        | (maxs[:, 2] >= image_shape[2])
        | scale_is_min
        | scale_is_max
    )
    keep = ~excluded
    return (
        vertex_positions[keep].astype(np.int32, copy=False),
        vertex_scales[keep].astype(np.int16, copy=False),
        vertex_energies[keep].astype(np.float32, copy=False),
    )


def _ellipsoid_offsets(radii_pixels: np.ndarray) -> np.ndarray:
    """Construct centered voxel offsets for a scale-specific ellipsoid."""
    radii = np.maximum(np.rint(radii_pixels).astype(int), 0)
    if np.all(radii == 0):
        return np.zeros((1, 3), dtype=np.int16)

    mask = ellipsoid(float(radii[0]), float(radii[1]), float(radii[2]), spacing=(1.0, 1.0, 1.0))
    coords = np.column_stack(np.where(mask))
    center = np.asarray(mask.shape, dtype=np.int64) // 2
    offsets: np.ndarray = (coords - center).astype(np.int16, copy=False)
    return offsets


def _choose_vertices_matlab_style(
    vertex_positions: np.ndarray,
    vertex_scales: np.ndarray,
    image_shape: tuple[int, int, int],
    lumen_radius_pixels_axes: np.ndarray,
    length_dilation_ratio: float,
    start_index: int = 0,
    end_index: int | None = None,
    chosen_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Choose non-overlapping vertices with MATLAB's paint-and-check semantics."""
    n_vertices = len(vertex_positions)
    if chosen_mask is None:
        chosen_mask = np.zeros(n_vertices, dtype=bool)
    else:
        chosen_mask = np.asarray(chosen_mask, dtype=bool).copy()

    if end_index is None:
        end_index = n_vertices

    scale_indices = np.rint(vertex_scales).astype(np.int64)
    scaled_radii = length_dilation_ratio * lumen_radius_pixels_axes
    template_cache = {
        int(scale_index): _ellipsoid_offsets(scaled_radii[scale_index])
        for scale_index in np.unique(scale_indices)
    }
    painted_image: np.ndarray = np.zeros(image_shape, dtype=bool)

    def paint(index: int) -> np.ndarray:
        center = np.rint(vertex_positions[index]).astype(np.int64)
        coords = template_cache[int(scale_indices[index])].astype(np.int64) + center
        valid = (
            (coords[:, 0] >= 0)
            & (coords[:, 0] < image_shape[0])
            & (coords[:, 1] >= 0)
            & (coords[:, 1] < image_shape[1])
            & (coords[:, 2] >= 0)
            & (coords[:, 2] < image_shape[2])
        )
        res: np.ndarray = coords[valid]
        return res

    for index in np.flatnonzero(chosen_mask[:start_index]):
        coords = paint(int(index))
        painted_image[coords[:, 0], coords[:, 1], coords[:, 2]] = True

    for index in range(start_index, min(end_index, n_vertices)):
        coords = paint(index)
        if coords.size == 0:
            chosen_mask[index] = False
            continue
        occupied = painted_image[coords[:, 0], coords[:, 1], coords[:, 2]].any()
        chosen_mask[index] = not occupied
        if chosen_mask[index]:
            painted_image[coords[:, 0], coords[:, 1], coords[:, 2]] = True

    return chosen_mask


def in_bounds(pos: np.ndarray, shape: tuple[int, ...]) -> bool:
    """Check if the floored position lies within array bounds."""
    # Optimization for 3D case which is the bottleneck in tracing
    if len(shape) == 3:
        res_3d: bool = 0 <= pos[0] < shape[0] and 0 <= pos[1] < shape[1] and 0 <= pos[2] < shape[2]
        return res_3d

    pos_int = np.floor(pos).astype(int)
    res: bool = np.all((pos_int >= 0) & (pos_int < np.array(shape)))  # type: ignore[assignment]
    return res


def compute_gradient(
    energy: np.ndarray, pos: np.ndarray, microns_per_voxel: np.ndarray
) -> np.ndarray:
    """Compute gradient at ``pos`` using central differences (wrapper for implementation)."""
    pos_int = np.round(pos).astype(np.int64)
    # Ensure proper dtypes for Numba compatibility (if enabled in impl)
    energy_arr = np.ascontiguousarray(energy, dtype=np.float64)
    mpv_arr = np.asarray(microns_per_voxel, dtype=np.float64)
    res: np.ndarray = compute_gradient_impl(energy_arr, pos_int, mpv_arr)
    return res


def generate_edge_directions(n_directions: int, seed: int | None = None) -> np.ndarray:
    """Generate uniformly distributed unit vectors on the sphere.

    Parameters
    ----------
    n_directions : int
        Number of direction vectors to generate.
    seed : int, optional
        Random seed for reproducibility. If None, uses unseeded RNG.
    """
    if n_directions <= 0:
        res_empty: np.ndarray = np.empty((0, 3), dtype=np.float64)
        return res_empty
    if n_directions == 1:
        res_single: np.ndarray = np.array([[0, 0, 1]], dtype=np.float64)
        return res_single

    # Generate random points from a 3D standard normal distribution
    rng = np.random.default_rng(seed)
    points = rng.standard_normal((n_directions, 3))
    # Normalize to unit vectors
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    res_arr: np.ndarray = (points / norms).astype(np.float64)
    return res_arr


def paint_vertex_image(
    vertex_positions: np.ndarray,  # Shape: (N, 3) - [y, x, z] positions
    vertex_scales: np.ndarray,  # Shape: (N,) - scale indices
    lumen_radius_pixels: np.ndarray,  # Shape: (M, 3) - [ry, rx, rz] per scale
    image_shape: tuple[int, int, int],  # (height, width, depth)
) -> np.ndarray:
    """
    Create a painted vertex-volume image (1-indexed, 0=background).

    This paints ellipsoidal occupancy regions around each vertex and is used for geometric
    overlap semantics rather than terminal center detection.

    Parameters
    ----------
    vertex_positions : np.ndarray
        Vertex positions as (y, x, z) coordinates
    vertex_scales : np.ndarray
        Scale index for each vertex
    lumen_radius_pixels : np.ndarray
        Radii for each scale in pixels [ry, rx, rz]
    image_shape : tuple
        Shape of the output volume (height, width, depth)

    Returns
    -------
    vertex_image : np.ndarray
        Volume where each voxel contains vertex index (1-indexed) or 0 for background
    """
    vertex_image = np.zeros(image_shape, dtype=np.uint16)  # Supports up to 65k vertices

    for i, (pos, scale) in enumerate(zip(vertex_positions, vertex_scales)):
        # Get ellipsoid radii for this vertex's scale
        radii = lumen_radius_pixels[scale]  # [ry, rx, rz]

        # Generate ellipsoid mask using skimage
        try:
            # ellipsoid returns a 3D boolean array
            ellipsoid_mask = ellipsoid(radii[0], radii[1], radii[2], spacing=(1.0, 1.0, 1.0))
            # Get coordinates of True voxels (centered at origin of mask array)
            coords = np.where(ellipsoid_mask)
            # Center the ellipsoid coordinates (they're currently offset from origin)
            center = np.array(ellipsoid_mask.shape) // 2
            rr = coords[0] - center[0]
            cc = coords[1] - center[1]
            dd = coords[2] - center[2]

            # Offset to vertex position (convert to int)
            y_coords = rr + int(np.round(pos[0]))
            x_coords = cc + int(np.round(pos[1]))
            z_coords = dd + int(np.round(pos[2]))

            # Clip to image bounds
            valid_mask = (
                (y_coords >= 0)
                & (y_coords < image_shape[0])
                & (x_coords >= 0)
                & (x_coords < image_shape[1])
                & (z_coords >= 0)
                & (z_coords < image_shape[2])
            )

            y_coords = y_coords[valid_mask]
            x_coords = x_coords[valid_mask]
            z_coords = z_coords[valid_mask]

            # Paint vertex index (1-indexed, so i+1)
            vertex_image[y_coords, x_coords, z_coords] = i + 1

        except Exception as e:
            logger.warning(f"Failed to paint vertex {i} at {pos} with scale {scale}: {e}")
            continue

    logger.info(f"Painted {len(vertex_positions)} vertices into volume image")
    res_v: np.ndarray = vertex_image
    return res_v


def paint_vertex_center_image(
    vertex_positions: np.ndarray,
    image_shape: tuple[int, int, int],
) -> np.ndarray:
    """Create a sparse image containing only vertex center identities."""
    center_image: np.ndarray = np.zeros(image_shape, dtype=np.uint16)
    if len(vertex_positions) == 0:
        return center_image

    coords = np.rint(np.asarray(vertex_positions, dtype=np.float32)[:, :3]).astype(np.int32)
    coords[:, 0] = np.clip(coords[:, 0], 0, image_shape[0] - 1)
    coords[:, 1] = np.clip(coords[:, 1], 0, image_shape[1] - 1)
    coords[:, 2] = np.clip(coords[:, 2], 0, image_shape[2] - 1)
    center_image[coords[:, 0], coords[:, 1], coords[:, 2]] = np.arange(
        1,
        len(coords) + 1,
        dtype=np.uint16,
    )
    res_ci: np.ndarray = center_image
    return res_ci


def extract_vertices(energy_data: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    """
    Extract vertices as local extrema in the energy field.
    MATLAB Equivalent: `get_vertices_V200.m`
    """
    logger.info("Extracting vertices")

    energy = energy_data["energy"]
    scale_indices = energy_data["scale_indices"]
    lumen_radius_pixels = energy_data["lumen_radius_pixels"]
    lumen_radius_pixels_axes = _coerce_radius_axes(
        lumen_radius_pixels,
        energy_data.get("lumen_radius_pixels_axes"),
    )
    energy_sign = energy_data.get("energy_sign", -1.0)
    lumen_radius_microns = energy_data["lumen_radius_microns"]

    # Parameters
    energy_upper_bound = params.get("energy_upper_bound", 0.0)
    space_strel_apothem = params.get("space_strel_apothem", 1)
    length_dilation_ratio = params.get("length_dilation_ratio", 1.0)
    max_voxels_per_node = params.get("max_voxels_per_node", 6000)
    vertex_positions, vertex_scales, vertex_energies = _matlab_vertex_candidates(
        energy,
        scale_indices,
        energy_sign,
        energy_upper_bound,
        space_strel_apothem,
        lumen_radius_pixels_axes[0],
        max_voxels_per_node,
    )

    vertex_positions, vertex_scales, vertex_energies = _crop_vertices_matlab_style(
        vertex_positions,
        vertex_scales,
        vertex_energies,
        energy.shape,
        lumen_radius_pixels_axes,
        length_dilation_ratio,
    )

    if len(vertex_positions) == 0:
        logger.info("Extracted 0 vertices")
        return _empty_vertices_result()

    sort_indices = _sort_vertex_order(vertex_positions, vertex_energies, energy.shape, energy_sign)
    vertex_positions = vertex_positions[sort_indices]
    vertex_scales = vertex_scales[sort_indices]
    vertex_energies = vertex_energies[sort_indices]

    chosen_mask = _choose_vertices_matlab_style(
        vertex_positions,
        vertex_scales,
        energy.shape,
        lumen_radius_pixels_axes,
        length_dilation_ratio,
    )
    vertex_positions = vertex_positions[chosen_mask]
    vertex_scales = vertex_scales[chosen_mask]
    vertex_energies = vertex_energies[chosen_mask]

    logger.info(f"Extracted {len(vertex_positions)} vertices")

    # Standardize output dtypes
    vertex_positions = vertex_positions.astype(np.float32)
    vertex_scales = vertex_scales.astype(np.int16)
    vertex_energies = vertex_energies.astype(np.float32)
    radii_pixels = lumen_radius_pixels[vertex_scales].astype(np.float32)
    radii_microns = lumen_radius_microns[vertex_scales].astype(np.float32)

    return {
        "positions": vertex_positions,
        "scales": vertex_scales,
        "energies": vertex_energies,
        "radii_pixels": radii_pixels,
        "radii_microns": radii_microns,
        "radii": radii_microns,
    }


def vertex_at_position(pos: np.ndarray, vertex_image: np.ndarray) -> int | None:
    """
    Fast O(1) vertex lookup using pre-computed vertex volume image.

    Parameters
    ----------
    pos : np.ndarray
        Position in voxel coordinates [y, x, z]
    vertex_image : np.ndarray
        Volume where each voxel contains vertex index (1-indexed) or 0

    Returns
    -------
    vertex_idx : Optional[int]
        Vertex index (0-indexed) if position is within a vertex region, None otherwise
    """
    pos_int = np.floor(pos).astype(int)

    # Check bounds
    if not np.all((pos_int >= 0) & (pos_int < np.array(vertex_image.shape))):
        return None

    vertex_id = vertex_image[pos_int[0], pos_int[1], pos_int[2]]

    if vertex_id > 0:
        return int(vertex_id - 1)  # Convert from 1-indexed to 0-indexed
    return None


def near_vertex(
    pos: np.ndarray,
    vertex_positions: np.ndarray,
    vertex_scales: np.ndarray,
    lumen_radius_microns: np.ndarray,
    microns_per_voxel: np.ndarray,
    tree: cKDTree | None = None,
    max_search_radius: float = 0.0,
) -> int | None:
    """Return the index of a nearby vertex if within its physical radius; otherwise None

    Uses a tolerance of 0.5 voxels to account for traces ending near but not exactly at vertices.
    """
    # Tolerance: 0.5 voxels in physical units (use average voxel size)
    tolerance_microns = 0.5 * np.mean(microns_per_voxel)

    if tree is not None:
        # Optimized spatial query
        pos_microns = pos * microns_per_voxel
        # Query candidates within max possible radius
        candidates = tree.query_ball_point(pos_microns, max_search_radius)
        for i in candidates:
            # Check specific radius for this candidate
            vertex_pos = vertex_positions[i]
            vertex_scale = vertex_scales[i]
            radius = lumen_radius_microns[vertex_scale]
            diff = pos_microns - (vertex_pos * microns_per_voxel)
            if np.linalg.norm(diff) <= radius + tolerance_microns:
                return i
        return None
    # Fallback linear scan
    for i, (vertex_pos, vertex_scale) in enumerate(zip(vertex_positions, vertex_scales)):
        radius = lumen_radius_microns[vertex_scale]
        diff = (pos - vertex_pos) * microns_per_voxel
        if np.linalg.norm(diff) <= radius + tolerance_microns:
            return i
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


def _clip_trace_indices(trace: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    """Convert a trace to clipped integer voxel indices."""
    coords = np.floor(np.asarray(trace, dtype=np.float32)[:, :3]).astype(np.int32, copy=False)
    for axis in range(3):
        coords[:, axis] = np.clip(coords[:, axis], 0, shape[axis] - 1)
    return coords


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


def _record_trace_diagnostics(
    diagnostics: dict[str, Any],
    trace_metadata: dict[str, Any],
) -> None:
    """Accumulate per-trace terminal-resolution and stop-reason diagnostics."""
    stop_reason = trace_metadata.get("stop_reason")
    if stop_reason:
        stop_reason_counts = diagnostics.setdefault(
            "stop_reason_counts", _empty_stop_reason_counts()
        )
        stop_reason_counts[stop_reason] = int(stop_reason_counts.get(stop_reason, 0)) + 1

    terminal_resolution = trace_metadata.get("terminal_resolution")
    if terminal_resolution == "direct_hit":
        diagnostics["terminal_direct_hit_count"] += 1
    elif terminal_resolution == "reverse_center_hit":
        diagnostics["terminal_reverse_center_hit_count"] += 1
    elif terminal_resolution == "reverse_near_hit":
        diagnostics["terminal_reverse_near_hit_count"] += 1


def _trace_scale_series(edge_trace: np.ndarray, scale_indices: np.ndarray | None) -> np.ndarray:
    """Sample projected scale indices along an edge trace."""
    if scale_indices is None:
        return np.zeros((len(edge_trace),), dtype=np.int16)
    idx = _clip_trace_indices(edge_trace, scale_indices.shape)
    return scale_indices[idx[:, 0], idx[:, 1], idx[:, 2]].astype(np.int16, copy=False)


def _trace_energy_series(edge_trace: np.ndarray, energy: np.ndarray) -> np.ndarray:
    """Sample projected energy values along an edge trace."""
    idx = _clip_trace_indices(edge_trace, energy.shape)
    return energy[idx[:, 0], idx[:, 1], idx[:, 2]].astype(np.float32, copy=False)


def _edge_metric_from_energy_trace(energy_trace: np.ndarray) -> float:
    """Match MATLAB's current edge quality metric: minimum max-energy is best."""
    arr = np.asarray(energy_trace, dtype=np.float32)
    if arr.size == 0:
        return 0.0
    value = float(np.nanmax(arr))
    if math.isnan(value):
        return -1000.0
    return value


def _use_matlab_frontier_tracer(energy_data: dict[str, Any], params: dict[str, Any]) -> bool:
    """Enable the parity-specific frontier tracer only for MATLAB-energy parity runs."""
    if not bool(params.get("comparison_exact_network", False)):
        return False
    return energy_data.get("energy_origin") == "matlab_batch_hdf5"


def _matlab_parity_edge_number_tolerance(params: dict[str, Any]) -> int:
    """Return the MATLAB-style source fanout cap for parity frontier tracing."""
    requested_edges = int(params.get("number_of_edges_per_vertex", 4))
    edge_number_tolerance = int(params.get("parity_edge_number_tolerance", 2))
    return max(1, min(requested_edges, edge_number_tolerance))


def _matlab_frontier_offsets(
    strel_apothem: int,
    microns_per_voxel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct MATLAB-style cube-neighborhood offsets with Y-fastest ordering."""
    local_range = np.arange(-strel_apothem, strel_apothem + 1, dtype=np.int32)
    offsets = np.array(
        [[y, x, z] for z in local_range for x in local_range for y in local_range],
        dtype=np.int32,
    )
    distances = np.sqrt(np.sum((offsets.astype(np.float64) * microns_per_voxel) ** 2, axis=1))
    return offsets, distances.astype(np.float32, copy=False)


def _coord_to_matlab_linear_index(coord: np.ndarray, shape: tuple[int, int, int]) -> int:
    """Convert a 0-based ``(y, x, z)`` coordinate into MATLAB linear order."""
    y, x, z = (int(value) for value in coord[:3])
    return int(y + x * shape[0] + z * shape[0] * shape[1])


def _matlab_linear_index_to_coord(index: int, shape: tuple[int, int, int]) -> np.ndarray:
    """Convert a 0-based MATLAB linear index into a ``(y, x, z)`` coordinate."""
    xy_plane = shape[0] * shape[1]
    z = index // xy_plane
    pos_xy = index - z * xy_plane
    x = pos_xy // shape[0]
    y = pos_xy - x * shape[0]
    return np.array([y, x, z], dtype=np.int32)


def _path_coords_from_linear_indices(
    path_linear: list[int],
    shape: tuple[int, int, int],
) -> np.ndarray:
    """Convert a linear-index path into origin-to-terminal spatial coordinates."""
    coords = [_matlab_linear_index_to_coord(index, shape) for index in reversed(path_linear)]
    return np.asarray(coords, dtype=np.float32)


def _path_max_energy_from_linear_indices(
    path_linear: list[int],
    energy: np.ndarray,
    shape: tuple[int, int, int],
) -> float:
    """Return the maximum sampled energy along a linear-index path."""
    if not path_linear:
        return float("-inf")
    samples = []
    for index in path_linear:
        coord = _matlab_linear_index_to_coord(index, shape)
        samples.append(float(energy[coord[0], coord[1], coord[2]]))
    return max(samples) if samples else float("-inf")


def _candidate_endpoint_pair_set(connections: np.ndarray) -> set[tuple[int, int]]:
    """Return the orientation-independent terminal endpoint pairs in a candidate payload."""
    pairs: set[tuple[int, int]] = set()
    connections = np.asarray(connections, dtype=np.int32).reshape(-1, 2)
    for start_vertex, end_vertex in connections:
        if int(start_vertex) < 0 or int(end_vertex) < 0:
            continue
        u, v = int(start_vertex), int(end_vertex)
        pair: tuple[int, int] = (u, v) if u < v else (v, u)
        pairs.add(pair)
    return pairs


def _rasterize_trace_segment(
    start: np.ndarray,
    end: np.ndarray,
    image_shape: tuple[int, int, int],
) -> np.ndarray:
    """Rasterize a straight voxel segment between two points, preserving endpoints."""
    start_coord = np.rint(np.asarray(start, dtype=np.float32)[:3]).astype(np.int32, copy=False)
    end_coord = np.rint(np.asarray(end, dtype=np.float32)[:3]).astype(np.int32, copy=False)
    max_coord = np.asarray(image_shape, dtype=np.int32) - 1
    start_coord = np.clip(start_coord, 0, max_coord)
    end_coord = np.clip(end_coord, 0, max_coord)

    steps = int(np.max(np.abs(end_coord - start_coord)))
    if steps <= 0:
        return start_coord.reshape(1, 3).astype(np.float32, copy=False)

    coords = np.rint(np.linspace(start_coord, end_coord, num=steps + 1)).astype(np.int32)
    deduped = [coords[0]]
    for coord in coords[1:]:
        if not np.array_equal(coord, deduped[-1]):
            deduped.append(coord)
    return np.asarray(deduped, dtype=np.float32)


def _build_watershed_join_trace(
    start: np.ndarray,
    contact: np.ndarray,
    end: np.ndarray,
    image_shape: tuple[int, int, int],
) -> np.ndarray:
    """Construct a simple ordered trace that joins two vertices through a watershed contact."""
    start_half = _rasterize_trace_segment(start, contact, image_shape)
    end_half = _rasterize_trace_segment(contact, end, image_shape)
    if len(end_half) > 0 and len(start_half) > 0 and np.array_equal(start_half[-1], end_half[0]):
        end_half = end_half[1:]
    if len(end_half) == 0:
        return start_half
    return np.vstack([start_half, end_half]).astype(np.float32, copy=False)


def _best_watershed_contact_coords(
    labels: np.ndarray,
    energy: np.ndarray,
) -> dict[tuple[int, int], np.ndarray]:
    """Return the lowest-energy face contact voxel for each touching watershed pair."""
    best_contacts: dict[tuple[int, int], tuple[float, np.ndarray]] = {}
    shifts = (
        np.array((1, 0, 0), dtype=np.int32),
        np.array((0, 1, 0), dtype=np.int32),
        np.array((0, 0, 1), dtype=np.int32),
    )

    for shift in shifts:
        source_slices = tuple(slice(None, -int(delta)) if delta else slice(None) for delta in shift)
        target_slices = tuple(slice(int(delta), None) if delta else slice(None) for delta in shift)
        source_labels = labels[source_slices]
        target_labels = labels[target_slices]
        is_touching = (source_labels != target_labels) & (source_labels > 0) & (target_labels > 0)
        if not np.any(is_touching):
            continue

        source_coords = np.argwhere(is_touching).astype(np.int32, copy=False)
        target_coords = source_coords + shift
        source_pairs = source_labels[is_touching].astype(np.int32, copy=False) - 1
        target_pairs = target_labels[is_touching].astype(np.int32, copy=False) - 1
        pair_indices = np.stack([source_pairs, target_pairs], axis=1)
        pair_indices.sort(axis=1)

        source_energy = energy[source_slices][is_touching]
        target_energy = energy[target_slices][is_touching]
        prefer_target = target_energy < source_energy
        contact_coords = source_coords.copy()
        contact_coords[prefer_target] = target_coords[prefer_target]
        contact_energy = np.where(prefer_target, target_energy, source_energy).astype(
            np.float32,
            copy=False,
        )

        order = np.lexsort((contact_energy, pair_indices[:, 1], pair_indices[:, 0]))
        pair_indices = pair_indices[order]
        contact_coords = contact_coords[order]
        contact_energy = contact_energy[order]
        keep: np.ndarray = np.ones((len(pair_indices),), dtype=bool)
        keep[1:] = np.any(pair_indices[1:] != pair_indices[:-1], axis=1)

        for pair_array, coord, pair_energy in zip(
            pair_indices[keep],
            contact_coords[keep],
            contact_energy[keep],
        ):
            pair = (int(pair_array[0]), int(pair_array[1]))
            best = best_contacts.get(pair)
            if best is None or float(pair_energy) < best[0]:
                best_contacts[pair] = (float(pair_energy), coord.astype(np.int32, copy=False))

    return {pair: coord for pair, (_, coord) in best_contacts.items()}


def _supplement_matlab_frontier_candidates_with_watershed_joins(
    candidates: dict[str, Any],
    energy: np.ndarray,
    scale_indices: np.ndarray | None,
    vertex_positions: np.ndarray,
    energy_sign: float,
    max_edges_per_vertex: int = 4,
    enforce_frontier_reachability: bool = True,
    require_mutual_frontier_participation: bool = False,
) -> dict[str, Any]:
    """Add parity-only watershed contact candidates that the local frontier misses.

    Phase 2 gates (applied in order):
    1. Already-existing pair skip
    2. Short-trace rejection (trace length <= 1)
    3. Non-negative energy rejection (max energy along trace >= 0)
    4. Frontier reachability: at least one vertex in the pair must already
       have a frontier candidate to *any* vertex.
    5. Optional mutual frontier participation: both vertices in the pair
       must already participate in frontier candidates.
    6. Per-origin supplement cap: each seed origin can contribute at most
       ``max_edges_per_vertex`` supplement candidates.
    """
    if len(vertex_positions) < 2:
        return candidates

    image_shape = energy.shape
    markers = np.zeros(image_shape, dtype=np.int32)
    idxs = np.floor(vertex_positions).astype(int)
    idxs = np.clip(idxs, 0, np.array(image_shape) - 1)
    markers[idxs[:, 0], idxs[:, 1], idxs[:, 2]] = np.arange(1, len(vertex_positions) + 1)
    labels = watershed(-energy_sign * energy, markers)

    existing_pairs = _candidate_endpoint_pair_set(candidates.get("connections", np.zeros((0, 2))))
    contact_coords_by_pair = _best_watershed_contact_coords(labels, energy)
    endpoint_pair_degree_counts: dict[int, int] = {}
    for start_vertex, end_vertex in existing_pairs:
        endpoint_pair_degree_counts[int(start_vertex)] = (
            endpoint_pair_degree_counts.get(int(start_vertex), 0) + 1
        )
        endpoint_pair_degree_counts[int(end_vertex)] = (
            endpoint_pair_degree_counts.get(int(end_vertex), 0) + 1
        )

    # Phase 2: build frontier reachability set — vertices that participate in
    # at least one frontier candidate (before supplementation)
    frontier_vertices: set[int] = set()
    existing_connections = np.asarray(
        candidates.get("connections", np.zeros((0, 2))), dtype=np.int32
    ).reshape(-1, 2)
    for start_vertex, end_vertex in existing_connections:
        if int(start_vertex) >= 0:
            frontier_vertices.add(int(start_vertex))
        if int(end_vertex) >= 0:
            frontier_vertices.add(int(end_vertex))

    supplement_payload: dict[str, Any] = {
        "candidate_source": "watershed",
        "traces": [],
        "connections": [],
        "metrics": [],
        "energy_traces": [],
        "scale_traces": [],
        "origin_indices": [],
        "connection_sources": [],
        "diagnostics": {
            "watershed_join_supplement_count": 0,
            "watershed_per_origin_candidate_counts": {},
        },
    }

    # Diagnostic counters
    n_already_existing = 0
    n_short_trace = 0
    n_energy_rejected = 0
    n_reachability_rejected = 0
    n_mutual_frontier_rejected = 0
    n_cap_rejected = 0
    n_endpoint_degree_rejected = 0
    n_accepted = 0
    n_total_watershed_pairs = len(contact_coords_by_pair)

    # Phase 2: per-origin supplement cap tracking
    origin_supplement_counts: dict[int, int] = {}

    for pair, contact_coord in sorted(contact_coords_by_pair.items()):
        if pair in existing_pairs:
            n_already_existing += 1
            continue

        # Phase 2 gate: frontier reachability — at least one endpoint must have
        # participated in a frontier candidate
        if (
            enforce_frontier_reachability
            and pair[0] not in frontier_vertices
            and pair[1] not in frontier_vertices
        ):
            n_reachability_rejected += 1
            continue

        if (
            enforce_frontier_reachability
            and require_mutual_frontier_participation
            and (pair[0] not in frontier_vertices or pair[1] not in frontier_vertices)
        ):
            n_mutual_frontier_rejected += 1
            continue

        if (
            endpoint_pair_degree_counts.get(pair[0], 0) >= max_edges_per_vertex
            or endpoint_pair_degree_counts.get(pair[1], 0) >= max_edges_per_vertex
        ):
            n_endpoint_degree_rejected += 1
            continue

        # Phase 2 gate: per-origin supplement cap
        seed_origin = pair[0]
        current_origin_count = origin_supplement_counts.get(seed_origin, 0)
        if current_origin_count >= max_edges_per_vertex:
            n_cap_rejected += 1
            continue

        trace = _build_watershed_join_trace(
            vertex_positions[pair[0]],
            contact_coord,
            vertex_positions[pair[1]],
            image_shape,
        )
        if len(trace) <= 1:
            n_short_trace += 1
            continue

        energy_trace = _trace_energy_series(trace, energy)
        if float(np.nanmax(np.asarray(energy_trace, dtype=np.float32))) >= 0:
            n_energy_rejected += 1
            continue

        scale_trace = _trace_scale_series(trace, scale_indices)
        supplement_payload["traces"].append(trace)
        supplement_payload["connections"].append([pair[0], pair[1]])
        supplement_payload["metrics"].append(_edge_metric_from_energy_trace(energy_trace))
        supplement_payload["energy_traces"].append(energy_trace)
        supplement_payload["scale_traces"].append(scale_trace)
        supplement_payload["origin_indices"].append(pair[0])
        supplement_payload["connection_sources"].append("watershed")
        supplement_payload["diagnostics"]["watershed_join_supplement_count"] += 1
        n_accepted += 1
        origin_supplement_counts[seed_origin] = current_origin_count + 1
        existing_pairs.add(pair)
        endpoint_pair_degree_counts[pair[0]] = endpoint_pair_degree_counts.get(pair[0], 0) + 1
        endpoint_pair_degree_counts[pair[1]] = endpoint_pair_degree_counts.get(pair[1], 0) + 1
        supplement_payload["diagnostics"]["watershed_per_origin_candidate_counts"][
            str(seed_origin)
        ] = int(origin_supplement_counts.get(seed_origin, 0))

    supplement_payload["diagnostics"]["watershed_total_pairs"] = n_total_watershed_pairs
    supplement_payload["diagnostics"]["watershed_already_existing"] = n_already_existing
    supplement_payload["diagnostics"]["watershed_short_trace_rejected"] = n_short_trace
    supplement_payload["diagnostics"]["watershed_energy_rejected"] = n_energy_rejected
    supplement_payload["diagnostics"]["watershed_reachability_rejected"] = n_reachability_rejected
    supplement_payload["diagnostics"]["watershed_mutual_frontier_rejected"] = (
        n_mutual_frontier_rejected
    )
    supplement_payload["diagnostics"]["watershed_cap_rejected"] = n_cap_rejected
    supplement_payload["diagnostics"]["watershed_endpoint_degree_rejected"] = (
        n_endpoint_degree_rejected
    )
    supplement_payload["diagnostics"]["watershed_accepted"] = n_accepted

    logger.info(
        "Watershed supplement: %d total pairs, %d already existing, "
        "%d reachability rejected, %d mutual-frontier rejected, "
        "%d endpoint-degree rejected, %d cap rejected, "
        "%d short-trace rejected, %d energy rejected, %d accepted",
        n_total_watershed_pairs,
        n_already_existing,
        n_reachability_rejected,
        n_mutual_frontier_rejected,
        n_endpoint_degree_rejected,
        n_cap_rejected,
        n_short_trace,
        n_energy_rejected,
        n_accepted,
    )

    if supplement_payload["connections"]:
        _append_candidate_unit(candidates, supplement_payload)
    else:
        # Still merge the diagnostic counters even when no candidates were added
        _merge_edge_diagnostics(
            candidates.get("diagnostics", {}), supplement_payload["diagnostics"]
        )
    return candidates


def _prune_frontier_indices_beyond_found_vertices(
    candidate_coords: np.ndarray,
    origin_position_microns: np.ndarray,
    displacement_vectors: list[np.ndarray],
    microns_per_voxel: np.ndarray,
) -> np.ndarray:
    """Remove frontier voxels that lie beyond an already-found terminal direction."""
    if len(candidate_coords) == 0 or not displacement_vectors:
        return candidate_coords

    vectors_from_origin = (
        candidate_coords.astype(np.float64) * microns_per_voxel - origin_position_microns
    )
    indices_beyond: np.ndarray = np.zeros((len(candidate_coords),), dtype=bool)
    for displacement in displacement_vectors:
        # MATLAB parity note: The threshold of 1.0 matches the normalized
        # dot-product gate in get_edges_for_vertex.m. 'displacement' has been
        # pre-normalized by ||d||^2 (not ||d||), so the dot product gives
        # (cos(theta) * ||v_from_origin||) / ||d|| which exceeds 1.0 when the
        # candidate voxel lies at least one ||d|| beyond the found vertex in
        # the same direction. This prevents the frontier from exploring beyond
        # already-resolved terminal vertices.
        indices_beyond |= np.sum(displacement * vectors_from_origin, axis=1) > 1.0
    return candidate_coords[~indices_beyond]


def _resolve_frontier_edge_connection(
    current_path_linear: list[int],
    terminal_vertex_idx: int,
    seed_origin_idx: int,
    edge_paths_linear: list[list[int]],
    edge_pairs: list[tuple[int, int]],
    pointer_index_map: dict[int, int],
    energy: np.ndarray,
    shape: tuple[int, int, int],
) -> tuple[int | None, int | None]:
    """Resolve MATLAB-style parent/child validity for a frontier-found terminal."""
    root_index = current_path_linear[-1]
    root_pointer = int(pointer_index_map.get(root_index, 0))
    parent_index = -root_pointer if root_pointer < 0 else 0

    if parent_index == 0:
        return seed_origin_idx, terminal_vertex_idx

    parent_path = edge_paths_linear[parent_index - 1]
    parent_pointers = {
        -int(pointer_index_map.get(index, 0))
        for index in parent_path
        if int(pointer_index_map.get(index, 0)) < 0
    }
    parent_pointers.discard(0)
    parent_pointers.discard(parent_index)
    if parent_pointers:
        return None, None

    parent_terminal, parent_origin = edge_pairs[parent_index - 1]
    if parent_terminal < 0 or parent_origin < 0:
        return None, None

    parent_energy = _path_max_energy_from_linear_indices(parent_path, energy, shape)
    child_energy = _path_max_energy_from_linear_indices(current_path_linear, energy, shape)
    # MATLAB parity note: This mirrors get_edges_for_vertex.m's "child is better
    # than parent" rejection. In MATLAB, when a child path has a better (lower,
    # more negative) energy than its parent, the child is considered to be
    # stealing the parent's best voxels, so the child is invalidated.
    # The strict <= comparison preserves MATLAB's exact behavior.
    if child_energy <= parent_energy:
        return None, None

    if root_index not in parent_path:
        return None, None

    bifurcation_index = parent_path.index(root_index)
    parent_1 = parent_path[:bifurcation_index]
    parent_2 = parent_path[bifurcation_index + 1 :]
    half_candidates: list[tuple[int, float]] = [
        (
            parent_terminal,
            _path_max_energy_from_linear_indices(parent_1, energy, shape)
            if parent_1
            else float("-inf"),
        )
    ]
    if parent_2:
        half_candidates.append(
            (parent_origin, _path_max_energy_from_linear_indices(parent_2, energy, shape))
        )
    origin_vertex_idx = min(half_candidates, key=lambda item: item[1])[0]
    if origin_vertex_idx < 0:
        return None, None
    return origin_vertex_idx, terminal_vertex_idx


def _trace_origin_edges_matlab_frontier(
    energy: np.ndarray,
    scale_indices: np.ndarray | None,
    vertex_positions: np.ndarray,
    vertex_scales: np.ndarray,
    lumen_radius_microns: np.ndarray,
    microns_per_voxel: np.ndarray,
    vertex_center_image: np.ndarray,
    origin_vertex_idx: int,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Trace a single origin using a MATLAB-style best-first voxel frontier."""
    shape = energy.shape
    # MATLAB's watershed frontier seeds at most ``edge_number_tolerance``
    # directions from a source vertex. Clamp the parity helper to that same
    # fanout so imported-MATLAB runs do not over-generate per-origin edges.
    max_edges_per_vertex = _matlab_parity_edge_number_tolerance(params)
    max_length_ratio = float(params.get("max_edge_length_per_origin_radius", 60.0))
    strel_apothem = int(
        params.get(
            "space_strel_apothem_edges",
            params.get(
                "space_strel_apothem", max(1, round(params.get("step_size_per_origin_radius", 1.0)))
            ),
        )
    )
    offsets, offset_distances = _matlab_frontier_offsets(strel_apothem, microns_per_voxel)
    origin_coord = np.rint(vertex_positions[origin_vertex_idx]).astype(np.int32)
    origin_coord[0] = np.clip(origin_coord[0], 0, shape[0] - 1)
    origin_coord[1] = np.clip(origin_coord[1], 0, shape[1] - 1)
    origin_coord[2] = np.clip(origin_coord[2], 0, shape[2] - 1)
    origin_linear = _coord_to_matlab_linear_index(origin_coord, shape)
    origin_position_microns = origin_coord.astype(np.float64) * microns_per_voxel
    origin_scale = int(vertex_scales[origin_vertex_idx])
    origin_radius_microns = float(lumen_radius_microns[origin_scale])
    max_edge_length_microns = max_length_ratio * origin_radius_microns
    max_edge_length_voxels = int(np.round(max_edge_length_microns / np.min(microns_per_voxel))) + 1
    max_number_of_indices = max(1, max_edge_length_voxels * max_edges_per_vertex)

    diagnostics = _empty_edge_diagnostics()
    traces: list[np.ndarray] = []
    connections: list[list[int]] = []
    metrics: list[float] = []
    energy_traces: list[np.ndarray] = []
    scale_traces: list[np.ndarray] = []
    origin_indices: list[int] = []
    edge_paths_linear: list[list[int]] = []
    edge_pairs: list[tuple[int, int]] = []
    displacement_vectors: list[np.ndarray] = []
    has_valid_terminal_edge = False
    previous_indices_visited: list[int] = []
    pointer_index_map: dict[int, int] = {origin_linear: 0}
    pointer_energy_map: dict[int, float] = {}
    # MATLAB seeds the origin distance map at 1 before any expansion. Matching
    # that off-by-one budget keeps the frontier's max-length cutoff aligned with
    # get_edges_for_vertex.m.
    distance_map: dict[int, float] = {origin_linear: 1.0}
    available_map: dict[int, float] = {}
    available_heap: list[tuple[float, int]] = []

    # MATLAB does not even enter the edge-search loop when the origin lies too
    # close to the border for the current structuring element.
    if np.any(origin_coord < strel_apothem) or np.any(
        origin_coord >= (np.asarray(shape, dtype=np.int32) - strel_apothem)
    ):
        diagnostics["stop_reason_counts"]["bounds"] += 1
        return {
            "origin_index": origin_vertex_idx,
            "candidate_source": "frontier",
            "traces": traces,
            "connections": connections,
            "metrics": metrics,
            "energy_traces": energy_traces,
            "scale_traces": scale_traces,
            "origin_indices": [origin_vertex_idx] * len(traces),
            "connection_sources": ["frontier"] * len(traces),
            "diagnostics": diagnostics,
        }

    current_linear = origin_linear

    while (
        len(edge_paths_linear) < max_edges_per_vertex
        and len(previous_indices_visited) < max_number_of_indices
    ):
        current_coord = _matlab_linear_index_to_coord(current_linear, shape)
        current_energy = float(energy[current_coord[0], current_coord[1], current_coord[2]])
        terminal_vertex_idx = (
            int(vertex_center_image[current_coord[0], current_coord[1], current_coord[2]]) - 1
        )
        if terminal_vertex_idx == origin_vertex_idx:
            terminal_vertex_idx = -1

        previous_indices_visited.append(current_linear)
        current_visit_order = len(previous_indices_visited)
        pointer_energy_map[current_linear] = float("-inf")

        neighbor_coords = current_coord + offsets
        valid_mask = (
            (neighbor_coords[:, 0] >= 0)
            & (neighbor_coords[:, 0] < shape[0])
            & (neighbor_coords[:, 1] >= 0)
            & (neighbor_coords[:, 1] < shape[1])
            & (neighbor_coords[:, 2] >= 0)
            & (neighbor_coords[:, 2] < shape[2])
        )
        neighbor_coords = neighbor_coords[valid_mask]
        neighbor_distances = offset_distances[valid_mask]
        new_coords: list[np.ndarray] = []
        new_distances: list[float] = []
        for coord, distance in zip(neighbor_coords, neighbor_distances):
            linear_index = _coord_to_matlab_linear_index(coord, shape)
            if pointer_energy_map.get(linear_index, 0.0) > current_energy:
                pointer_index_map[linear_index] = current_visit_order
                pointer_energy_map[linear_index] = current_energy
                distance_map[linear_index] = distance_map[current_linear] + float(distance)
                new_coords.append(coord.astype(np.int32, copy=False))
                new_distances.append(float(distance_map[linear_index]))

        if terminal_vertex_idx >= 0:
            diagnostics["stop_reason_counts"]["terminal_frontier_hit"] += 1
            path_linear = [current_linear]
            tracing_linear = current_linear
            while int(pointer_index_map.get(tracing_linear, 0)) > 0:
                tracing_linear = previous_indices_visited[
                    int(pointer_index_map[tracing_linear]) - 1
                ]
                path_linear.append(tracing_linear)
            for path_index in path_linear[:-1]:
                pointer_index_map[path_index] = -(len(edge_paths_linear) + 1)

            origin_idx, terminal_idx = _resolve_frontier_edge_connection(
                path_linear,
                terminal_vertex_idx,
                origin_vertex_idx,
                edge_paths_linear,
                edge_pairs,
                pointer_index_map,
                energy,
                shape,
            )

            edge_paths_linear.append(path_linear)
            edge_pairs.append(
                (
                    int(terminal_idx) if terminal_idx is not None else -1,
                    int(origin_idx) if origin_idx is not None else -1,
                )
            )

            current_position = current_coord.astype(np.float64) * microns_per_voxel
            displacement = current_position - origin_position_microns
            displacement_norm_sq = float(np.sum(displacement**2))
            if displacement_norm_sq > 0:
                displacement_vectors.append(displacement / displacement_norm_sq)

            if origin_idx is not None and terminal_idx is not None:
                has_valid_terminal_edge = True
                edge_trace = _path_coords_from_linear_indices(path_linear, shape)
                energy_trace = _trace_energy_series(edge_trace, energy)
                scale_trace = _trace_scale_series(edge_trace, scale_indices)
                traces.append(edge_trace)
                connections.append([int(origin_idx), int(terminal_idx)])
                metrics.append(_edge_metric_from_energy_trace(energy_trace))
                energy_traces.append(energy_trace)
                scale_traces.append(scale_trace)
                origin_indices.append(origin_vertex_idx)
                diagnostics["terminal_direct_hit_count"] += 1
        else:
            if new_coords:
                new_coords_array = np.asarray(new_coords, dtype=np.int32)
                new_distances_array = np.asarray(new_distances, dtype=np.float32)
                diagnostics["stop_reason_counts"]["length_limit"] += int(
                    np.sum(new_distances_array >= max_edge_length_microns)
                )
                within_length = new_distances_array < max_edge_length_microns
                new_coords_array = new_coords_array[within_length]
                if len(new_coords_array) and has_valid_terminal_edge:
                    new_coords_array = _prune_frontier_indices_beyond_found_vertices(
                        new_coords_array,
                        origin_position_microns,
                        displacement_vectors,
                        microns_per_voxel,
                    )
                for coord in new_coords_array:
                    linear_index = _coord_to_matlab_linear_index(coord, shape)
                    available_energy = float(energy[coord[0], coord[1], coord[2]])
                    available_map[linear_index] = available_energy
                    # MATLAB parity note: the tiebreaker for equal-energy
                    # frontier voxels is the linear index, which corresponds to
                    # MATLAB's column-major order. This matches get_edges_by_
                    # watershed.m's implicit ordering from the priority queue.
                    heappush(available_heap, (available_energy, linear_index))

        available_map.pop(current_linear, None)
        next_current = None
        stopped_on_nonnegative = False
        while available_heap:
            candidate_energy, candidate_linear = heappop(available_heap)
            if available_map.get(candidate_linear) != candidate_energy:
                continue
            if candidate_energy >= 0:
                available_map.pop(candidate_linear, None)
                diagnostics["stop_reason_counts"]["frontier_exhausted_nonnegative"] += 1
                available_heap.clear()
                stopped_on_nonnegative = True
                next_current = None
                break
            next_current = int(candidate_linear)
            break

        if next_current is None:
            if not available_map and not stopped_on_nonnegative:
                diagnostics["stop_reason_counts"]["frontier_exhausted_nonnegative"] += 1
            break

        current_linear = next_current

    return {
        "origin_index": origin_vertex_idx,
        "candidate_source": "frontier",
        "traces": traces,
        "connections": connections,
        "metrics": metrics,
        "energy_traces": energy_traces,
        "scale_traces": scale_traces,
        "origin_indices": [origin_vertex_idx] * len(traces),
        "connection_sources": ["frontier"] * len(traces),
        "diagnostics": diagnostics,
    }


def _append_candidate_unit(target: dict[str, Any], unit_payload: dict[str, Any]) -> None:
    """Append a per-origin candidate payload into the aggregate candidate manifest."""
    unit_traces = [np.asarray(trace, dtype=np.float32) for trace in unit_payload["traces"]]
    unit_connections = np.asarray(unit_payload["connections"], dtype=np.int32).reshape(-1, 2)
    unit_metrics = np.asarray(unit_payload["metrics"], dtype=np.float32).reshape(-1)
    unit_origin_indices = np.asarray(
        unit_payload.get("origin_indices", []), dtype=np.int32
    ).reshape(-1)
    unit_connection_sources = _normalize_candidate_connection_sources(
        unit_payload.get("connection_sources"),
        len(unit_connections),
        default_source=str(unit_payload.get("candidate_source", "unknown")),
    )

    target["traces"].extend(unit_traces)
    target["energy_traces"].extend(
        np.asarray(trace, dtype=np.float32) for trace in unit_payload["energy_traces"]
    )
    target["scale_traces"].extend(
        np.asarray(trace, dtype=np.int16) for trace in unit_payload["scale_traces"]
    )

    if unit_connections.size:
        target["connections"] = (
            unit_connections
            if target["connections"].size == 0
            else np.vstack([target["connections"], unit_connections])
        )
        target["metrics"] = np.concatenate([target["metrics"], unit_metrics])
        target["origin_indices"] = np.concatenate([target["origin_indices"], unit_origin_indices])
        target.setdefault("connection_sources", []).extend(unit_connection_sources)

    _merge_edge_diagnostics(
        cast("dict[str, Any]", target["diagnostics"]),
        cast("dict[str, Any]", unit_payload.get("diagnostics", {})),
    )


def _normalize_candidate_origin_counts(raw_counts: dict[Any, Any] | None) -> dict[str, int]:
    """Return a JSON-safe mapping from origin index to candidate count."""
    normalized: dict[str, int] = {}
    if not raw_counts:
        return normalized

    for key, value in raw_counts.items():
        try:
            normalized[str(int(key))] = int(value)
        except (TypeError, ValueError):
            continue
    return normalized


def _normalize_candidate_connection_sources(
    raw_sources: Any,
    candidate_connection_count: int,
    *,
    default_source: str = "unknown",
) -> list[str]:
    """Return a normalized per-connection source label list."""
    if candidate_connection_count <= 0:
        return []

    if isinstance(raw_sources, np.ndarray):
        source_values = np.asarray(raw_sources).reshape(-1).tolist()
    elif isinstance(raw_sources, (list, tuple)):
        source_values = list(raw_sources)
    else:
        source_values = []

    allowed_sources = {"frontier", "watershed", "fallback", "unknown"}
    default_label = default_source if default_source in allowed_sources else "unknown"
    normalized: list[str] = []
    for index in range(candidate_connection_count):
        if index < len(source_values):
            source_label = str(source_values[index]).strip().lower()
            normalized.append(source_label if source_label in allowed_sources else default_label)
            continue
        normalized.append(default_label)
    return normalized


def _build_edge_candidate_audit(
    candidates: dict[str, Any],
    vertex_count: int,
    use_frontier_tracer: bool,
    frontier_origin_counts: dict[int, int] | None = None,
    supplement_origin_counts: dict[int, int] | None = None,
) -> dict[str, Any]:
    """Build a stable, JSON-serializable summary of edge-candidate provenance."""
    connections = np.asarray(
        candidates.get("connections", np.zeros((0, 2), dtype=np.int32)), dtype=np.int32
    )
    if connections.ndim == 1 and connections.size > 0:
        connections = connections.reshape(-1, 2)
    candidate_connection_count = int(connections.shape[0]) if connections.size else 0

    origin_indices = np.asarray(
        candidates.get("origin_indices", np.zeros((0,), dtype=np.int32)), dtype=np.int32
    )
    origin_indices = origin_indices.reshape(-1)
    if origin_indices.size != candidate_connection_count:
        origin_indices = np.zeros((candidate_connection_count,), dtype=np.int32)

    connection_sources = _normalize_candidate_connection_sources(
        candidates.get("connection_sources"),
        candidate_connection_count,
        default_source=str(candidates.get("candidate_source", "unknown")),
    )

    total_origin_counts: dict[int, int] = {}
    total_origin_pairs: dict[int, set[tuple[int, int]]] = {}
    source_pair_sets: dict[str, set[tuple[int, int]]] = {
        "frontier": set(),
        "watershed": set(),
        "fallback": set(),
    }
    pair_sources: dict[tuple[int, int], set[str]] = {}
    source_origin_pair_sets: dict[str, dict[int, set[tuple[int, int]]]] = {
        "frontier": {},
        "watershed": {},
        "fallback": {},
    }
    source_origin_sets: dict[str, set[int]] = {
        "frontier": set(),
        "watershed": set(),
        "fallback": set(),
    }
    for index, origin_index in enumerate(origin_indices):
        origin_index_int = int(origin_index)
        if origin_index_int < 0:
            continue
        total_origin_counts[origin_index_int] = total_origin_counts.get(origin_index_int, 0) + 1

        source_label = connection_sources[index] if index < len(connection_sources) else "unknown"
        if source_label in source_origin_sets:
            source_origin_sets[source_label].add(origin_index_int)

        if index >= len(connections):
            continue
        start_vertex, end_vertex = (int(value) for value in connections[index][:2])
        if start_vertex < 0 or end_vertex < 0:
            continue
        endpoint_pair = (
            (start_vertex, end_vertex) if start_vertex < end_vertex else (end_vertex, start_vertex)
        )
        total_origin_pairs.setdefault(origin_index_int, set()).add(endpoint_pair)
        if source_label in source_pair_sets:
            pair_sources.setdefault(endpoint_pair, set()).add(source_label)
            source_pair_sets[source_label].add(endpoint_pair)
            source_origin_pair_sets[source_label].setdefault(origin_index_int, set()).add(
                endpoint_pair
            )

    frontier_origin_counts = {
        int(origin): int(count) for origin, count in (frontier_origin_counts or {}).items()
    }
    supplement_origin_counts = {
        int(origin): int(count) for origin, count in (supplement_origin_counts or {}).items()
    }
    if frontier_origin_counts or supplement_origin_counts:
        frontier_connection_count = sum(frontier_origin_counts.values())
        supplement_connection_count = sum(supplement_origin_counts.values())
        fallback_connection_count = max(
            0, candidate_connection_count - frontier_connection_count - supplement_connection_count
        )
        frontier_origin_count = len(frontier_origin_counts)
        supplement_origin_count = len(supplement_origin_counts)
        origin_count_union = set(frontier_origin_counts.keys()) | set(
            supplement_origin_counts.keys()
        )
        fallback_origin_count = max(0, len(total_origin_counts) - len(origin_count_union))
    else:
        frontier_connection_count = len(
            [source for source in connection_sources if source == "frontier"]
        )
        supplement_connection_count = len(
            [source for source in connection_sources if source == "watershed"]
        )
        fallback_connection_count = max(
            0, candidate_connection_count - frontier_connection_count - supplement_connection_count
        )
        frontier_origin_count = len(source_origin_sets["frontier"])
        supplement_origin_count = len(source_origin_sets["watershed"])
        fallback_origin_count = max(
            0,
            len(total_origin_counts)
            - len(source_origin_sets["frontier"] | source_origin_sets["watershed"]),
        )

    per_origin_payload: list[dict[str, Any]] = []
    all_origins = (
        set(total_origin_counts.keys())
        | set(frontier_origin_counts.keys())
        | set(supplement_origin_counts.keys())
    )
    for origin_index in sorted(all_origins):
        frontier_count = int(frontier_origin_counts.get(origin_index, 0))
        supplement_count = int(supplement_origin_counts.get(origin_index, 0))
        total_count = int(total_origin_counts.get(origin_index, 0))
        fallback_count = max(0, total_count - frontier_count - supplement_count)
        candidate_pairs = total_origin_pairs.get(origin_index, set())
        frontier_pairs = source_origin_pair_sets["frontier"].get(origin_index, set())
        watershed_pairs = source_origin_pair_sets["watershed"].get(origin_index, set())
        fallback_pairs = source_origin_pair_sets["fallback"].get(origin_index, set())
        per_origin_payload.append(
            {
                "origin_index": origin_index,
                "frontier_candidate_count": frontier_count,
                "watershed_candidate_count": supplement_count,
                "fallback_candidate_count": fallback_count,
                "candidate_connection_count": total_count,
                "candidate_endpoint_pair_count": len(candidate_pairs),
                "candidate_endpoint_pair_samples": sorted(candidate_pairs)[:3],
                "frontier_endpoint_pair_count": len(frontier_pairs),
                "frontier_endpoint_pair_samples": sorted(frontier_pairs)[:3],
                "watershed_endpoint_pair_count": len(watershed_pairs),
                "watershed_endpoint_pair_samples": sorted(watershed_pairs)[:3],
                "fallback_endpoint_pair_count": len(fallback_pairs),
                "fallback_endpoint_pair_samples": sorted(fallback_pairs)[:3],
            }
        )

    diag = candidates.get("diagnostics", {})
    candidate_diagnostics: dict[str, int] = {
        "candidate_traced_edge_count": int(diag.get("candidate_traced_edge_count", 0)),
        "terminal_edge_count": int(diag.get("terminal_edge_count", 0)),
        "chosen_edge_count": int(diag.get("chosen_edge_count", 0)),
        "watershed_join_supplement_count": int(diag.get("watershed_join_supplement_count", 0)),
        "watershed_endpoint_degree_rejected": int(
            diag.get("watershed_endpoint_degree_rejected", 0)
        ),
        "watershed_total_pairs": int(diag.get("watershed_total_pairs", 0)),
        "watershed_already_existing": int(diag.get("watershed_already_existing", 0)),
        "watershed_short_trace_rejected": int(diag.get("watershed_short_trace_rejected", 0)),
        "watershed_energy_rejected": int(diag.get("watershed_energy_rejected", 0)),
        "watershed_reachability_rejected": int(diag.get("watershed_reachability_rejected", 0)),
        "watershed_mutual_frontier_rejected": int(
            diag.get("watershed_mutual_frontier_rejected", 0)
        ),
        "watershed_cap_rejected": int(diag.get("watershed_cap_rejected", 0)),
        "watershed_accepted": int(diag.get("watershed_accepted", 0)),
        "frontier_origins_with_candidates": int(diag.get("frontier_origins_with_candidates", 0)),
        "frontier_origins_without_candidates": int(
            diag.get("frontier_origins_without_candidates", 0)
        ),
    }

    fallback_source_total = {
        "candidate_connection_count": fallback_connection_count,
        "candidate_origin_count": fallback_origin_count,
        "candidate_endpoint_pair_count": len(source_pair_sets["fallback"]),
        "candidate_endpoint_pair_samples": sorted(source_pair_sets["fallback"])[:5],
    }
    frontier_only_pairs = sorted(
        pair for pair, sources in pair_sources.items() if sources == {"frontier"}
    )
    watershed_only_pairs = sorted(
        pair for pair, sources in pair_sources.items() if sources == {"watershed"}
    )
    fallback_only_pairs = sorted(
        pair for pair, sources in pair_sources.items() if sources == {"fallback"}
    )
    multi_source_pairs = sorted(pair for pair, sources in pair_sources.items() if len(sources) > 1)

    return {
        "schema_version": 1,
        "vertex_count": int(vertex_count),
        "use_frontier_tracer": bool(use_frontier_tracer),
        "candidate_traces": len(candidates.get("traces", [])),
        "candidate_connection_count": candidate_connection_count,
        "candidate_origin_count": len(total_origin_counts),
        "source_breakdown": {
            "frontier": {
                "candidate_connection_count": frontier_connection_count,
                "candidate_origin_count": frontier_origin_count,
                "candidate_endpoint_pair_count": len(source_pair_sets["frontier"]),
                "candidate_endpoint_pair_samples": sorted(source_pair_sets["frontier"])[:5],
            },
            "watershed": {
                "candidate_connection_count": supplement_connection_count,
                "candidate_origin_count": supplement_origin_count,
                "candidate_endpoint_pair_count": len(source_pair_sets["watershed"]),
                "candidate_endpoint_pair_samples": sorted(source_pair_sets["watershed"])[:5],
            },
            "fallback": fallback_source_total,
        },
        "frontier_per_origin_candidate_counts": frontier_origin_counts,
        "watershed_per_origin_candidate_counts": _normalize_candidate_origin_counts(
            diag.get("watershed_per_origin_candidate_counts")
        ),
        "pair_source_breakdown": {
            "frontier_only_pair_count": len(frontier_only_pairs),
            "watershed_only_pair_count": len(watershed_only_pairs),
            "fallback_only_pair_count": len(fallback_only_pairs),
            "multi_source_pair_count": len(multi_source_pairs),
            "frontier_only_endpoint_pair_samples": frontier_only_pairs[:5],
            "watershed_only_endpoint_pair_samples": watershed_only_pairs[:5],
            "fallback_only_endpoint_pair_samples": fallback_only_pairs[:5],
        },
        "per_origin_summary": per_origin_payload,
        "diagnostic_counters": candidate_diagnostics,
    }


def _generate_edge_candidates_matlab_frontier(
    energy: np.ndarray,
    scale_indices: np.ndarray | None,
    vertex_positions: np.ndarray,
    vertex_scales: np.ndarray,
    lumen_radius_microns: np.ndarray,
    microns_per_voxel: np.ndarray,
    vertex_center_image: np.ndarray,
    params: dict[str, Any],
) -> dict[str, Any]:
    """Generate edge candidates using the MATLAB-style best-first frontier search."""
    candidates: dict[str, Any] = {
        "traces": [],
        "connections": np.zeros((0, 2), dtype=np.int32),
        "metrics": np.zeros((0,), dtype=np.float32),
        "energy_traces": [],
        "scale_traces": [],
        "origin_indices": np.zeros((0,), dtype=np.int32),
        "connection_sources": [],
        "diagnostics": _empty_edge_diagnostics(),
    }
    per_origin_candidate_counts: dict[int, int] = {}
    for origin_vertex_idx in range(len(vertex_positions)):
        unit_payload = _trace_origin_edges_matlab_frontier(
            energy,
            scale_indices,
            vertex_positions,
            vertex_scales,
            lumen_radius_microns,
            microns_per_voxel,
            vertex_center_image,
            origin_vertex_idx,
            params,
        )
        n_unit_traces = len(unit_payload.get("traces", []))
        if n_unit_traces > 0:
            per_origin_candidate_counts[origin_vertex_idx] = n_unit_traces
        _append_candidate_unit(candidates, unit_payload)

    # Phase 1 per-origin summary diagnostics
    candidates["diagnostics"]["frontier_origins_with_candidates"] = len(per_origin_candidate_counts)
    candidates["diagnostics"]["frontier_origins_without_candidates"] = len(vertex_positions) - len(
        per_origin_candidate_counts
    )
    candidates["diagnostics"]["frontier_per_origin_candidate_counts"] = per_origin_candidate_counts
    logger.info(
        "Frontier candidates: %d origins produced candidates, %d did not",
        len(per_origin_candidate_counts),
        len(vertex_positions) - len(per_origin_candidate_counts),
    )
    return candidates


def _generate_edge_candidates(
    energy: np.ndarray,
    scale_indices: np.ndarray | None,
    vertex_positions: np.ndarray,
    vertex_scales: np.ndarray,
    lumen_radius_pixels: np.ndarray,
    lumen_radius_microns: np.ndarray,
    microns_per_voxel: np.ndarray,
    vertex_center_image: np.ndarray | None,
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
        start_radius = _scalar_radius(lumen_radius_pixels[start_scale])
        step_size = start_radius * step_size_ratio
        max_length = start_radius * max_length_ratio
        max_steps = max(1, int(np.ceil(max_length / max(step_size, 1e-12))))

        if direction_method == "hessian":
            directions = estimate_vessel_directions(
                energy, start_pos, start_radius, microns_per_voxel
            )
            if directions.shape[0] < max_edges_per_vertex:
                extra = generate_edge_directions(
                    max_edges_per_vertex - directions.shape[0], seed=vertex_idx
                )
                directions = np.vstack([directions, extra])
            else:
                directions = directions[:max_edges_per_vertex]
        else:
            directions = generate_edge_directions(max_edges_per_vertex, seed=vertex_idx)

        for direction in directions:
            edge_trace, trace_metadata = trace_edge(
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
                tree=tree,
                max_search_radius=max_search_radius,
                origin_vertex_idx=vertex_idx,
                return_metadata=True,
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

    return {
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


def _offset_coords(
    position: np.ndarray,
    offsets: np.ndarray,
    image_shape: tuple[int, int, int],
) -> np.ndarray:
    """Apply relative offsets to a voxel and clip to valid image bounds."""
    coords = offsets + np.rint(position[:3]).astype(np.int32)
    valid = (
        (coords[:, 0] >= 0)
        & (coords[:, 0] < image_shape[0])
        & (coords[:, 1] >= 0)
        & (coords[:, 1] < image_shape[1])
        & (coords[:, 2] >= 0)
        & (coords[:, 2] < image_shape[2])
    )
    return coords[valid]


def _construct_structuring_element_offsets_matlab(radii: np.ndarray) -> np.ndarray:
    """Construct MATLAB-shaped ellipsoid offsets using the original radius equation."""
    radii = np.asarray(radii, dtype=np.float32).reshape(3)
    rounded = np.rint(radii).astype(np.int32)
    safe_radii = np.maximum(radii.astype(np.float64), 1e-6)

    offsets = np.array(
        [
            [y, x, z]
            for z in range(-rounded[2], rounded[2] + 1)
            for x in range(-rounded[1], rounded[1] + 1)
            for y in range(-rounded[0], rounded[0] + 1)
        ],
        dtype=np.int32,
    )
    distances = (
        (offsets[:, 0].astype(np.float64) ** 2) / (safe_radii[0] ** 2)
        + (offsets[:, 1].astype(np.float64) ** 2) / (safe_radii[1] ** 2)
        + (offsets[:, 2].astype(np.float64) ** 2) / (safe_radii[2] ** 2)
    )
    kept = offsets[distances <= 1.0]
    if kept.size == 0:
        return np.zeros((1, 3), dtype=np.int32)
    return kept.astype(np.int32, copy=False)


def _offset_coords_matlab(
    position: np.ndarray,
    offsets: np.ndarray,
    image_shape: tuple[int, int, int],
) -> np.ndarray:
    """Apply offsets with MATLAB's edge-handling rule of snapping overflow back to center."""
    base = np.rint(position[:3]).astype(np.int32)
    coords = offsets.astype(np.int32, copy=False) + base
    for axis, size in enumerate(image_shape):
        invalid = (coords[:, axis] < 0) | (coords[:, axis] >= size)
        coords[invalid, axis] = base[axis]
    return coords


def _clean_edges_vertex_degree_excess_python(
    connections: np.ndarray,
    metrics: np.ndarray,
    max_edges_per_vertex: int,
) -> np.ndarray:
    """Mirror MATLAB's excess-degree cleanup on best-to-worst sorted edges."""
    if connections.size == 0 or max_edges_per_vertex <= 0:
        return np.ones((len(connections),), dtype=bool)

    # Edge cleanup
    keep: np.ndarray = np.ones((len(connections),), dtype=bool)
    n_vertices = int(np.max(connections)) + 1 if connections.size else 0
    if n_vertices <= 0:
        return keep

    adjacency: list[list[int]] = [[] for _ in range(n_vertices)]
    for edge_index, (start_vertex, end_vertex) in enumerate(connections):
        adjacency[int(start_vertex)].append(edge_index)
        adjacency[int(end_vertex)].append(edge_index)

    for edge_indices in adjacency:
        excess = len(edge_indices) - max_edges_per_vertex
        if excess > 0:
            for edge_index in sorted(edge_indices, reverse=True)[:excess]:
                keep[edge_index] = False
    return keep


def _clean_edges_orphans_python(
    traces: list[np.ndarray],
    image_shape: tuple[int, int, int],
    vertex_positions: np.ndarray,
) -> np.ndarray:
    """Remove edges whose endpoints do not touch a vertex or any interior edge voxel."""
    if not traces:
        return np.zeros((0,), dtype=bool)

    vertex_coords = np.rint(np.asarray(vertex_positions, dtype=np.float32)).astype(
        np.int32, copy=False
    )
    vertex_coords[:, 0] = np.clip(vertex_coords[:, 0], 0, image_shape[0] - 1)
    vertex_coords[:, 1] = np.clip(vertex_coords[:, 1], 0, image_shape[1] - 1)
    vertex_coords[:, 2] = np.clip(vertex_coords[:, 2], 0, image_shape[2] - 1)
    vertex_locations = {
        int(y + x * image_shape[0] + z * image_shape[0] * image_shape[1])
        for y, x, z in vertex_coords.tolist()
    }

    locations = []
    for trace in traces:
        coords = _clip_trace_indices(np.asarray(trace, dtype=np.float32), image_shape)
        locations.append(
            np.asarray(
                coords[:, 0]
                + coords[:, 1] * image_shape[0]
                + coords[:, 2] * image_shape[0] * image_shape[1],
                dtype=np.int64,
            )
        )

    keep: np.ndarray = np.ones((len(locations),), dtype=bool)
    changed = True
    while changed:
        changed = False
        interior: set[int] = set()
        exterior_locations: list[tuple[int, int]] = []
        for edge_index, edge_locations in enumerate(locations):
            if not keep[edge_index] or edge_locations.size == 0:
                continue
            if edge_locations.size > 2:
                interior.update(int(value) for value in edge_locations[1:-1].tolist())
            exterior_locations.append((edge_index, int(edge_locations[0])))
            exterior_locations.append((edge_index, int(edge_locations[-1])))

        removable: set[int] = set()
        allowed = interior | vertex_locations
        for edge_index, location in exterior_locations:
            if location not in allowed:
                removable.add(edge_index)

        if removable:
            changed = True
            for edge_index in removable:
                keep[edge_index] = False
    return keep


def _clean_edges_cycles_python(connections: np.ndarray) -> np.ndarray:
    """Remove cycle-closing edges while preserving the best-to-worst order."""
    if connections.size == 0:
        return np.zeros((0,), dtype=bool)

    keep: np.ndarray = np.ones((len(connections),), dtype=bool)
    n_vertices = int(np.max(connections)) + 1
    parent = np.arange(n_vertices, dtype=np.int32)

    def find(vertex: int) -> int:
        while parent[vertex] != vertex:
            parent[vertex] = parent[parent[vertex]]
            vertex = parent[vertex]
        return int(vertex)

    for edge_index, (start_vertex, end_vertex) in enumerate(connections):
        root_start = find(int(start_vertex))
        root_end = find(int(end_vertex))
        if root_start == root_end:
            keep[edge_index] = False
            continue
        parent[root_end] = root_start
    return keep


def _choose_edges_matlab_style(
    candidates: dict[str, Any],
    vertex_positions: np.ndarray,
    vertex_scales: np.ndarray,
    lumen_radius_pixels_axes: np.ndarray,
    image_shape: tuple[int, int, int],
    params: dict[str, Any],
) -> dict[str, Any]:
    """Choose final edges using MATLAB-shaped filtering and cleanup semantics.

    This function is a **downstream safety net**. It should not be used to
    compensate for upstream semantic drift in the frontier tracer or watershed
    supplement. If parity is failing here, the root cause is almost certainly
    in ``_trace_origin_edges_matlab_frontier`` or
    ``_supplement_matlab_frontier_candidates_with_watershed_joins``.

    MATLAB-parity-critical steps (in order):
        1. **Self-edge / dangling removal** — Filters edges where start == end
           or end < 0. MATLAB equivalent: implicit in the edge loop.
        2. **Non-negative energy rejection** — Removes edges whose energy trace
           max is >= 0. MATLAB equivalent: ``max(trace) >= 0`` guard.
        3. **Directed dedup, then undirected (antiparallel) dedup** — Prefer
           the best-metric copy. MATLAB equivalent: unique-pair filter.
        4. **Conflict painting** — Reject edges that overlap with already-chosen
           edges in voxel space. MATLAB equivalent: 3D label image overlap test.
        5. **Degree pruning** — Cap edges per vertex to ``number_of_edges_per_vertex``.
           MATLAB equivalent: explicit cap in ``choose_edges_V300.m``.
        6. **Orphan pruning** — Remove edges whose endpoints touch neither a
           vertex nor any interior edge voxel. Python-only safety net.
        7. **Cycle pruning** — Remove cycle-closing edges via union-find.
           MATLAB equivalent: acyclic spanning-tree construction.
    """
    traces = candidates["traces"]
    connections = np.asarray(candidates["connections"], dtype=np.int32).reshape(-1, 2)
    metrics = np.asarray(candidates["metrics"], dtype=np.float32).reshape(-1)
    energy_traces = candidates["energy_traces"]
    scale_traces = candidates["scale_traces"]
    diagnostics = _empty_edge_diagnostics()
    candidate_diagnostics = candidates.get("diagnostics", {})
    for key, value in candidate_diagnostics.items():
        diagnostics[key] = value.copy() if key == "stop_reason_counts" else value
    diagnostics["candidate_traced_edge_count"] = len(traces)
    diagnostics["terminal_edge_count"] = (
        int(np.sum(connections[:, 1] >= 0)) if len(connections) else 0
    )
    diagnostics["self_edge_count"] = (
        int(np.sum(connections[:, 0] == connections[:, 1])) if len(connections) else 0
    )
    diagnostics["dangling_edge_count"] = (
        int(np.sum(connections[:, 1] < 0)) if len(connections) else 0
    )

    if len(traces) == 0:
        empty = _empty_edges_result(vertex_positions)
        empty["diagnostics"] = diagnostics
        return empty

    valid = (connections[:, 0] != connections[:, 1]) & (connections[:, 1] >= 0)
    filtered_indices = np.flatnonzero(valid)

    if filtered_indices.size:
        nonnegative_max = np.array(
            [
                np.nanmax(np.asarray(energy_traces[index], dtype=np.float32)) >= 0
                for index in filtered_indices
            ],
            dtype=bool,
        )
        diagnostics["negative_energy_rejected_count"] = int(np.sum(nonnegative_max))
        filtered_indices = filtered_indices[~nonnegative_max]
    else:
        diagnostics["negative_energy_rejected_count"] = 0

    if filtered_indices.size == 0:
        empty = _empty_edges_result(vertex_positions)
        empty["diagnostics"] = diagnostics
        return empty

    edge_lengths = np.asarray(
        [len(np.asarray(energy_traces[index])) for index in filtered_indices],
        dtype=np.int32,
    )
    length_sorted = filtered_indices[np.argsort(edge_lengths, kind="stable")]
    ordered = length_sorted[np.argsort(metrics[length_sorted], kind="stable")]

    directed_seen: set[tuple[int, int]] = set()
    directed_indices: list[int] = []
    for index in ordered:
        pair_d = (int(connections[index, 0]), int(connections[index, 1]))
        if pair_d in directed_seen:
            diagnostics["duplicate_directed_pair_count"] += 1
            continue
        directed_seen.add(pair_d)
        directed_indices.append(int(index))

    # Conflict painting
    undirected_seen_u: set[tuple[int, int]] = set()
    antiparallel_indices_u: list[int] = []
    for index in directed_indices:
        u, v = int(connections[index, 0]), int(connections[index, 1])
        pair_u: tuple[int, int] = (u, v) if u < v else (v, u)
        if pair_u in undirected_seen_u:
            diagnostics["antiparallel_pair_count"] += 1
            continue
        undirected_seen_u.add(pair_u)
        antiparallel_indices_u.append(int(index))

    sigma_per_influence_vertices = float(params.get("sigma_per_influence_vertices", 1.0))
    sigma_per_influence_edges = float(params.get("sigma_per_influence_edges", 0.5))
    vertex_offset_cache: dict[int, np.ndarray] = {}
    edge_offset_cache: dict[int, np.ndarray] = {}
    painted_image: np.ndarray = np.zeros(image_shape, dtype=np.int32)

    def vertex_offsets(scale: int) -> np.ndarray:
        if scale not in vertex_offset_cache:
            radii = sigma_per_influence_vertices * lumen_radius_pixels_axes[int(scale)]
            vertex_offset_cache[scale] = _construct_structuring_element_offsets_matlab(radii)
        return vertex_offset_cache[scale]

    def edge_offsets(scale: int) -> np.ndarray:
        if scale not in edge_offset_cache:
            radii = sigma_per_influence_edges * lumen_radius_pixels_axes[int(scale)]
            edge_offset_cache[scale] = _construct_structuring_element_offsets_matlab(radii)
        return edge_offset_cache[scale]

    chosen_indices: list[int] = []
    for index in antiparallel_indices_u:
        start_vertex, end_vertex = (int(value) for value in connections[index])
        endpoint_snapshots: list[tuple[np.ndarray, np.ndarray]] = []

        for vertex_index in (start_vertex, end_vertex):
            coords = _offset_coords_matlab(
                vertex_positions[vertex_index],
                vertex_offsets(int(vertex_scales[vertex_index])),
                image_shape,
            )
            snapshot = painted_image[coords[:, 0], coords[:, 1], coords[:, 2]].copy()
            painted_image[coords[:, 0], coords[:, 1], coords[:, 2]] = 0
            endpoint_snapshots.append((coords, snapshot))

        chosen = True
        trace = np.asarray(traces[index], dtype=np.float32)
        scale_trace = np.asarray(scale_traces[index], dtype=np.int16)
        for point_index, point in enumerate(trace):
            scale_value = int(scale_trace[min(point_index, len(scale_trace) - 1)])
            coords = _offset_coords_matlab(point, edge_offsets(scale_value), image_shape)
            if coords.size == 0:
                continue
            conflicting = {
                int(value)
                for value in painted_image[coords[:, 0], coords[:, 1], coords[:, 2]].tolist()
                if int(value) not in {0, start_vertex + 1, end_vertex + 1}
            }
            if conflicting:
                diagnostics["conflict_rejected_count"] += 1
                chosen = False
                break

        if chosen:
            for vertex_index, (coords, _snapshot) in zip(
                (start_vertex, end_vertex), endpoint_snapshots
            ):
                painted_image[coords[:, 0], coords[:, 1], coords[:, 2]] = vertex_index + 1
            chosen_indices.append(index)
        else:
            for coords, snapshot in endpoint_snapshots:
                painted_image[coords[:, 0], coords[:, 1], coords[:, 2]] = snapshot

    if not chosen_indices:
        empty = _empty_edges_result(vertex_positions)
        empty["diagnostics"] = diagnostics
        return empty

    chosen_connections = connections[chosen_indices]
    chosen_metrics = metrics[chosen_indices]
    keep_degree: np.ndarray = _clean_edges_vertex_degree_excess_python(
        chosen_connections,
        chosen_metrics,
        int(params.get("number_of_edges_per_vertex", 4)),
    )
    diagnostics["degree_pruned_count"] = int(np.sum(~keep_degree))

    after_degree_indices = [
        index for keep, index in zip(keep_degree.tolist(), chosen_indices) if keep
    ]
    if not after_degree_indices:
        empty = _empty_edges_result(vertex_positions)
        empty["diagnostics"] = diagnostics
        return empty

    keep_orphans: np.ndarray = _clean_edges_orphans_python(
        [traces[index] for index in after_degree_indices],
        image_shape,
        vertex_positions,
    )
    diagnostics["orphan_pruned_count"] = int(np.sum(~keep_orphans))
    after_orphan_indices = [
        index for keep, index in zip(keep_orphans.tolist(), after_degree_indices) if keep
    ]
    if not after_orphan_indices:
        empty = _empty_edges_result(vertex_positions)
        empty["diagnostics"] = diagnostics
        return empty

    keep_cycles: np.ndarray = _clean_edges_cycles_python(connections[after_orphan_indices])
    diagnostics["cycle_pruned_count"] = int(np.sum(~keep_cycles))
    final_indices = [
        index for keep, index in zip(keep_cycles.tolist(), after_orphan_indices) if keep
    ]

    result: dict[str, Any] = _empty_edges_result(vertex_positions)
    result["traces"] = [np.asarray(traces[index], dtype=np.float32) for index in final_indices]
    result["connections"] = np.asarray(connections[final_indices], dtype=np.int32).reshape(-1, 2)
    result["energies"] = np.asarray(metrics[final_indices], dtype=np.float32)
    result["energy_traces"] = [
        np.asarray(energy_traces[index], dtype=np.float32) for index in final_indices
    ]
    result["scale_traces"] = [
        np.asarray(scale_traces[index], dtype=np.int16) for index in final_indices
    ]
    result["chosen_candidate_indices"] = np.asarray(final_indices, dtype=np.int32)
    diagnostics["chosen_edge_count"] = len(final_indices)
    result["diagnostics"] = diagnostics
    return result


def estimate_vessel_directions(
    energy: np.ndarray, pos: np.ndarray, radius: float, microns_per_voxel: np.ndarray
) -> np.ndarray:
    """Estimate vessel directions at a vertex via local Hessian analysis."""
    # Determine a small neighborhood around the vertex
    sigma = max(radius / 2.0, 1.0)
    center = np.round(pos).astype(int)
    r = int(max(1, np.ceil(sigma)))
    slices = tuple(slice(max(c - r, 0), min(c + r + 1, s)) for c, s in zip(center, energy.shape))
    patch = energy[slices]
    # Fallback to uniform directions if patch is too small
    if patch.ndim != 3 or min(patch.shape) < 3:
        return generate_edge_directions(2, seed=0)

    # Rescale patch to account for anisotropic voxel spacing
    scale = microns_per_voxel / microns_per_voxel.min()
    if not np.allclose(scale, 1):
        patch = ndi.zoom(patch, scale, order=1, mode="nearest")

    # --- EXPLANATION FOR JUNIOR DEVS ---
    # WHY: We need to find the direction of the vessel at this point.
    # HOW: We calculate the Hessian matrix (second-order partial derivatives) of intensity.
    #      The eigenvalues of the Hessian describe the local curvature:
    #      - Small eigenvector -> Direction of least curvature (along the vessel).
    #      - Large eigenvectors -> Direction of high curvature (across the vessel wall).
    #      We pick the eigenvector corresponding to the smallest absolute eigenvalue
    #      as the vessel direction.
    # -----------------------------------

    # Compute Hessian in the local patch and extract center values
    try:
        raw_hessian = feature.hessian_matrix(
            patch,
            sigma=sigma,
            mode="nearest",
            order="rc",
            use_gaussian_derivatives=False,
        )
    except TypeError:
        raw_hessian = feature.hessian_matrix(
            patch,
            sigma=sigma,
            mode="nearest",
            order="rc",
        )
    hessian_elems = [h * (radius**2) for h in raw_hessian]
    patch_center = tuple(np.array(patch.shape) // 2)
    Hxx, Hxy, Hxz, Hyy, Hyz, Hzz = [h[patch_center] for h in hessian_elems]
    H = np.array(
        [
            [Hxx, Hxy, Hxz],
            [Hxy, Hyy, Hyz],
            [Hxz, Hyz, Hzz],
        ]
    )
    # Eigen decomposition to find principal axis
    try:
        w, v = np.linalg.eigh(H)
    except np.linalg.LinAlgError:
        return generate_edge_directions(2, seed=0)
    if not np.all(np.isfinite(w)):
        return generate_edge_directions(2, seed=0)

    # Fallback if eigenvalues are nearly isotropic or all zero
    w_abs = np.sort(np.abs(w))
    max_eig = w_abs[-1]
    if max_eig == 0 or (w_abs[1] - w_abs[0]) < 1e-6 * max_eig:
        return generate_edge_directions(2, seed=0)

    direction = v[:, np.argmin(np.abs(w))]
    norm = np.linalg.norm(direction)
    if norm == 0 or not np.isfinite(norm):
        return generate_edge_directions(2, seed=0)
    direction = direction / norm
    return np.stack((direction, -direction))


def trace_edge(
    energy: np.ndarray,
    start_pos: np.ndarray,
    direction: np.ndarray,
    step_size: float,
    max_edge_energy: float,
    vertex_positions: np.ndarray,
    vertex_scales: np.ndarray,
    lumen_radius_pixels: np.ndarray,
    lumen_radius_microns: np.ndarray,
    max_steps: int,
    microns_per_voxel: np.ndarray,
    energy_sign: float,
    discrete_steps: bool = False,
    vertex_center_image: np.ndarray | None = None,
    vertex_image: np.ndarray | None = None,
    tree: cKDTree | None = None,
    max_search_radius: float = 0.0,
    origin_vertex_idx: int | None = None,
    return_metadata: bool = False,
) -> list[np.ndarray] | tuple[list[np.ndarray], dict[str, Any]]:
    """Trace an edge through the energy field with adaptive step sizing."""
    if vertex_center_image is None:
        vertex_center_image = vertex_image

    # Tracing state
    trace: list[np.ndarray] = [np.asarray(start_pos, dtype=np.float32).copy()]
    stop_reason: str = "max_steps"
    direct_terminal_vertex: int | None = None

    def finish(reason: str, terminal_vertex: int | None = None) -> Any:
        finalized_trace, metadata = _finalize_traced_edge(
            trace,
            stop_reason=reason,
            direct_terminal_vertex=terminal_vertex,
            vertex_center_image=vertex_center_image,
            vertex_positions=vertex_positions,
            vertex_scales=vertex_scales,
            lumen_radius_microns=lumen_radius_microns,
            microns_per_voxel=microns_per_voxel,
            origin_vertex=origin_vertex_idx if origin_vertex_idx is not None else -1,
            tree=tree,
            max_search_radius=max_search_radius,
        )
        if return_metadata:
            return finalized_trace, metadata
        return finalized_trace

    # Scalarize position and direction
    current_pos_y, current_pos_x, current_pos_z = (
        float(start_pos[0]),
        float(start_pos[1]),
        float(start_pos[2]),
    )
    current_dir_y, current_dir_x, current_dir_z = (
        float(direction[0]),
        float(direction[1]),
        float(direction[2]),
    )

    # Precompute for optimized gradient calc
    inv_mpv_2x_y = 1.0 / (2.0 * float(microns_per_voxel[0]))
    inv_mpv_2x_x = 1.0 / (2.0 * float(microns_per_voxel[1]))
    inv_mpv_2x_z = 1.0 / (2.0 * float(microns_per_voxel[2]))

    # Precompute shape scalars
    dim_y: int = int(energy.shape[0])
    dim_x: int = int(energy.shape[1])
    dim_z: int = int(energy.shape[2])
    dim_y_minus_2: int = dim_y - 2
    dim_x_minus_2: int = dim_x - 2
    dim_z_minus_2: int = dim_z - 2

    res_v: Any = finish("bounds")
    if dim_y < 3 or dim_x < 3 or dim_z < 3:
        return res_v

    pos_y = math.floor(current_pos_y)
    pos_x = math.floor(current_pos_x)
    pos_z = math.floor(current_pos_z)
    prev_energy = energy[pos_y, pos_x, pos_z]
    if not math.isfinite(prev_energy):
        return finish("nan")

    for _ in range(max_steps):
        attempt = 0
        while attempt < 10:
            next_pos_y = current_pos_y + current_dir_y * step_size
            next_pos_x = current_pos_x + current_dir_x * step_size
            next_pos_z = current_pos_z + current_dir_z * step_size
            if not (
                math.isfinite(next_pos_y)
                and math.isfinite(next_pos_x)
                and math.isfinite(next_pos_z)
            ):
                return finish("nan")

            if discrete_steps:
                # Rounding logic for discrete steps
                r_next_pos_y = round(next_pos_y)
                r_next_pos_x = round(next_pos_x)
                r_next_pos_z = round(next_pos_z)
                # Check if position changed
                if (
                    r_next_pos_y == round(current_pos_y)
                    and r_next_pos_x == round(current_pos_x)
                    and r_next_pos_z == round(current_pos_z)
                ):
                    return finish("max_steps")
                next_pos_y, next_pos_x, next_pos_z = (
                    float(r_next_pos_y),
                    float(r_next_pos_x),
                    float(r_next_pos_z),
                )

            # Inline bounds check for speed
            if (
                next_pos_y < 0
                or next_pos_y >= dim_y
                or next_pos_x < 0
                or next_pos_x >= dim_x
                or next_pos_z < 0
                or next_pos_z >= dim_z
            ):
                return finish("bounds")

            pos_y = math.floor(next_pos_y)
            pos_x = math.floor(next_pos_x)
            pos_z = math.floor(next_pos_z)
            current_energy = energy[pos_y, pos_x, pos_z]
            if not math.isfinite(current_energy):
                return finish("nan")

            if (energy_sign < 0 and current_energy > max_edge_energy) or (
                energy_sign > 0 and current_energy < max_edge_energy
            ):
                return finish("energy_threshold")
            if (energy_sign < 0 and current_energy > prev_energy) or (
                energy_sign > 0 and current_energy < prev_energy
            ):
                step_size *= 0.5
                if step_size < 0.5:
                    return finish("energy_rise_step_halving")
                attempt += 1
                continue
            break

        # Update current position
        current_pos_y, current_pos_x, current_pos_z = next_pos_y, next_pos_x, next_pos_z
        current_pos_arr = np.array([current_pos_y, current_pos_x, current_pos_z], dtype=np.float32)
        trace.append(current_pos_arr)

        prev_energy = current_energy

        # Optimized gradient computation
        # Use scalar args to avoid allocating arrays
        pos_y_r: int = round(current_pos_y)
        pos_x_r: int = round(current_pos_x)
        pos_z_r: int = round(current_pos_z)

        # Inline gradient computation to avoid function call and allocation
        # Manual clamping
        gp_y: int = pos_y_r
        if gp_y < 1:
            gp_y = 1
        elif gp_y > dim_y_minus_2:
            gp_y = dim_y_minus_2

        gp_x: int = pos_x_r
        if gp_x < 1:
            gp_x = 1
        elif gp_x > dim_x_minus_2:
            gp_x = dim_x_minus_2

        gp_z: int = pos_z_r
        if gp_z < 1:
            gp_z = 1
        elif gp_z > dim_z_minus_2:
            gp_z = dim_z_minus_2

        # Compute gradient components
        grad_y = (energy[gp_y + 1, gp_x, gp_z] - energy[gp_y - 1, gp_x, gp_z]) * inv_mpv_2x_y
        grad_x = (energy[gp_y, gp_x + 1, gp_z] - energy[gp_y, gp_x - 1, gp_z]) * inv_mpv_2x_x
        grad_z = (energy[gp_y, gp_x, gp_z + 1] - energy[gp_y, gp_x, gp_z - 1]) * inv_mpv_2x_z

        # Manual norm
        grad_norm = math.sqrt(grad_y**2 + grad_x**2 + grad_z**2)

        if grad_norm > 1e-12:
            # Project gradient onto plane perpendicular to current direction
            dot_prod = grad_y * current_dir_y + grad_x * current_dir_x + grad_z * current_dir_z

            perp_grad_y = grad_y - current_dir_y * dot_prod
            perp_grad_x = grad_x - current_dir_x * dot_prod
            perp_grad_z = grad_z - current_dir_z * dot_prod

            # Steer along ridge by opposing gradient direction
            sign = 1.0 if energy_sign >= 0 else -1.0
            current_dir_y = current_dir_y - sign * perp_grad_y
            current_dir_x = current_dir_x - sign * perp_grad_x
            current_dir_z = current_dir_z - sign * perp_grad_z

            norm = math.sqrt(current_dir_y**2 + current_dir_x**2 + current_dir_z**2)
            if norm > 1e-12:
                inv_norm = 1.0 / norm
                current_dir_y *= inv_norm
                current_dir_x *= inv_norm
                current_dir_z *= inv_norm

        if vertex_center_image is not None:
            terminal_vertex_idx = vertex_at_position(current_pos_arr, vertex_center_image)
        else:
            terminal_vertex_idx = near_vertex(
                current_pos_arr,
                vertex_positions,
                vertex_scales,
                lumen_radius_microns,
                microns_per_voxel,
                tree=tree,
                max_search_radius=max_search_radius,
            )
        if origin_vertex_idx is not None and terminal_vertex_idx == origin_vertex_idx:
            terminal_vertex_idx = None
        if terminal_vertex_idx is not None:
            direct_terminal_vertex = int(terminal_vertex_idx)
            stop_reason = "direct_terminal_hit"
            break

    return finish(stop_reason, direct_terminal_vertex)  # type: ignore[no-any-return]


def extract_edges(
    energy_data: dict[str, Any], vertices: dict[str, Any], params: dict[str, Any]
) -> dict[str, Any]:
    """
    Extract edges by tracing from vertices through energy field.
    MATLAB Equivalent: `get_edges_V300.m`
    """
    logger.info("Extracting edges")

    energy = energy_data["energy"]
    vertex_positions = vertices["positions"]
    vertex_scales = vertices["scales"]
    lumen_radius_pixels = energy_data["lumen_radius_pixels"]
    lumen_radius_microns = energy_data["lumen_radius_microns"]
    energy_sign = energy_data.get("energy_sign", -1.0)

    microns_per_voxel = np.array(params.get("microns_per_voxel", [1.0, 1.0, 1.0]), dtype=float)
    scale_indices = energy_data.get("scale_indices")

    if len(vertex_positions) == 0:
        logger.info("Extracted 0 edges")
        return _empty_edges_result(vertex_positions)

    lumen_radius_pixels_axes = energy_data["lumen_radius_pixels_axes"]
    logger.info("Creating vertex center lookup image...")
    vertex_center_image = paint_vertex_center_image(vertex_positions, energy.shape)
    logger.info("Vertex center lookup image created")

    vertex_positions_microns = vertex_positions * microns_per_voxel
    tree = cKDTree(vertex_positions_microns)
    max_vertex_radius = np.max(lumen_radius_microns) if len(lumen_radius_microns) > 0 else 0.0
    max_search_radius = max_vertex_radius * 5.0
    if _use_matlab_frontier_tracer(energy_data, params):
        enforce_frontier_reachability_gate = bool(
            params.get("parity_frontier_reachability_gate", True)
        )
        require_mutual_frontier_participation = bool(
            params.get("parity_require_mutual_frontier_participation", True)
        )
        candidates = _generate_edge_candidates_matlab_frontier(
            energy,
            scale_indices,
            vertex_positions,
            vertex_scales,
            lumen_radius_microns,
            microns_per_voxel,
            vertex_center_image,
            params,
        )
        candidates = _supplement_matlab_frontier_candidates_with_watershed_joins(
            candidates,
            energy,
            scale_indices,
            vertex_positions,
            energy_sign,
            max_edges_per_vertex=int(params.get("number_of_edges_per_vertex", 4)),
            enforce_frontier_reachability=enforce_frontier_reachability_gate,
            require_mutual_frontier_participation=require_mutual_frontier_participation,
        )
    else:
        candidates = _generate_edge_candidates(
            energy,
            scale_indices,
            vertex_positions,
            vertex_scales,
            lumen_radius_pixels,
            lumen_radius_microns,
            microns_per_voxel,
            vertex_center_image,
            tree,
            max_search_radius,
            params,
            energy_sign,
        )
    chosen = _choose_edges_matlab_style(
        candidates,
        vertex_positions.astype(np.float32),
        vertex_scales,
        lumen_radius_pixels_axes,
        energy.shape,
        params,
    )
    logger.info(
        "Extracted %d chosen edges from %d traced candidates",
        len(chosen["traces"]),
        chosen["diagnostics"]["candidate_traced_edge_count"],
    )
    return chosen


def extract_edges_watershed(
    energy_data: dict[str, Any], vertices: dict[str, Any], params: dict[str, Any]
) -> dict[str, Any]:
    """Extract edges using watershed segmentation seeded at vertices."""
    logger.info("Extracting edges via watershed")

    energy = energy_data["energy"]
    energy_sign = float(energy_data.get("energy_sign", -1.0))
    vertex_positions = vertices["positions"]

    markers = np.zeros_like(energy, dtype=np.int32)
    idxs = np.floor(vertex_positions).astype(int)
    idxs = np.clip(idxs, 0, np.array(energy.shape) - 1)
    markers[idxs[:, 0], idxs[:, 1], idxs[:, 2]] = np.arange(1, len(vertex_positions) + 1)

    logger.info("Running watershed on volume (this may take several minutes)...")
    labels = watershed(-energy_sign * energy, markers)
    logger.info("Watershed complete, extracting edges between regions...")
    structure = ndi.generate_binary_structure(3, 1)

    edges: list[np.ndarray] = []
    connections: list[list[int]] = []
    edge_energies: list[float] = []
    seen = set()
    n_vertices = len(vertex_positions)
    log_interval = max(1, n_vertices // 20)  # Log ~20 times over the loop

    for label in range(1, n_vertices + 1):
        if label % log_interval == 0 or label == n_vertices:
            logger.info(
                "Watershed progress: vertex %d / %d, edges so far: %d",
                label,
                n_vertices,
                len(edges),
            )
        region = labels == label
        dilated = ndi.binary_dilation(region, structure)
        neighbors = np.unique(labels[dilated & (labels != label)])
        for neighbor in neighbors:
            if neighbor <= label or neighbor == 0:
                continue
            pair = (label - 1, neighbor - 1)
            if pair in seen:
                continue
            boundary = (ndi.binary_dilation(labels == neighbor, structure) & region) | (
                ndi.binary_dilation(region, structure) & (labels == neighbor)
            )
            coords = np.argwhere(boundary)
            if coords.size == 0:
                continue
            coords = coords.astype(np.float32)
            edges.append(coords)
            idx = np.floor(coords).astype(int)
            energies = energy[idx[:, 0], idx[:, 1], idx[:, 2]]
            edge_energies.append(float(np.mean(energies)))
            connections.append([label - 1, neighbor - 1])
            seen.add(pair)

    logger.info("Extracted %d watershed edges", len(edges))

    return {
        "traces": edges,
        "connections": np.asarray(connections, dtype=np.int32),
        "energies": np.asarray(edge_energies, dtype=np.float32),
        "vertex_positions": vertex_positions.astype(np.float32),
    }


def extract_vertices_resumable(
    energy_data: dict[str, Any],
    params: dict[str, Any],
    stage_controller: StageController,
) -> dict[str, Any]:
    """Extract vertices with persisted MATLAB-style scan, crop, and choose state."""
    from slavv.runtime.run_state import atomic_joblib_dump

    energy = energy_data["energy"]
    scale_indices = energy_data["scale_indices"]
    lumen_radius_pixels = energy_data["lumen_radius_pixels"]
    lumen_radius_pixels_axes = _coerce_radius_axes(
        lumen_radius_pixels,
        energy_data.get("lumen_radius_pixels_axes"),
    )
    energy_sign = energy_data.get("energy_sign", -1.0)
    lumen_radius_microns = energy_data["lumen_radius_microns"]
    energy_upper_bound = params.get("energy_upper_bound", 0.0)
    space_strel_apothem = params.get("space_strel_apothem", 1)
    length_dilation_ratio = params.get("length_dilation_ratio", 1.0)
    max_voxels_per_node = params.get("max_voxels_per_node", 6000)
    block_size = int(params.get("resume_vertex_block_size", 256))

    candidate_path = stage_controller.artifact_path("candidates.pkl")
    cropped_path = stage_controller.artifact_path("cropped_candidates.pkl")
    chosen_mask_path = stage_controller.artifact_path("chosen_mask.pkl")
    choose_state = stage_controller.load_state()

    stage_controller.begin(
        detail="Scanning MATLAB-style vertex candidates",
        units_total=3,
        substage="candidate_scan",
    )
    if not candidate_path.exists():
        positions, scales, energies = _matlab_vertex_candidates(
            energy,
            scale_indices,
            energy_sign,
            energy_upper_bound,
            space_strel_apothem,
            lumen_radius_pixels_axes[0],
            max_voxels_per_node,
        )
        atomic_joblib_dump(
            {
                "positions": positions,
                "scales": scales,
                "energies": energies,
            },
            candidate_path,
        )
    stage_controller.update(units_total=3, units_completed=1, substage="candidate_scan")

    candidate_data = joblib.load(candidate_path)
    vertex_positions = candidate_data["positions"]
    vertex_scales = candidate_data["scales"]
    vertex_energies = candidate_data["energies"]
    if len(vertex_positions) == 0:
        return _empty_vertices_result()

    if not cropped_path.exists():
        positions, scales, energies = _crop_vertices_matlab_style(
            vertex_positions,
            vertex_scales,
            vertex_energies,
            energy.shape,
            lumen_radius_pixels_axes,
            length_dilation_ratio,
        )
        sort_indices = _sort_vertex_order(positions, energies, energy.shape, energy_sign)
        atomic_joblib_dump(
            {
                "positions": positions[sort_indices],
                "scales": scales[sort_indices],
                "energies": energies[sort_indices],
            },
            cropped_path,
        )
    stage_controller.update(units_total=3, units_completed=2, substage="crop_sort")

    ordered = joblib.load(cropped_path)
    vertex_positions = ordered["positions"]
    vertex_scales = ordered["scales"]
    vertex_energies = ordered["energies"]
    if len(vertex_positions) == 0:
        return _empty_vertices_result()

    if chosen_mask_path.exists():
        chosen_mask = joblib.load(chosen_mask_path)
    else:
        chosen_mask = np.zeros(len(vertex_positions), dtype=bool)
    next_index = int(choose_state.get("next_index", 0))
    total_blocks = max(1, int(np.ceil(len(vertex_positions) / max(block_size, 1))))
    completed_blocks = min(total_blocks, next_index // max(block_size, 1))
    stage_controller.update(
        units_total=3 + total_blocks,
        units_completed=2 + completed_blocks,
        substage="choose_paint",
        detail=f"Vertex choose/paint {next_index}/{len(vertex_positions)}",
        resumed=next_index > 0,
    )

    for block_start in range(next_index, len(vertex_positions), max(block_size, 1)):
        block_end = min(len(vertex_positions), block_start + max(block_size, 1))
        chosen_mask = _choose_vertices_matlab_style(
            vertex_positions,
            vertex_scales,
            energy.shape,
            lumen_radius_pixels_axes,
            length_dilation_ratio,
            start_index=block_start,
            end_index=block_end,
            chosen_mask=chosen_mask,
        )
        atomic_joblib_dump(chosen_mask, chosen_mask_path)
        stage_controller.save_state({"next_index": block_end, "block_size": block_size})
        completed_blocks = min(total_blocks, (block_end + block_size - 1) // max(block_size, 1))
        stage_controller.update(
            units_total=3 + total_blocks,
            units_completed=2 + completed_blocks,
            substage="choose_paint",
            detail=f"Vertex choose/paint {block_end}/{len(vertex_positions)}",
            resumed=next_index > 0,
        )

    vertex_positions = vertex_positions[chosen_mask].astype(np.float32)
    vertex_scales = vertex_scales[chosen_mask].astype(np.int16)
    vertex_energies = vertex_energies[chosen_mask].astype(np.float32)
    radii_pixels = lumen_radius_pixels[vertex_scales].astype(np.float32)
    radii_microns = lumen_radius_microns[vertex_scales].astype(np.float32)
    stage_controller.update(
        units_total=3 + total_blocks,
        units_completed=3 + total_blocks,
        substage="finalize",
    )
    return {
        "positions": vertex_positions,
        "scales": vertex_scales,
        "energies": vertex_energies,
        "radii_pixels": radii_pixels,
        "radii_microns": radii_microns,
        "radii": radii_microns,
    }


def _load_edge_units(
    units_dir: Path,
    n_vertices: int,
) -> tuple[dict[str, Any], set[int]]:
    payload = {
        "traces": [],
        "connections": np.zeros((0, 2), dtype=np.int32),
        "metrics": np.zeros((0,), dtype=np.float32),
        "energy_traces": [],
        "scale_traces": [],
        "origin_indices": np.zeros((0,), dtype=np.int32),
        "connection_sources": [],
        "diagnostics": _empty_edge_diagnostics(),
    }
    completed: set[int] = set()

    if not units_dir.exists():
        return payload, completed

    traces: list[np.ndarray] = []
    connections: list[list[int]] = []
    metrics: list[float] = []
    energy_traces: list[np.ndarray] = []
    scale_traces: list[np.ndarray] = []
    origin_indices: list[int] = []
    connection_sources: list[str] = []
    for unit_file in sorted(units_dir.glob("*.pkl")):
        unit_payload = joblib.load(unit_file)
        origin_index = int(unit_payload["origin_index"])
        completed.add(origin_index)
        unit_traces = [np.asarray(trace, dtype=np.float32) for trace in unit_payload["traces"]]
        unit_connections = np.asarray(unit_payload["connections"], dtype=np.int32).reshape(-1, 2)
        traces.extend(unit_traces)
        connections.extend(
            [int(connection[0]), int(connection[1])] for connection in unit_connections
        )
        metrics.extend(float(metric) for metric in unit_payload["metrics"])
        energy_traces.extend(
            np.asarray(trace, dtype=np.float32) for trace in unit_payload["energy_traces"]
        )
        scale_traces.extend(
            np.asarray(trace, dtype=np.int16) for trace in unit_payload["scale_traces"]
        )
        origin_indices.extend(int(value) for value in unit_payload.get("origin_indices", []))
        connection_sources.extend(
            _normalize_candidate_connection_sources(
                unit_payload.get("connection_sources"),
                len(unit_connections),
                default_source=str(unit_payload.get("candidate_source", "unknown")),
            )
        )
        _merge_edge_diagnostics(
            cast("dict[str, Any]", payload["diagnostics"]),
            cast("dict[str, Any]", unit_payload.get("diagnostics", {})),
        )

    payload["traces"] = traces
    payload["connections"] = np.asarray(connections, dtype=np.int32).reshape(-1, 2)
    payload["metrics"] = np.asarray(metrics, dtype=np.float32)
    payload["energy_traces"] = energy_traces
    payload["scale_traces"] = scale_traces
    payload["origin_indices"] = np.asarray(origin_indices, dtype=np.int32)
    payload["connection_sources"] = connection_sources
    return payload, completed


def extract_edges_resumable(
    energy_data: dict[str, Any],
    vertices: dict[str, Any],
    params: dict[str, Any],
    stage_controller: StageController,
) -> dict[str, Any]:
    """Trace edges with per-origin persisted units."""
    from slavv.runtime.run_state import atomic_joblib_dump, atomic_write_json

    energy = energy_data["energy"]
    vertex_positions = vertices["positions"]
    vertex_scales = vertices["scales"]
    lumen_radius_pixels = energy_data["lumen_radius_pixels"]
    lumen_radius_microns = energy_data["lumen_radius_microns"]
    scale_indices = energy_data.get("scale_indices")
    energy_sign = energy_data.get("energy_sign", -1.0)
    max_edges_per_vertex = params.get("number_of_edges_per_vertex", 4)
    step_size_ratio = params.get("step_size_per_origin_radius", 1.0)
    max_edge_energy = params.get("max_edge_energy", 0.0)
    max_length_ratio = params.get("max_edge_length_per_origin_radius", 60.0)
    microns_per_voxel = np.array(params.get("microns_per_voxel", [1.0, 1.0, 1.0]), dtype=float)
    discrete_tracing = params.get("discrete_tracing", False)
    direction_method = params.get("direction_method", "hessian")

    if len(vertex_positions) == 0:
        return _empty_edges_result(vertex_positions)

    units_dir = stage_controller.artifact_path("units")
    units_dir.mkdir(parents=True, exist_ok=True)
    candidate_manifest_path = stage_controller.artifact_path("candidates.pkl")
    candidate_audit_path = stage_controller.artifact_path("candidate_audit.json")
    chosen_manifest_path = stage_controller.artifact_path("chosen_edges.pkl")
    candidates, completed = _load_edge_units(units_dir, len(vertex_positions))

    lumen_radius_pixels_axes = energy_data["lumen_radius_pixels_axes"]
    logger.info("Creating vertex center lookup image...")
    vertex_center_image = paint_vertex_center_image(vertex_positions, energy.shape)
    logger.info("Vertex center lookup image created")
    vertex_positions_microns = vertex_positions * microns_per_voxel
    tree = cKDTree(vertex_positions_microns)
    max_vertex_radius = np.max(lumen_radius_microns) if len(lumen_radius_microns) > 0 else 0.0
    max_search_radius = max_vertex_radius * 5.0
    energy_prepared = np.ascontiguousarray(energy, dtype=np.float64)
    mpv_prepared = np.asarray(microns_per_voxel, dtype=np.float64)
    use_frontier_tracer = _use_matlab_frontier_tracer(energy_data, params)

    stage_controller.begin(
        detail="Tracing edges with resumable origin units",
        units_total=len(vertex_positions) + 2,
        units_completed=len(completed),
        substage="trace_origins",
        resumed=bool(completed),
    )

    frontier_origin_counts: dict[int, int] = {}
    if use_frontier_tracer:
        for origin_index, count in (
            candidates.get("diagnostics", {})
            .get("frontier_per_origin_candidate_counts", {})
            .items()
        ):
            try:
                frontier_origin_counts[int(origin_index)] = int(count)
            except (TypeError, ValueError):
                continue

    # Results for the current vertex
    unit_traces: list[np.ndarray] = []
    unit_connections: list[list[int]] = []
    unit_metrics: list[float] = []
    unit_energy_traces: list[np.ndarray] = []
    unit_scale_traces: list[np.ndarray] = []
    unit_trace_metadata_v: list[dict[str, Any]] = []

    for vertex_idx, (start_pos, start_scale) in enumerate(zip(vertex_positions, vertex_scales)):
        if vertex_idx in completed:
            continue

        if use_frontier_tracer:
            frontier_payload = _trace_origin_edges_matlab_frontier(
                energy,
                scale_indices,
                vertex_positions,
                vertex_scales,
                lumen_radius_microns,
                microns_per_voxel,
                vertex_center_image,
                vertex_idx,
                params,
            )
            unit_traces = cast("list[np.ndarray]", frontier_payload["traces"])
            unit_connections = cast("list[list[int]]", frontier_payload["connections"])
            unit_metrics = cast("list[float]", frontier_payload["metrics"])
            unit_energy_traces = cast("list[np.ndarray]", frontier_payload["energy_traces"])
            unit_scale_traces = cast("list[np.ndarray]", frontier_payload["scale_traces"])
            unit_diagnostics = cast("dict[str, Any]", frontier_payload["diagnostics"])
            unit_trace_metadata_v = []
            frontier_count = len(unit_connections)
            if frontier_count > 0:
                frontier_origin_counts[vertex_idx] = frontier_count
        else:
            unit_traces = []
            unit_connections = []
            unit_metrics = []
            unit_energy_traces = []
            unit_scale_traces = []
            unit_trace_metadata_v = []
            start_radius = _scalar_radius(lumen_radius_pixels[start_scale])
            step_size = start_radius * step_size_ratio
            max_length = start_radius * max_length_ratio
            max_steps = max(1, int(np.ceil(max_length / max(step_size, 1e-12))))
            if direction_method == "hessian":
                directions = estimate_vessel_directions(
                    energy,
                    start_pos,
                    start_radius,
                    microns_per_voxel,
                )
                if directions.shape[0] < max_edges_per_vertex:
                    extra = generate_edge_directions(
                        max_edges_per_vertex - directions.shape[0], seed=vertex_idx
                    )
                    directions = np.vstack([directions, extra])
                else:
                    directions = directions[:max_edges_per_vertex]
            else:
                directions = generate_edge_directions(max_edges_per_vertex, seed=vertex_idx)

            for direction in directions:
                res_te = trace_edge(
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
                    tree=tree,
                    max_search_radius=max_search_radius,
                    origin_vertex_idx=vertex_idx,
                    return_metadata=True,
                )
                edge_trace, trace_metadata = cast("tuple[list[np.ndarray], dict[str, Any]]", res_te)
                if len(edge_trace) <= 1:
                    continue
                edge_arr = np.asarray(edge_trace, dtype=np.float32)
                term_v: int = (
                    int(trace_metadata["terminal_vertex"])
                    if trace_metadata["terminal_vertex"] is not None
                    else -1
                )
                energy_trace = _trace_energy_series(edge_arr, energy)
                scale_trace = _trace_scale_series(edge_arr, scale_indices)
                unit_traces.append(edge_arr)
                unit_connections.append([vertex_idx, term_v])
                unit_metrics.append(_edge_metric_from_energy_trace(energy_trace))
                unit_energy_traces.append(energy_trace)
                unit_scale_traces.append(scale_trace)
                unit_trace_metadata_v.append(trace_metadata)

            unit_diagnostics = _empty_edge_diagnostics()
            for item in unit_trace_metadata_v:
                trace_metadata_item = cast("dict[str, Any]", item)
                _record_trace_diagnostics(unit_diagnostics, trace_metadata_item)

        payload = {
            "origin_index": vertex_idx,
            "candidate_source": "fallback",
            "traces": unit_traces,
            "connections": unit_connections,
            "metrics": unit_metrics,
            "energy_traces": unit_energy_traces,
            "scale_traces": unit_scale_traces,
            "origin_indices": [vertex_idx] * len(unit_traces),
            "connection_sources": ["fallback"] * len(unit_traces),
            "diagnostics": unit_diagnostics,
        }
        atomic_joblib_dump(payload, units_dir / f"vertex_{vertex_idx:06d}.pkl")
        _append_candidate_unit(candidates, payload)
        completed.add(vertex_idx)
        stage_controller.save_state({"last_completed_origin": vertex_idx})
        stage_controller.update(
            units_total=len(vertex_positions) + 2,
            units_completed=len(completed),
            substage="trace_origins",
            detail=f"Tracing origin {vertex_idx + 1}/{len(vertex_positions)}",
            resumed=bool(completed - {vertex_idx}),
        )

    if use_frontier_tracer:
        enforce_frontier_reachability_gate = bool(
            params.get("parity_frontier_reachability_gate", True)
        )
        require_mutual_frontier_participation = bool(
            params.get("parity_require_mutual_frontier_participation", True)
        )
        candidates = _supplement_matlab_frontier_candidates_with_watershed_joins(
            candidates,
            energy,
            scale_indices,
            vertex_positions,
            energy_sign,
            max_edges_per_vertex=int(params.get("number_of_edges_per_vertex", 4)),
            enforce_frontier_reachability=enforce_frontier_reachability_gate,
            require_mutual_frontier_participation=require_mutual_frontier_participation,
        )
        supplement_origin_counts = _normalize_candidate_origin_counts(
            candidates.get("diagnostics", {}).get("watershed_per_origin_candidate_counts")
        )
    else:
        supplement_origin_counts = {}

    candidate_audit = _build_edge_candidate_audit(
        candidates,
        len(vertex_positions),
        use_frontier_tracer=use_frontier_tracer,
        frontier_origin_counts=frontier_origin_counts,
        supplement_origin_counts={
            int(origin_index): int(count)
            for origin_index, count in (supplement_origin_counts or {}).items()
        },
    )
    atomic_write_json(candidate_audit_path, candidate_audit)

    atomic_joblib_dump(candidates, candidate_manifest_path)
    stage_controller.update(
        units_total=len(vertex_positions) + 2,
        units_completed=len(completed) + 1,
        substage="consolidate_candidates",
        detail="Consolidated edge candidates",
        resumed=bool(completed),
    )
    chosen = _choose_edges_matlab_style(
        candidates,
        vertex_positions.astype(np.float32),
        vertex_scales,
        lumen_radius_pixels_axes,
        energy.shape,
        params,
    )
    atomic_joblib_dump(chosen, chosen_manifest_path)
    stage_controller.update(
        units_total=len(vertex_positions) + 2,
        units_completed=len(vertex_positions) + 2,
        substage="choose_edges",
        detail="Selected MATLAB-style terminal edges",
        resumed=bool(completed),
    )
    return chosen


def extract_edges_watershed_resumable(
    energy_data: dict[str, Any],
    vertices: dict[str, Any],
    params: dict[str, Any],
    stage_controller: StageController,
) -> dict[str, Any]:
    """Extract watershed edges with per-label persisted units."""
    from slavv.runtime.run_state import atomic_joblib_dump

    energy = energy_data["energy"]
    energy_sign = float(energy_data.get("energy_sign", -1.0))
    vertex_positions = vertices["positions"]
    markers = np.zeros_like(energy, dtype=np.int32)
    idxs = np.floor(vertex_positions).astype(int)
    idxs = np.clip(idxs, 0, np.array(energy.shape) - 1)
    markers[idxs[:, 0], idxs[:, 1], idxs[:, 2]] = np.arange(1, len(vertex_positions) + 1)

    logger.info("Running watershed on volume (this may take several minutes)...")
    labels = watershed(-energy_sign * energy, markers)
    logger.info("Watershed complete, extracting edges between regions...")
    structure = ndi.generate_binary_structure(3, 1)

    units_dir = stage_controller.artifact_path("units")
    units_dir.mkdir(parents=True, exist_ok=True)
    existing_payload, completed = _load_edge_units(units_dir, len(vertex_positions))
    edges = existing_payload["traces"]
    connections = (
        existing_payload["connections"].tolist() if existing_payload["connections"].size else []
    )
    edge_energies = existing_payload["metrics"].tolist()
    seen_pairs = {
        tuple(sorted((int(start), int(end))))
        for start, end in np.asarray(existing_payload["connections"], dtype=np.int32).reshape(-1, 2)
        if int(start) >= 0 and int(end) >= 0
    }
    stage_controller.begin(
        detail="Tracing watershed label adjacencies",
        units_total=len(vertex_positions),
        units_completed=len(completed),
        substage="watershed_labels",
        resumed=bool(completed),
    )

    for label in range(1, len(vertex_positions) + 1):
        origin_index = label - 1
        if origin_index in completed:
            continue
        region = labels == label
        dilated = ndi.binary_dilation(region, structure)
        neighbors = np.unique(labels[dilated & (labels != label)])
        unit_traces: list[np.ndarray] = []
        unit_connections: list[list[int]] = []
        unit_energies: list[float] = []
        for neighbor in neighbors:
            if neighbor <= label or neighbor == 0:
                continue
            pair = (label - 1, neighbor - 1)
            if pair in seen_pairs:
                continue
            boundary = (ndi.binary_dilation(labels == neighbor, structure) & region) | (
                ndi.binary_dilation(region, structure) & (labels == neighbor)
            )
            coords = np.argwhere(boundary)
            if coords.size == 0:
                continue
            coords = coords.astype(np.float32)
            idx = np.floor(coords).astype(int)
            energies = energy[idx[:, 0], idx[:, 1], idx[:, 2]]
            unit_traces.append(coords)
            unit_connections.append([label - 1, neighbor - 1])
            unit_energies.append(float(np.mean(energies)))
            seen_pairs.add(pair)

        payload = {
            "origin_index": origin_index,
            "candidate_source": "fallback",
            "traces": unit_traces,
            "connections": unit_connections,
            "metrics": unit_energies,
            "energy_traces": [np.asarray([energy], dtype=np.float32) for energy in unit_energies],
            "scale_traces": [np.zeros((len(trace),), dtype=np.int16) for trace in unit_traces],
            "origin_indices": [origin_index] * len(unit_traces),
            "connection_sources": ["fallback"] * len(unit_traces),
        }
        atomic_joblib_dump(payload, units_dir / f"label_{origin_index:06d}.pkl")
        edges.extend(unit_traces)
        connections.extend(unit_connections)
        edge_energies.extend(unit_energies)
        completed.add(origin_index)
        stage_controller.save_state({"last_completed_label": origin_index})
        stage_controller.update(
            units_total=len(vertex_positions),
            units_completed=len(completed),
            substage="watershed_labels",
            detail=f"Watershed label {label}/{len(vertex_positions)}",
            resumed=bool(completed - {origin_index}),
        )

    return {
        "traces": edges,
        "connections": np.asarray(connections, dtype=np.int32).reshape(-1, 2),
        "energies": np.asarray(edge_energies, dtype=np.float32),
        "vertex_positions": vertex_positions.astype(np.float32),
    }


__all__ = [
    "compute_gradient",
    "estimate_vessel_directions",
    "extract_edges",
    "extract_edges_resumable",
    "extract_edges_watershed",
    "extract_edges_watershed_resumable",
    "extract_vertices",
    "extract_vertices_resumable",
    "find_terminal_vertex",
    "generate_edge_directions",
    "in_bounds",
    "near_vertex",
    "trace_edge",
]
