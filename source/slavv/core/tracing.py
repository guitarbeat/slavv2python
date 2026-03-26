"""
Vertex and Edge tracing logic for SLAVV.
Handles vertex extraction (local maxima/minima) and edge tracing through the energy field.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

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
    return (
        slice(max(0, y - apothem), min(shape[0], y + apothem + 1)),
        slice(max(0, x - apothem), min(shape[1], x + apothem + 1)),
        slice(max(0, z - apothem), min(shape[2], z + apothem + 1)),
    )


def _matlab_linear_indices(coords: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    """Return MATLAB-style column-major linear indices for 0-based coordinates."""
    coords = np.asarray(coords, dtype=np.int64)
    return coords[:, 0] + coords[:, 1] * shape[0] + coords[:, 2] * shape[0] * shape[1]


def _sort_vertex_order(
    vertex_positions: np.ndarray,
    vertex_energies: np.ndarray,
    image_shape: tuple[int, int, int],
    energy_sign: float,
) -> np.ndarray:
    """Sort vertices like MATLAB: by energy, then by column-major linear index for ties."""
    if len(vertex_positions) == 0:
        return np.array([], dtype=np.int64)

    linear_indices = _matlab_linear_indices(vertex_positions, image_shape)
    if energy_sign < 0:
        return np.lexsort((linear_indices, vertex_energies))
    return np.lexsort((linear_indices, -vertex_energies))


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
        "terminal_direct_hit_count": 0,
        "terminal_reverse_center_hit_count": 0,
        "terminal_reverse_near_hit_count": 0,
        "stop_reason_counts": _empty_stop_reason_counts(),
    }


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
            return axes

    radii = np.asarray(lumen_radius_pixels, dtype=np.float32).reshape(-1, 1)
    return np.repeat(radii, 3, axis=1)


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
    return tuple(int(value) for value in lattice.tolist())


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
            window_extreme = np.nanmin(window)
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
    return (coords - center).astype(np.int16, copy=False)


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
        scale_index: _ellipsoid_offsets(scaled_radii[scale_index])
        for scale_index in np.unique(scale_indices)
    }
    painted_image = np.zeros(image_shape, dtype=bool)

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
        return coords[valid]

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
        return 0 <= pos[0] < shape[0] and 0 <= pos[1] < shape[1] and 0 <= pos[2] < shape[2]

    pos_int = np.floor(pos).astype(int)
    return np.all((pos_int >= 0) & (pos_int < np.array(shape)))


def compute_gradient(
    energy: np.ndarray, pos: np.ndarray, microns_per_voxel: np.ndarray
) -> np.ndarray:
    """Compute gradient at ``pos`` using central differences (wrapper for implementation)."""
    pos_int = np.round(pos).astype(np.int64)
    # Ensure proper dtypes for Numba compatibility (if enabled in impl)
    energy_arr = np.ascontiguousarray(energy, dtype=np.float64)
    mpv_arr = np.asarray(microns_per_voxel, dtype=np.float64)
    return compute_gradient_impl(energy_arr, pos_int, mpv_arr)


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
        return np.empty((0, 3))
    if n_directions == 1:
        return np.array([[0, 0, 1]], dtype=float)

    # Generate random points from a 3D standard normal distribution
    rng = np.random.default_rng(seed)
    points = rng.standard_normal((n_directions, 3))
    # Normalize to unit vectors
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return points / norms


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
    return vertex_image


def paint_vertex_center_image(
    vertex_positions: np.ndarray,
    image_shape: tuple[int, int, int],
) -> np.ndarray:
    """Create a sparse image containing only vertex center identities."""
    center_image = np.zeros(image_shape, dtype=np.uint16)
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
    logger.info("Painted %d vertex centers into lookup image", len(vertex_positions))
    return center_image


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
                extra = generate_edge_directions(max_edges_per_vertex - directions.shape[0])
                directions = np.vstack([directions, extra])
            else:
                directions = directions[:max_edges_per_vertex]
        else:
            directions = generate_edge_directions(max_edges_per_vertex)

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

    return {
        "traces": traces,
        "connections": np.asarray(connections, dtype=np.int32).reshape(-1, 2),
        "metrics": np.asarray(metrics, dtype=np.float32),
        "energy_traces": energy_traces,
        "scale_traces": scale_traces,
        "origin_indices": np.asarray(origin_indices, dtype=np.int32),
        "diagnostics": diagnostics,
    }


def _ellipsoid_offsets(radii: np.ndarray) -> np.ndarray:
    """Construct centered integer offsets for an ellipsoidal influence volume."""
    ry, rx, rz = [max(float(v), 0.5) for v in np.asarray(radii, dtype=np.float32)]
    mask = ellipsoid(ry, rx, rz, spacing=(1.0, 1.0, 1.0))
    coords = np.column_stack(np.where(mask))
    center = np.asarray(mask.shape) // 2
    return (coords - center).astype(np.int32, copy=False)


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


def _clean_edges_vertex_degree_excess_python(
    connections: np.ndarray,
    max_edges_per_vertex: int,
) -> np.ndarray:
    """Mirror MATLAB's excess-degree cleanup on best-to-worst sorted edges."""
    if connections.size == 0 or max_edges_per_vertex <= 0:
        return np.ones((len(connections),), dtype=bool)

    keep = np.ones((len(connections),), dtype=bool)
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

    keep = np.ones((len(locations),), dtype=bool)
    changed = True
    while changed:
        changed = False
        interior = set()
        exterior_locations: list[tuple[int, int]] = []
        for edge_index, edge_locations in enumerate(locations):
            if not keep[edge_index] or edge_locations.size == 0:
                continue
            if edge_locations.size > 2:
                interior.update(int(value) for value in edge_locations[1:-1].tolist())
            exterior_locations.append((edge_index, int(edge_locations[0])))
            exterior_locations.append((edge_index, int(edge_locations[-1])))

        removable = set()
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

    keep = np.ones((len(connections),), dtype=bool)
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
    """Choose final edges using MATLAB-shaped filtering and cleanup semantics."""
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

    order = np.argsort(metrics[filtered_indices], kind="stable")
    ordered = filtered_indices[order]

    directed_seen: set[tuple[int, int]] = set()
    directed_indices: list[int] = []
    for index in ordered:
        pair = (int(connections[index, 0]), int(connections[index, 1]))
        if pair in directed_seen:
            diagnostics["duplicate_directed_pair_count"] += 1
            continue
        directed_seen.add(pair)
        directed_indices.append(int(index))

    undirected_seen: set[tuple[int, int]] = set()
    antiparallel_indices: list[int] = []
    for index in directed_indices:
        pair = tuple(sorted((int(connections[index, 0]), int(connections[index, 1]))))
        if pair in undirected_seen:
            diagnostics["antiparallel_pair_count"] += 1
            continue
        undirected_seen.add(pair)
        antiparallel_indices.append(int(index))

    sigma_per_influence_vertices = float(params.get("sigma_per_influence_vertices", 1.0))
    sigma_per_influence_edges = float(params.get("sigma_per_influence_edges", 0.5))
    vertex_offset_cache: dict[int, np.ndarray] = {}
    edge_offset_cache: dict[int, np.ndarray] = {}
    painted_image = np.zeros(image_shape, dtype=np.int32)

    def vertex_offsets(scale: int) -> np.ndarray:
        if scale not in vertex_offset_cache:
            radii = sigma_per_influence_vertices * lumen_radius_pixels_axes[int(scale)]
            vertex_offset_cache[scale] = _ellipsoid_offsets(radii)
        return vertex_offset_cache[scale]

    def edge_offsets(scale: int) -> np.ndarray:
        if scale not in edge_offset_cache:
            radii = sigma_per_influence_edges * lumen_radius_pixels_axes[int(scale)]
            edge_offset_cache[scale] = _ellipsoid_offsets(radii)
        return edge_offset_cache[scale]

    chosen_indices: list[int] = []
    for index in antiparallel_indices:
        start_vertex, end_vertex = (int(value) for value in connections[index])
        endpoint_snapshots: list[tuple[np.ndarray, np.ndarray]] = []

        for vertex_index in (start_vertex, end_vertex):
            coords = _offset_coords(
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
            coords = _offset_coords(point, edge_offsets(scale_value), image_shape)
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
    keep_degree = _clean_edges_vertex_degree_excess_python(
        chosen_connections,
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

    keep_orphans = _clean_edges_orphans_python(
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

    keep_cycles = _clean_edges_cycles_python(connections[after_orphan_indices])
    diagnostics["cycle_pruned_count"] = int(np.sum(~keep_cycles))
    final_indices = [
        index for keep, index in zip(keep_cycles.tolist(), after_orphan_indices) if keep
    ]

    result = _empty_edges_result(vertex_positions)
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
        return generate_edge_directions(2)

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
    hessian_elems = [
        h * (radius**2)
        for h in feature.hessian_matrix(patch, sigma=sigma, mode="nearest", order="rc")
    ]
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
        return generate_edge_directions(2)
    if not np.all(np.isfinite(w)):
        return generate_edge_directions(2)

    # Fallback if eigenvalues are nearly isotropic or all zero
    w_abs = np.sort(np.abs(w))
    max_eig = w_abs[-1]
    if max_eig == 0 or (w_abs[1] - w_abs[0]) < 1e-6 * max_eig:
        return generate_edge_directions(2)

    direction = v[:, np.argmin(np.abs(w))]
    norm = np.linalg.norm(direction)
    if norm == 0 or not np.isfinite(norm):
        return generate_edge_directions(2)
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

    trace = [np.asarray(start_pos, dtype=np.float32).copy()]
    stop_reason = "max_steps"
    direct_terminal_vertex: int | None = None

    def finish(reason: str, terminal_vertex: int | None = None):
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
    dim_y, dim_x, dim_z = energy.shape
    dim_y_minus_2 = dim_y - 2
    dim_x_minus_2 = dim_x - 2
    dim_z_minus_2 = dim_z - 2

    if dim_y < 3 or dim_x < 3 or dim_z < 3:
        return finish("bounds")

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
        pos_y = round(current_pos_y)
        pos_x = round(current_pos_x)
        pos_z = round(current_pos_z)

        # Inline gradient computation to avoid function call and allocation
        # Manual clamping
        grad_pos_y = pos_y
        if grad_pos_y < 1:
            grad_pos_y = 1
        elif grad_pos_y > dim_y_minus_2:
            grad_pos_y = dim_y_minus_2

        grad_pos_x = pos_x
        if grad_pos_x < 1:
            grad_pos_x = 1
        elif grad_pos_x > dim_x_minus_2:
            grad_pos_x = dim_x_minus_2

        grad_pos_z = pos_z
        if grad_pos_z < 1:
            grad_pos_z = 1
        elif grad_pos_z > dim_z_minus_2:
            grad_pos_z = dim_z_minus_2

        # Compute gradient components
        grad_y = (
            energy[grad_pos_y + 1, grad_pos_x, grad_pos_z]
            - energy[grad_pos_y - 1, grad_pos_x, grad_pos_z]
        ) * inv_mpv_2x_y
        grad_x = (
            energy[grad_pos_y, grad_pos_x + 1, grad_pos_z]
            - energy[grad_pos_y, grad_pos_x - 1, grad_pos_z]
        ) * inv_mpv_2x_x
        grad_z = (
            energy[grad_pos_y, grad_pos_x, grad_pos_z + 1]
            - energy[grad_pos_y, grad_pos_x, grad_pos_z - 1]
        ) * inv_mpv_2x_z

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

    return finish(stop_reason, direct_terminal_vertex)


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

    edges = []
    connections = []
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
    for unit_file in sorted(units_dir.glob("*.pkl")):
        unit_payload = joblib.load(unit_file)
        origin_index = int(unit_payload["origin_index"])
        completed.add(origin_index)
        traces.extend(np.asarray(trace, dtype=np.float32) for trace in unit_payload["traces"])
        connections.extend(
            [int(connection[0]), int(connection[1])] for connection in unit_payload["connections"]
        )
        metrics.extend(float(metric) for metric in unit_payload["metrics"])
        energy_traces.extend(
            np.asarray(trace, dtype=np.float32) for trace in unit_payload["energy_traces"]
        )
        scale_traces.extend(
            np.asarray(trace, dtype=np.int16) for trace in unit_payload["scale_traces"]
        )
        origin_indices.extend(int(value) for value in unit_payload.get("origin_indices", []))
        unit_diagnostics = unit_payload.get("diagnostics", {})
        for key, value in unit_diagnostics.items():
            if key == "stop_reason_counts":
                for stop_reason, count in value.items():
                    payload["diagnostics"]["stop_reason_counts"][stop_reason] = int(
                        payload["diagnostics"]["stop_reason_counts"].get(stop_reason, 0)
                    ) + int(count)
            elif key in {
                "terminal_direct_hit_count",
                "terminal_reverse_center_hit_count",
                "terminal_reverse_near_hit_count",
            }:
                payload["diagnostics"][key] += int(value)

    payload["traces"] = traces
    payload["connections"] = np.asarray(connections, dtype=np.int32).reshape(-1, 2)
    payload["metrics"] = np.asarray(metrics, dtype=np.float32)
    payload["energy_traces"] = energy_traces
    payload["scale_traces"] = scale_traces
    payload["origin_indices"] = np.asarray(origin_indices, dtype=np.int32)
    return payload, completed


def extract_edges_resumable(
    energy_data: dict[str, Any],
    vertices: dict[str, Any],
    params: dict[str, Any],
    stage_controller: StageController,
) -> dict[str, Any]:
    """Trace edges with per-origin persisted units."""
    from slavv.runtime.run_state import atomic_joblib_dump

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

    stage_controller.begin(
        detail="Tracing edges with resumable origin units",
        units_total=len(vertex_positions) + 2,
        units_completed=len(completed),
        substage="trace_origins",
        resumed=bool(completed),
    )

    for vertex_idx, (start_pos, start_scale) in enumerate(zip(vertex_positions, vertex_scales)):
        if vertex_idx in completed:
            continue

        unit_traces: list[np.ndarray] = []
        unit_connections: list[list[int]] = []
        unit_metrics: list[float] = []
        unit_energy_traces: list[np.ndarray] = []
        unit_scale_traces: list[np.ndarray] = []
        unit_trace_metadata: list[dict[str, Any]] = []
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
                extra = generate_edge_directions(max_edges_per_vertex - directions.shape[0])
                directions = np.vstack([directions, extra])
            else:
                directions = directions[:max_edges_per_vertex]
        else:
            directions = generate_edge_directions(max_edges_per_vertex)

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
            unit_traces.append(edge_arr)
            unit_connections.append(
                [vertex_idx, terminal_vertex if terminal_vertex is not None else -1]
            )
            unit_metrics.append(_edge_metric_from_energy_trace(energy_trace))
            unit_energy_traces.append(energy_trace)
            unit_scale_traces.append(scale_trace)
            unit_trace_metadata.append(trace_metadata)

        unit_diagnostics = _empty_edge_diagnostics()
        for trace_metadata in unit_trace_metadata:
            _record_trace_diagnostics(unit_diagnostics, trace_metadata)

        payload = {
            "origin_index": vertex_idx,
            "traces": unit_traces,
            "connections": unit_connections,
            "metrics": unit_metrics,
            "energy_traces": unit_energy_traces,
            "scale_traces": unit_scale_traces,
            "origin_indices": [vertex_idx] * len(unit_traces),
            "diagnostics": unit_diagnostics,
        }
        atomic_joblib_dump(payload, units_dir / f"vertex_{vertex_idx:06d}.pkl")
        candidates["traces"].extend(unit_traces)
        if len(unit_connections) > 0:
            if candidates["connections"].size == 0:
                candidates["connections"] = np.asarray(unit_connections, dtype=np.int32).reshape(
                    -1, 2
                )
            else:
                candidates["connections"] = np.vstack(
                    [
                        candidates["connections"],
                        np.asarray(unit_connections, dtype=np.int32).reshape(-1, 2),
                    ]
                )
            candidates["metrics"] = np.concatenate(
                [candidates["metrics"], np.asarray(unit_metrics, dtype=np.float32)]
            )
            candidates["origin_indices"] = np.concatenate(
                [
                    candidates["origin_indices"],
                    np.asarray([vertex_idx] * len(unit_traces), dtype=np.int32),
                ]
            )
        candidates["energy_traces"].extend(unit_energy_traces)
        candidates["scale_traces"].extend(unit_scale_traces)
        for key, value in unit_diagnostics.items():
            if key == "stop_reason_counts":
                for stop_reason, count in value.items():
                    candidates["diagnostics"]["stop_reason_counts"][stop_reason] = int(
                        candidates["diagnostics"]["stop_reason_counts"].get(stop_reason, 0)
                    ) + int(count)
            elif key in {
                "terminal_direct_hit_count",
                "terminal_reverse_center_hit_count",
                "terminal_reverse_near_hit_count",
            }:
                candidates["diagnostics"][key] += int(value)
        completed.add(vertex_idx)
        stage_controller.save_state({"last_completed_origin": vertex_idx})
        stage_controller.update(
            units_total=len(vertex_positions) + 2,
            units_completed=len(completed),
            substage="trace_origins",
            detail=f"Tracing origin {vertex_idx + 1}/{len(vertex_positions)}",
            resumed=bool(completed - {vertex_idx}),
        )

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
            "traces": unit_traces,
            "connections": unit_connections,
            "metrics": unit_energies,
            "energy_traces": [np.asarray([energy], dtype=np.float32) for energy in unit_energies],
            "scale_traces": [np.zeros((len(trace),), dtype=np.int16) for trace in unit_traces],
            "origin_indices": [origin_index] * len(unit_traces),
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
