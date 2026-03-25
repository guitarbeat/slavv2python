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


def _matlab_vertex_candidates(
    energy: np.ndarray,
    scale_indices: np.ndarray,
    energy_sign: float,
    energy_upper_bound: float,
    space_strel_apothem: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find MATLAB-style spatial extrema on the min-projected energy volume."""
    apothem = _vertex_window_apothem(space_strel_apothem)
    footprint = np.ones((2 * apothem + 1, 2 * apothem + 1, 2 * apothem + 1), dtype=bool)

    if energy_sign < 0:
        filt = ndi.minimum_filter(energy, footprint=footprint, mode="nearest")
        extrema = (energy <= filt) & (energy < energy_upper_bound)
    else:
        filt = ndi.maximum_filter(energy, footprint=footprint, mode="nearest")
        extrema = (energy >= filt) & (energy > energy_upper_bound)

    if apothem > 0:
        extrema[:apothem, :, :] = False
        extrema[-apothem:, :, :] = False
        extrema[:, :apothem, :] = False
        extrema[:, -apothem:, :] = False
        extrema[:, :, :apothem] = False
        extrema[:, :, -apothem:] = False

    coords = np.where(extrema)
    return np.column_stack(coords), scale_indices[coords], energy[coords]


def _suppress_vertices_matlab_style(
    vertex_positions: np.ndarray,
    energy_shape: tuple[int, int, int],
    space_strel_apothem: int,
    start_index: int = 0,
    keep_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Greedily keep vertices and zero a fixed voxel neighborhood like MATLAB."""
    apothem = _vertex_window_apothem(space_strel_apothem)
    if keep_mask is None:
        keep_mask = np.ones(len(vertex_positions), dtype=bool)

    suppressed = np.zeros(energy_shape, dtype=bool)
    for prior_index in range(start_index):
        if not keep_mask[prior_index]:
            continue
        slices = _vertex_neighborhood_slices(vertex_positions[prior_index], apothem, energy_shape)
        suppressed[slices] = True

    for index in range(start_index, len(vertex_positions)):
        pos = vertex_positions[index]
        if suppressed[tuple(int(coord) for coord in pos)]:
            keep_mask[index] = False
            continue

        keep_mask[index] = True
        slices = _vertex_neighborhood_slices(pos, apothem, energy_shape)
        suppressed[slices] = True

    return keep_mask


def _crop_vertices_matlab_style(
    vertex_positions: np.ndarray,
    vertex_scales: np.ndarray,
    vertex_energies: np.ndarray,
    lumen_radius_pixels_axes: np.ndarray,
    image_shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Remove boundary-touching and extreme-scale vertices like MATLAB."""
    if len(vertex_positions) == 0:
        return vertex_positions, vertex_scales, vertex_energies

    radii = np.round(lumen_radius_pixels_axes[vertex_scales]).astype(int)
    shape = np.asarray(image_shape, dtype=int)
    excluded = (
        (vertex_positions[:, 0] + radii[:, 0] > shape[0] - 1)
        | (vertex_positions[:, 1] + radii[:, 1] > shape[1] - 1)
        | (vertex_positions[:, 2] + radii[:, 2] > shape[2] - 1)
        | (vertex_positions[:, 0] - radii[:, 0] < 0)
        | (vertex_positions[:, 1] - radii[:, 1] < 0)
        | (vertex_positions[:, 2] - radii[:, 2] < 0)
        | (vertex_scales == 0)
        | (vertex_scales == len(lumen_radius_pixels_axes) - 1)
    )
    keep = ~excluded
    return vertex_positions[keep], vertex_scales[keep], vertex_energies[keep]


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
    Create a volume where each voxel is labeled with its vertex membership (1-indexed, 0=background).

    This matches MATLAB's approach for fast O(1) vertex detection during edge tracing.
    Paints ellipsoidal regions around each vertex with the vertex index.

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


def extract_vertices(energy_data: dict[str, Any], params: dict[str, Any]) -> dict[str, Any]:
    """
    Extract vertices as local extrema in the energy field.
    MATLAB Equivalent: `get_vertices_V200.m`
    """
    logger.info("Extracting vertices")

    energy = energy_data["energy"]
    scale_indices = energy_data["scale_indices"]
    lumen_radius_pixels = energy_data["lumen_radius_pixels"]
    lumen_radius_pixels_axes = np.asarray(
        energy_data.get(
            "lumen_radius_pixels_axes",
            np.repeat(np.asarray(lumen_radius_pixels)[:, None], 3, axis=1),
        )
    )
    energy_sign = energy_data.get("energy_sign", -1.0)
    lumen_radius_microns = energy_data["lumen_radius_microns"]

    # Parameters
    energy_upper_bound = params.get("energy_upper_bound", 0.0)
    space_strel_apothem = params.get("space_strel_apothem", 1)
    vertex_positions, vertex_scales, vertex_energies = _matlab_vertex_candidates(
        energy,
        scale_indices,
        energy_sign,
        energy_upper_bound,
        space_strel_apothem,
    )

    # Sort by energy (best first depending on sign)
    if energy_sign < 0:
        sort_indices = np.argsort(vertex_energies)
    else:
        sort_indices = np.argsort(-vertex_energies)
    vertex_positions = vertex_positions[sort_indices]
    vertex_scales = vertex_scales[sort_indices]
    vertex_energies = vertex_energies[sort_indices]

    if len(vertex_positions) == 0:
        logger.info("Extracted 0 vertices")
        return {
            "positions": np.empty((0, 3), dtype=np.float32),
            "scales": np.empty((0,), dtype=np.int16),
            "energies": np.empty((0,), dtype=np.float32),
            "radii_pixels": np.empty((0,), dtype=np.float32),
            "radii_microns": np.empty((0,), dtype=np.float32),
            "radii": np.empty((0,), dtype=np.float32),
        }

    keep_mask = _suppress_vertices_matlab_style(
        vertex_positions,
        energy.shape,
        space_strel_apothem,
    )
    vertex_positions = vertex_positions[keep_mask]
    vertex_scales = vertex_scales[keep_mask]
    vertex_energies = vertex_energies[keep_mask]
    vertex_positions, vertex_scales, vertex_energies = _crop_vertices_matlab_style(
        vertex_positions,
        vertex_scales,
        vertex_energies,
        lumen_radius_pixels_axes,
        energy.shape,
    )

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
    vertex_image: np.ndarray | None = None,
    tree: cKDTree | None = None,
    max_search_radius: float = 0.0,
) -> list[np.ndarray]:
    """Trace an edge through the energy field with adaptive step sizing."""
    trace = [start_pos.copy()]
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
        return trace

    pos_y = math.floor(current_pos_y)
    pos_x = math.floor(current_pos_x)
    pos_z = math.floor(current_pos_z)
    prev_energy = energy[pos_y, pos_x, pos_z]

    for _ in range(max_steps):
        attempt = 0
        while attempt < 10:
            next_pos_y = current_pos_y + current_dir_y * step_size
            next_pos_x = current_pos_x + current_dir_x * step_size
            next_pos_z = current_pos_z + current_dir_z * step_size

            if not all(
                math.isfinite(value)
                for value in (
                    next_pos_y,
                    next_pos_x,
                    next_pos_z,
                    current_dir_y,
                    current_dir_x,
                    current_dir_z,
                )
            ):
                return trace

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
                    return trace
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
                return trace

            pos_y = math.floor(next_pos_y)
            pos_x = math.floor(next_pos_x)
            pos_z = math.floor(next_pos_z)
            current_energy = energy[pos_y, pos_x, pos_z]

            if (energy_sign < 0 and current_energy > max_edge_energy) or (
                energy_sign > 0 and current_energy < max_edge_energy
            ):
                return trace
            if (energy_sign < 0 and current_energy > prev_energy) or (
                energy_sign > 0 and current_energy < prev_energy
            ):
                step_size *= 0.5
                if step_size < 0.5:
                    return trace
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
            else:
                return trace

            if not all(
                math.isfinite(value) for value in (current_dir_y, current_dir_x, current_dir_z)
            ):
                return trace

        # Check if we've reached a vertex (use vertex_image for O(1) lookup if available)
        if vertex_image is not None:
            # OPTIMIZATION: Inlined vertex_at_position to avoid function call and numpy overhead
            # vertex_at_position uses np.floor, so we use math.floor here to match logic
            vi_y = math.floor(current_pos_y)
            vi_x = math.floor(current_pos_x)
            vi_z = math.floor(current_pos_z)

            # Bounds check is implicitly handled because (current_pos_y, x, z) are
            # constrained to be within energy.shape, and vertex_image has the same shape.
            # See loop bounds check above.

            v_id = vertex_image[vi_y, vi_x, vi_z]
            if v_id > 0:
                terminal_vertex_idx = int(v_id - 1)
            else:
                terminal_vertex_idx = None
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
        if terminal_vertex_idx is not None:
            trace.append(vertex_positions[terminal_vertex_idx].copy())
            break

    return trace


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

    # Parameters
    max_edges_per_vertex = params.get("number_of_edges_per_vertex", 4)
    step_size_ratio = params.get("step_size_per_origin_radius", 1.0)
    max_edge_energy = params.get("max_edge_energy", 0.0)
    max_edge_length_ratio = params.get("max_edge_length_per_origin_radius", 60.0)
    microns_per_voxel = np.array(params.get("microns_per_voxel", [1.0, 1.0, 1.0]), dtype=float)
    discrete_tracing = params.get("discrete_tracing", False)
    direction_method = params.get("direction_method", "hessian")

    edges = []
    edge_connections = []
    edge_energies: list[float] = []
    edges_per_vertex = np.zeros(len(vertex_positions), dtype=int)
    existing_pairs = set()

    if len(vertex_positions) == 0:
        logger.info("Extracted 0 edges")
        return {
            "traces": [],
            "connections": np.zeros((0, 2), dtype=np.int32),
            "energies": np.zeros((0,), dtype=np.float32),
            "vertex_positions": vertex_positions.astype(np.float32),
        }

    # Build vertex volume image for O(1) vertex detection (matching MATLAB approach)
    logger.info("Creating vertex volume image...")
    lumen_radius_pixels_axes = energy_data["lumen_radius_pixels_axes"]
    vertex_image = paint_vertex_image(
        vertex_positions, vertex_scales, lumen_radius_pixels_axes, energy.shape
    )
    logger.info("Vertex volume image created")

    # Also build cKDTree as fallback for out-of-volume queries
    vertex_positions_microns = vertex_positions * microns_per_voxel
    tree = cKDTree(vertex_positions_microns)
    max_vertex_radius = np.max(lumen_radius_microns) if len(lumen_radius_microns) > 0 else 0.0
    max_search_radius = max_vertex_radius * 5.0

    # Prepare arrays once for performance (avoiding overhead in trace_edge)
    energy_prepared = np.ascontiguousarray(energy, dtype=np.float64)
    mpv_prepared = np.asarray(microns_per_voxel, dtype=np.float64)

    for vertex_idx, (start_pos, start_scale) in enumerate(zip(vertex_positions, vertex_scales)):
        if edges_per_vertex[vertex_idx] >= max_edges_per_vertex:
            continue
        start_radius = lumen_radius_pixels[start_scale]
        step_size = start_radius * step_size_ratio
        max_length = start_radius * max_edge_length_ratio
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
            if edges_per_vertex[vertex_idx] >= max_edges_per_vertex:
                break
            edge_trace = trace_edge(
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
                vertex_image=vertex_image,
                tree=tree,
                max_search_radius=max_search_radius,
            )
            if len(edge_trace) > 1:  # Valid edge found
                # Use vertex image for fast O(1) lookup at endpoint
                terminal_vertex = vertex_at_position(edge_trace[-1], vertex_image)

                # If endpoint check failed and trace is short, check earlier points
                if terminal_vertex is None and len(edge_trace) <= 5:
                    for point in reversed(edge_trace[-len(edge_trace) : -1]):
                        terminal_vertex = vertex_at_position(point, vertex_image)
                        if terminal_vertex is not None and terminal_vertex != vertex_idx:
                            break
                        if terminal_vertex == vertex_idx:
                            terminal_vertex = None

                # Skip self-connections
                if terminal_vertex == vertex_idx:
                    continue
                if terminal_vertex is not None:
                    if edges_per_vertex[terminal_vertex] >= max_edges_per_vertex:
                        continue
                    pair = tuple(sorted((vertex_idx, terminal_vertex)))
                    if pair in existing_pairs:
                        continue
                edge_arr = np.asarray(edge_trace, dtype=np.float32)
                edges.append(edge_arr)
                idx = np.floor(edge_arr).astype(int)
                energies = energy[idx[:, 0], idx[:, 1], idx[:, 2]]
                edge_energies.append(float(np.mean(energies)))
                edge_connections.append(
                    [
                        vertex_idx,
                        terminal_vertex if terminal_vertex is not None else -1,
                    ]
                )
                edges_per_vertex[vertex_idx] += 1
                if terminal_vertex is not None:
                    edges_per_vertex[terminal_vertex] += 1
                    existing_pairs.add(pair)

    logger.info(f"Extracted {len(edges)} edges")

    edge_connections = np.asarray(edge_connections, dtype=np.int32).reshape(-1, 2)

    return {
        "traces": edges,
        "connections": edge_connections,
        "energies": np.asarray(edge_energies, dtype=np.float32),
        "vertex_positions": vertex_positions.astype(np.float32),
    }


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
    """Extract vertices with persisted materialization and suppression state."""
    from slavv.runtime.run_state import atomic_joblib_dump

    energy = energy_data["energy"]
    scale_indices = energy_data["scale_indices"]
    lumen_radius_pixels = energy_data["lumen_radius_pixels"]
    lumen_radius_pixels_axes = np.asarray(
        energy_data.get(
            "lumen_radius_pixels_axes",
            np.repeat(np.asarray(lumen_radius_pixels)[:, None], 3, axis=1),
        )
    )
    energy_sign = energy_data.get("energy_sign", -1.0)
    lumen_radius_microns = energy_data["lumen_radius_microns"]
    energy_upper_bound = params.get("energy_upper_bound", 0.0)
    space_strel_apothem = params.get("space_strel_apothem", 1)
    block_size = int(params.get("resume_vertex_block_size", 256))

    candidate_path = stage_controller.artifact_path("candidates.pkl")
    ordered_path = stage_controller.artifact_path("ordered_candidates.pkl")
    keep_mask_path = stage_controller.artifact_path("keep_mask.pkl")
    suppression_state = stage_controller.load_state()

    stage_controller.begin(
        detail="Preparing vertex candidates", units_total=4, substage="materialize"
    )
    if not candidate_path.exists():
        positions, scales, energies = _matlab_vertex_candidates(
            energy,
            scale_indices,
            energy_sign,
            energy_upper_bound,
            space_strel_apothem,
        )
        atomic_joblib_dump(
            {
                "positions": positions,
                "scales": scales,
                "energies": energies,
            },
            candidate_path,
        )
    stage_controller.update(units_total=4, units_completed=1, substage="materialize")

    candidate_data = joblib.load(candidate_path)
    vertex_positions = candidate_data["positions"]
    vertex_scales = candidate_data["scales"]
    vertex_energies = candidate_data["energies"]
    if len(vertex_positions) == 0:
        return {
            "positions": np.empty((0, 3), dtype=np.float32),
            "scales": np.empty((0,), dtype=np.int16),
            "energies": np.empty((0,), dtype=np.float32),
            "radii_pixels": np.empty((0,), dtype=np.float32),
            "radii_microns": np.empty((0,), dtype=np.float32),
            "radii": np.empty((0,), dtype=np.float32),
        }

    if not ordered_path.exists():
        if energy_sign < 0:
            sort_indices = np.argsort(vertex_energies)
        else:
            sort_indices = np.argsort(-vertex_energies)
        atomic_joblib_dump(
            {
                "positions": vertex_positions[sort_indices],
                "scales": vertex_scales[sort_indices],
                "energies": vertex_energies[sort_indices],
            },
            ordered_path,
        )
    stage_controller.update(units_total=4, units_completed=2, substage="order")

    ordered = joblib.load(ordered_path)
    vertex_positions = ordered["positions"]
    vertex_scales = ordered["scales"]
    vertex_energies = ordered["energies"]

    if keep_mask_path.exists():
        keep_mask = joblib.load(keep_mask_path)
    else:
        keep_mask = np.ones(len(vertex_positions), dtype=bool)
    next_index = int(suppression_state.get("next_index", 0))
    total_blocks = max(1, int(np.ceil(len(vertex_positions) / max(block_size, 1))))
    completed_blocks = min(total_blocks, next_index // max(block_size, 1))
    stage_controller.update(
        units_total=4 + total_blocks,
        units_completed=2 + completed_blocks,
        substage="suppression",
        detail=f"Vertex suppression {next_index}/{len(vertex_positions)}",
        resumed=next_index > 0,
    )

    keep_mask = _suppress_vertices_matlab_style(
        vertex_positions,
        energy.shape,
        space_strel_apothem,
        start_index=next_index,
        keep_mask=keep_mask,
    )

    for i in range(next_index, len(vertex_positions)):
        if (i + 1) % block_size == 0 or i == len(vertex_positions) - 1:
            atomic_joblib_dump(keep_mask, keep_mask_path)
            stage_controller.save_state({"next_index": i + 1, "block_size": block_size})
            completed_blocks = min(total_blocks, (i + 1 + block_size - 1) // block_size)
            stage_controller.update(
                units_total=4 + total_blocks,
                units_completed=2 + completed_blocks,
                substage="suppression",
                detail=f"Vertex suppression {i + 1}/{len(vertex_positions)}",
                resumed=next_index > 0,
            )

    vertex_positions = vertex_positions[keep_mask].astype(np.float32)
    vertex_scales = vertex_scales[keep_mask].astype(np.int16)
    vertex_energies = vertex_energies[keep_mask].astype(np.float32)
    vertex_positions, vertex_scales, vertex_energies = _crop_vertices_matlab_style(
        vertex_positions,
        vertex_scales,
        vertex_energies,
        lumen_radius_pixels_axes,
        energy.shape,
    )
    radii_pixels = lumen_radius_pixels[vertex_scales].astype(np.float32)
    radii_microns = lumen_radius_microns[vertex_scales].astype(np.float32)
    stage_controller.update(
        units_total=4 + total_blocks, units_completed=3 + total_blocks, substage="finalize"
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
) -> tuple[
    list[np.ndarray], list[list[int]], list[float], np.ndarray, set[tuple[int, int]], set[int]
]:
    edges: list[np.ndarray] = []
    edge_connections: list[list[int]] = []
    edge_energies: list[float] = []
    edges_per_vertex = np.zeros(n_vertices, dtype=int)
    existing_pairs: set[tuple[int, int]] = set()
    completed: set[int] = set()

    if not units_dir.exists():
        return edges, edge_connections, edge_energies, edges_per_vertex, existing_pairs, completed

    for unit_file in sorted(units_dir.glob("*.pkl")):
        payload = joblib.load(unit_file)
        origin_index = int(payload["origin_index"])
        completed.add(origin_index)
        for trace, connection, energy in zip(
            payload["traces"],
            payload["connections"],
            payload["energies"],
        ):
            edges.append(np.asarray(trace, dtype=np.float32))
            edge_connections.append([int(connection[0]), int(connection[1])])
            edge_energies.append(float(energy))
            start_vertex, end_vertex = int(connection[0]), int(connection[1])
            if start_vertex >= 0:
                edges_per_vertex[start_vertex] += 1
            if end_vertex >= 0:
                edges_per_vertex[end_vertex] += 1
                existing_pairs.add(tuple(sorted((start_vertex, end_vertex))))
    return edges, edge_connections, edge_energies, edges_per_vertex, existing_pairs, completed


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
    energy_sign = energy_data.get("energy_sign", -1.0)
    max_edges_per_vertex = params.get("number_of_edges_per_vertex", 4)
    step_size_ratio = params.get("step_size_per_origin_radius", 1.0)
    max_edge_energy = params.get("max_edge_energy", 0.0)
    max_edge_length_ratio = params.get("max_edge_length_per_origin_radius", 60.0)
    microns_per_voxel = np.array(params.get("microns_per_voxel", [1.0, 1.0, 1.0]), dtype=float)
    discrete_tracing = params.get("discrete_tracing", False)
    direction_method = params.get("direction_method", "hessian")

    if len(vertex_positions) == 0:
        return {
            "traces": [],
            "connections": np.zeros((0, 2), dtype=np.int32),
            "energies": np.zeros((0,), dtype=np.float32),
            "vertex_positions": vertex_positions.astype(np.float32),
        }

    units_dir = stage_controller.artifact_path("units")
    units_dir.mkdir(parents=True, exist_ok=True)
    edges, edge_connections, edge_energies, edges_per_vertex, existing_pairs, completed = (
        _load_edge_units(
            units_dir,
            len(vertex_positions),
        )
    )

    logger.info("Creating vertex volume image...")
    lumen_radius_pixels_axes = energy_data["lumen_radius_pixels_axes"]
    vertex_image = paint_vertex_image(
        vertex_positions,
        vertex_scales,
        lumen_radius_pixels_axes,
        energy.shape,
    )
    logger.info("Vertex volume image created")
    vertex_positions_microns = vertex_positions * microns_per_voxel
    tree = cKDTree(vertex_positions_microns)
    max_vertex_radius = np.max(lumen_radius_microns) if len(lumen_radius_microns) > 0 else 0.0
    max_search_radius = max_vertex_radius * 5.0
    energy_prepared = np.ascontiguousarray(energy, dtype=np.float64)
    mpv_prepared = np.asarray(microns_per_voxel, dtype=np.float64)

    stage_controller.begin(
        detail="Tracing edges with resumable origin units",
        units_total=len(vertex_positions),
        units_completed=len(completed),
        substage="trace_origins",
        resumed=bool(completed),
    )

    for vertex_idx, (start_pos, start_scale) in enumerate(zip(vertex_positions, vertex_scales)):
        if vertex_idx in completed:
            continue

        unit_traces: list[np.ndarray] = []
        unit_connections: list[list[int]] = []
        unit_energies: list[float] = []
        if edges_per_vertex[vertex_idx] < max_edges_per_vertex:
            start_radius = lumen_radius_pixels[start_scale]
            step_size = start_radius * step_size_ratio
            max_length = start_radius * max_edge_length_ratio
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
                if edges_per_vertex[vertex_idx] >= max_edges_per_vertex:
                    break
                edge_trace = trace_edge(
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
                    vertex_image=vertex_image,
                    tree=tree,
                    max_search_radius=max_search_radius,
                )
                if len(edge_trace) <= 1:
                    continue
                terminal_vertex = vertex_at_position(edge_trace[-1], vertex_image)
                if terminal_vertex is None and len(edge_trace) <= 5:
                    for point in reversed(edge_trace[:-1]):
                        terminal_vertex = vertex_at_position(point, vertex_image)
                        if terminal_vertex is not None and terminal_vertex != vertex_idx:
                            break
                        if terminal_vertex == vertex_idx:
                            terminal_vertex = None
                if terminal_vertex == vertex_idx:
                    continue
                if terminal_vertex is not None:
                    if edges_per_vertex[terminal_vertex] >= max_edges_per_vertex:
                        continue
                    pair = tuple(sorted((vertex_idx, terminal_vertex)))
                    if pair in existing_pairs:
                        continue
                edge_arr = np.asarray(edge_trace, dtype=np.float32)
                idx = np.floor(edge_arr).astype(int)
                energies = energy[idx[:, 0], idx[:, 1], idx[:, 2]]
                unit_traces.append(edge_arr)
                unit_energies.append(float(np.mean(energies)))
                unit_connections.append(
                    [vertex_idx, terminal_vertex if terminal_vertex is not None else -1]
                )
                edges_per_vertex[vertex_idx] += 1
                if terminal_vertex is not None:
                    edges_per_vertex[terminal_vertex] += 1
                    existing_pairs.add(pair)

        payload = {
            "origin_index": vertex_idx,
            "traces": unit_traces,
            "connections": unit_connections,
            "energies": unit_energies,
        }
        atomic_joblib_dump(payload, units_dir / f"vertex_{vertex_idx:06d}.pkl")
        edges.extend(unit_traces)
        edge_connections.extend(unit_connections)
        edge_energies.extend(unit_energies)
        completed.add(vertex_idx)
        stage_controller.save_state({"last_completed_origin": vertex_idx})
        stage_controller.update(
            units_total=len(vertex_positions),
            units_completed=len(completed),
            substage="trace_origins",
            detail=f"Tracing origin {vertex_idx + 1}/{len(vertex_positions)}",
            resumed=bool(completed - {vertex_idx}),
        )

    return {
        "traces": edges,
        "connections": np.asarray(edge_connections, dtype=np.int32).reshape(-1, 2),
        "energies": np.asarray(edge_energies, dtype=np.float32),
        "vertex_positions": vertex_positions.astype(np.float32),
    }


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
    edges, connections, edge_energies, _, seen_pairs, completed = _load_edge_units(
        units_dir,
        len(vertex_positions),
    )
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
            "energies": unit_energies,
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
