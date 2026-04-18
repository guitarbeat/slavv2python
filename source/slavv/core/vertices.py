"""Vertex extraction and vertex-paint helpers for SLAVV."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from skimage.draw import ellipsoid
from typing_extensions import TypeAlias

from ..utils.safe_unpickle import safe_load

if TYPE_CHECKING:
    from slavv.runtime import StageController

logger = logging.getLogger(__name__)

Int16Array: TypeAlias = "np.ndarray"
Int32Array: TypeAlias = "np.ndarray"
Int64Array: TypeAlias = "np.ndarray"
Float32Array: TypeAlias = "np.ndarray"
Float64Array: TypeAlias = "np.ndarray"
BoolArray: TypeAlias = "np.ndarray"


def _vertex_window_apothem(space_strel_apothem: int) -> int:
    """Normalize the MATLAB vertex neighborhood radius in voxel units."""
    return max(space_strel_apothem, 0)


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
    linear_indices: Int64Array = (
        coords[:, 0] + coords[:, 1] * shape[0] + coords[:, 2] * shape[0] * shape[1]
    )
    return cast("np.ndarray", linear_indices)


def _sort_vertex_order(
    vertex_positions: np.ndarray,
    vertex_energies: np.ndarray,
    image_shape: tuple[int, int, int],
    energy_sign: float,
) -> np.ndarray:
    """Sort vertices like MATLAB: by energy, then by column-major linear index for ties."""
    if len(vertex_positions) == 0:
        empty_order: Int64Array = np.array([], dtype=np.int64)
        return cast("np.ndarray", empty_order)

    linear_indices = _matlab_linear_indices(vertex_positions, image_shape)
    if energy_sign < 0:
        sort_order: Int64Array = np.asarray(
            np.lexsort((linear_indices, vertex_energies)),
            dtype=np.int64,
        )
        return cast("np.ndarray", sort_order)
    sort_order = np.asarray(
        np.lexsort((linear_indices, -vertex_energies)),
        dtype=np.int64,
    )
    return cast("np.ndarray", sort_order)


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


def _coerce_radius_axes(
    lumen_radius_pixels: np.ndarray,
    lumen_radius_pixels_axes: np.ndarray | None,
) -> np.ndarray:
    """Normalize scale radii into a `(num_scales, 3)` axis-aware array."""
    if lumen_radius_pixels_axes is not None:
        axes: Float32Array = np.asarray(lumen_radius_pixels_axes, dtype=np.float32)
        if axes.ndim == 2 and axes.shape[1] == 3:
            return cast("np.ndarray", axes)

    radii = np.asarray(lumen_radius_pixels, dtype=np.float32).reshape(-1, 1)
    repeated_radii: Float32Array = np.repeat(radii, 3, axis=1)
    return cast("np.ndarray", repeated_radii)


def _chunk_lattice_dimensions(
    image_shape: tuple[int, int, int],
    strel_size_pixels: np.ndarray,
    max_voxels_per_node: float,
) -> tuple[int, int, int]:
    """Approximate MATLAB's 3D chunk lattice sizing."""
    target_voxels = max(max_voxels_per_node, 1.0)
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
    overlap_arr: Int64Array = np.asarray(overlap, dtype=np.int64)
    borders = [
        np.rint(np.linspace(0, image_shape[axis], lattice_dims[axis] + 1)).astype(np.int64)
        for axis in range(3)
    ]

    for y_index in range(lattice_dims[0]):
        y_start = max(int(borders[0][y_index] - overlap_arr[0]), 0)
        y_end = min(int(borders[0][y_index + 1] + overlap_arr[0]), image_shape[0])
        for x_index in range(lattice_dims[1]):
            x_start = max(int(borders[1][x_index] - overlap_arr[1]), 0)
            x_end = min(int(borders[1][x_index + 1] + overlap_arr[1]), image_shape[1])
            for z_index in range(lattice_dims[2]):
                z_start = max(int(borders[2][z_index] - overlap_arr[2]), 0)
                z_end = min(int(borders[2][z_index + 1] + overlap_arr[2]), image_shape[2])
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
        empty_offsets: np.ndarray = np.zeros((1, 3), dtype=np.int16)
        return empty_offsets

    mask = ellipsoid(float(radii[0]), float(radii[1]), float(radii[2]), spacing=(1.0, 1.0, 1.0))
    coords = np.column_stack(np.where(mask))
    center: Int64Array = np.asarray(mask.shape, dtype=np.int64) // 2
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
    scaled_radii: Float32Array = np.asarray(
        length_dilation_ratio * lumen_radius_pixels_axes,
        dtype=np.float32,
    )
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
            center: Int64Array = np.array(ellipsoid_mask.shape, dtype=np.int64) // 2
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

    candidate_data = safe_load(candidate_path)
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

    ordered = safe_load(cropped_path)
    vertex_positions = ordered["positions"]
    vertex_scales = ordered["scales"]
    vertex_energies = ordered["energies"]
    if len(vertex_positions) == 0:
        return _empty_vertices_result()

    if chosen_mask_path.exists():
        chosen_mask = safe_load(chosen_mask_path)
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
