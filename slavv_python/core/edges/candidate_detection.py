"""MATLAB-style vertex candidate scanning and selection."""

from __future__ import annotations

import numpy as np
from joblib import Parallel, delayed
from skimage.draw import ellipsoid
from typing_extensions import TypeAlias

from slavv_python.core.vertices.results import sort_vertex_order

Int16Array: TypeAlias = "np.ndarray"
Int32Array: TypeAlias = "np.ndarray"
Int64Array: TypeAlias = "np.ndarray"
Float32Array: TypeAlias = "np.ndarray"


def vertex_window_apothem(space_strel_apothem: int) -> int:
    """Normalize the MATLAB vertex neighborhood radius in voxel units."""
    return max(space_strel_apothem, 0)


def vertex_neighborhood_slices(
    pos: np.ndarray, apothem: int, shape: tuple[int, int, int]
) -> tuple[slice, slice, slice]:
    """Return clipped cube slices centered on ``pos``."""
    y, x, z = (int(coord) for coord in pos)
    return (
        slice(max(0, y - apothem), min(shape[0], y + apothem + 1)),
        slice(max(0, x - apothem), min(shape[1], x + apothem + 1)),
        slice(max(0, z - apothem), min(shape[2], z + apothem + 1)),
    )


def chunk_lattice_dimensions(
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
    values = lattice.tolist()
    return (int(values[0]), int(values[1]), int(values[2]))


def iter_overlapping_chunks(
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


def matlab_vertex_candidates_in_chunk(
    energy: np.ndarray,
    scale_indices: np.ndarray,
    energy_sign: float,
    energy_upper_bound: float,
    space_strel_apothem: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run MATLAB-style candidate scanning within one overlapped chunk."""
    apothem = vertex_window_apothem(space_strel_apothem)
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
    order = sort_vertex_order(candidate_positions, candidate_energies, energy.shape, energy_sign)
    candidate_positions = candidate_positions[order]

    accepted_positions: list[np.ndarray] = []
    accepted_scales: list[int] = []
    accepted_energies: list[float] = []

    for pos in candidate_positions:
        y, x, z = (int(coord) for coord in pos)
        if not active_mask[y, x, z]:
            continue

        slices = vertex_neighborhood_slices(pos, apothem, energy.shape)
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


def matlab_vertex_candidates(
    energy: np.ndarray,
    scale_indices: np.ndarray,
    energy_sign: float,
    energy_upper_bound: float,
    space_strel_apothem: int,
    strel_size_pixels: np.ndarray,
    max_voxels_per_node: float,
    *,
    n_jobs: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find MATLAB-style candidate vertices on the projected energy volume."""
    apothem = vertex_window_apothem(space_strel_apothem)
    overlap = (apothem, apothem, apothem)
    lattice_dims = chunk_lattice_dimensions(energy.shape, strel_size_pixels, max_voxels_per_node)

    def _worker(chunk_slice):
        chunk_positions, chunk_scales, chunk_energies = matlab_vertex_candidates_in_chunk(
            energy[chunk_slice],
            scale_indices[chunk_slice],
            energy_sign,
            energy_upper_bound,
            space_strel_apothem,
        )
        if len(chunk_positions) == 0:
            return None

        chunk_offset = np.array(
            [chunk_slice[0].start, chunk_slice[1].start, chunk_slice[2].start],
            dtype=np.int32,
        )
        return chunk_positions + chunk_offset, chunk_scales, chunk_energies

    results = Parallel(n_jobs=n_jobs)(
        delayed(_worker)(chunk_slice)
        for chunk_slice in iter_overlapping_chunks(energy.shape, lattice_dims, overlap)
    )

    accepted_positions: list[np.ndarray] = []
    accepted_scales: list[np.ndarray] = []
    accepted_energies: list[np.ndarray] = []

    for res in results:
        if res is not None:
            pos, scales, energies = res
            accepted_positions.append(pos)
            accepted_scales.append(scales)
            accepted_energies.append(energies)

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


def crop_vertices_matlab_style(
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


def ellipsoid_offsets(radii_pixels: np.ndarray) -> np.ndarray:
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


def choose_vertices_matlab_style(
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
        int(scale_index): ellipsoid_offsets(scaled_radii[scale_index])
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


__all__ = [
    "choose_vertices_matlab_style",
    "chunk_lattice_dimensions",
    "crop_vertices_matlab_style",
    "ellipsoid_offsets",
    "iter_overlapping_chunks",
    "matlab_vertex_candidates",
    "matlab_vertex_candidates_in_chunk",
    "vertex_neighborhood_slices",
    "vertex_window_apothem",
]
