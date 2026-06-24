"""MATLAB-style vertex candidate scanning, crop, and conflict-aware selection."""

from __future__ import annotations

from typing import cast

import numpy as np
from joblib import Parallel, delayed
from skimage.draw import ellipsoid
from typing_extensions import TypeAlias

from slavv_python.pipeline.policy import PipelinePolicy
from slavv_python.pipeline.vertices.results import sort_vertex_order
from slavv_python.utils.lattice import compute_chunking_lattice, iter_chunk_slices

# Backward-compatible aliases for edge-stage re-exports.
chunk_lattice_dimensions = compute_chunking_lattice
iter_overlapping_chunks = iter_chunk_slices

try:
    from numba import njit
except ImportError:
    njit = None

Int16Array: TypeAlias = "np.ndarray"
Int32Array: TypeAlias = "np.ndarray"
Int64Array: TypeAlias = "np.ndarray"
Float32Array: TypeAlias = "np.ndarray"
Float64Array: TypeAlias = "np.ndarray"
BoolArray: TypeAlias = "np.ndarray"


def vertex_window_apothem(space_strel_apothem: int) -> int:
    """Normalize the vertex neighborhood radius in voxel units."""
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


def matlab_vertex_candidates_in_chunk(
    energy: np.ndarray,
    scale_indices: np.ndarray,
    energy_sign: float,
    energy_upper_bound: float,
    space_strel_apothem: int,
    policy: PipelinePolicy,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find candidate vertices within one overlapped chunk."""
    apothem = vertex_window_apothem(space_strel_apothem)
    interior_mask = np.zeros(energy.shape, dtype=bool)
    if apothem == 0:
        interior_mask[:] = True
    else:
        # Indexing with -0 or 0 is problematic, so we use None for the end if 0.
        interior_mask[
            apothem : -apothem or None,
            apothem : -apothem or None,
            apothem : -apothem or None,
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
            np.empty((0,), dtype=policy.precision),
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
            window_extreme: float = float(np.nanmin(window))
            is_vertex = np.isfinite(window_extreme) and energy[y, x, z] <= window_extreme
        else:
            window_extreme = float(np.nanmax(window))
            is_vertex = np.isfinite(window_extreme) and energy[y, x, z] >= window_extreme

        if is_vertex:
            accepted_positions.append(pos.astype(np.int32, copy=False))
            accepted_scales.append(int(scale_indices[y, x, z]))
            accepted_energies.append(float(energy[y, x, z]))

        active_mask[slices] = False

    return (
        np.asarray(accepted_positions, dtype=np.int32).reshape(-1, 3),
        np.asarray(accepted_scales, dtype=np.int16),
        np.asarray(accepted_energies, dtype=policy.precision),
    )


def matlab_vertex_candidates(
    energy: np.ndarray,
    scale_indices: np.ndarray,
    energy_sign: float,
    energy_upper_bound: float,
    space_strel_apothem: int,
    strel_size_pixels: np.ndarray,
    max_voxels_per_node: float,
    policy: PipelinePolicy,
    *,
    n_jobs: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find candidate vertices on the projected energy volume."""
    apothem = vertex_window_apothem(space_strel_apothem)
    overlap = (apothem, apothem, apothem)
    lattice_dims, _ = compute_chunking_lattice(
        energy.shape, strel_size_pixels, max_voxels_per_node, policy
    )

    def _worker(chunk_slice):
        chunk_positions, chunk_scales, chunk_energies = matlab_vertex_candidates_in_chunk(
            energy[chunk_slice],
            scale_indices[chunk_slice],
            energy_sign,
            energy_upper_bound,
            space_strel_apothem,
            policy,
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
        for chunk_slice in iter_chunk_slices(energy.shape, lattice_dims, overlap, policy)
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
            np.empty((0,), dtype=policy.precision),
        )

    return (
        np.vstack(accepted_positions).astype(np.int32, copy=False),
        np.concatenate(accepted_scales).astype(np.int16, copy=False),
        np.concatenate(accepted_energies).astype(policy.precision, copy=False),
    )


def crop_vertices_matlab_style(
    vertex_positions: np.ndarray,
    vertex_scales: np.ndarray,
    vertex_energies: np.ndarray,
    image_shape: tuple[int, int, int],
    lumen_radius_pixels_axes: np.ndarray,
    length_dilation_ratio: float,
    policy: PipelinePolicy,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Crop candidate vertices against image bounds and extreme scales."""
    if len(vertex_positions) == 0:
        return (
            np.empty((0, 3), dtype=np.int32),
            np.empty((0,), dtype=np.int16),
            np.empty((0,), dtype=policy.precision),
        )

    scale_indices = policy.round(vertex_scales).astype(np.int64)
    scaled_radii = policy.round(
        length_dilation_ratio * lumen_radius_pixels_axes[scale_indices]
    ).astype(np.int64)
    positions = policy.round(vertex_positions).astype(np.int64)

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
        vertex_energies[keep].astype(policy.precision, copy=False),
    )


def ellipsoid_offsets(radii_pixels: np.ndarray, policy: PipelinePolicy | None = None) -> np.ndarray:
    """Construct centered voxel offsets for a scale-specific ellipsoid."""
    policy = policy or PipelinePolicy(np.float64, "matlab", "half-up", "incremental")
    radii = np.maximum(policy.round(radii_pixels).astype(np.int16), 0)
    if np.all(radii == 0):
        return np.zeros((1, 3), dtype=np.int16)

    mask = ellipsoid(float(radii[0]), float(radii[1]), float(radii[2]), spacing=(1.0, 1.0, 1.0))
    coords = np.column_stack(np.where(mask))
    center = np.asarray(mask.shape, dtype=np.int64) // 2
    return cast("np.ndarray", (coords - center).astype(np.int16, copy=False))


def _choose_vertices_loop_python(
    painted_image: np.ndarray,
    vertex_positions: np.ndarray,
    vertex_scales: np.ndarray,
    all_offsets: np.ndarray,
    template_starts: np.ndarray,
    template_ends: np.ndarray,
    chosen_mask: np.ndarray,
    start_index: int,
    end_index: int,
    image_shape: tuple[int, int, int],
    rounding_mode: str = "half-up",
) -> None:
    """Pure-Python fallback for vertex selection loop."""
    for i in range(start_index, end_index):
        scale = vertex_scales[i]
        pos = vertex_positions[i]

        if rounding_mode == "half-up":
            cy = int(np.floor(pos[0] + 0.5))
            cx = int(np.floor(pos[1] + 0.5))
            cz = int(np.floor(pos[2] + 0.5))
        else:
            cy = int(np.round(pos[0]))
            cx = int(np.round(pos[1]))
            cz = int(np.round(pos[2]))

        t_start = template_starts[scale]
        t_end = template_ends[scale]
        occupied = False
        for j in range(t_start, t_end):
            y: int = int(cy + all_offsets[j, 0])
            x: int = int(cx + all_offsets[j, 1])
            z: int = int(cz + all_offsets[j, 2])

            if (
                0 <= y < image_shape[0]
                and 0 <= x < image_shape[1]
                and 0 <= z < image_shape[2]
                and painted_image[y, x, z]
            ):
                occupied = True
                break

        if not occupied:
            chosen_mask[i] = True
            for j in range(t_start, t_end):
                y = int(cy + all_offsets[j, 0])
                x = int(cx + all_offsets[j, 1])
                z = int(cz + all_offsets[j, 2])

                if 0 <= y < image_shape[0] and 0 <= x < image_shape[1] and 0 <= z < image_shape[2]:
                    painted_image[y, x, z] = True
        else:
            chosen_mask[i] = False


if njit is not None:
    _choose_vertices_loop_numba = njit(cache=False)(_choose_vertices_loop_python)
else:
    _choose_vertices_loop_numba = None


def choose_vertices_matlab_style(
    vertex_positions: np.ndarray,
    vertex_scales: np.ndarray,
    image_shape: tuple[int, int, int],
    lumen_radius_pixels_axes: np.ndarray,
    length_dilation_ratio: float,
    policy: PipelinePolicy,
    start_index: int = 0,
    end_index: int | None = None,
    chosen_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Choose non-overlapping vertices with paint-and-check semantics."""
    n_vertices = len(vertex_positions)
    if chosen_mask is None:
        chosen_mask = np.zeros(n_vertices, dtype=bool)
    else:
        chosen_mask = np.asarray(chosen_mask, dtype=bool).copy()

    if end_index is None:
        end_index = n_vertices

    scale_indices = policy.round(vertex_scales).astype(np.int64)
    scaled_radii = length_dilation_ratio * lumen_radius_pixels_axes

    # Prepare templates
    unique_scales = np.unique(scale_indices)
    max_scale = int(np.max(unique_scales)) if unique_scales.size > 0 else -1
    template_starts: np.ndarray = np.zeros(max_scale + 1, dtype=np.int32)
    template_ends: np.ndarray = np.zeros(max_scale + 1, dtype=np.int32)

    all_offsets_list = []
    current_offset = 0
    for scale_index in unique_scales:
        offsets = ellipsoid_offsets(scaled_radii[scale_index], policy)
        all_offsets_list.append(offsets)
        template_starts[scale_index] = current_offset
        current_offset += len(offsets)
        template_ends[scale_index] = current_offset

    if all_offsets_list:
        all_offsets = np.vstack(all_offsets_list).astype(np.int16)
    else:
        all_offsets = np.empty((0, 3), dtype=np.int16)

    painted_image: np.ndarray = np.zeros(image_shape, dtype=bool)

    # Pre-chosen vertices
    for index in np.flatnonzero(chosen_mask[:start_index]):
        scale = scale_indices[index]
        pos = vertex_positions[index]
        cy, cx, cz = policy.round(pos).astype(np.int32)

        t_start = template_starts[scale]
        t_end = template_ends[scale]
        for j in range(t_start, t_end):
            y = cy + all_offsets[j, 0]
            x = cx + all_offsets[j, 1]
            z = cz + all_offsets[j, 2]
            if 0 <= y < image_shape[0] and 0 <= x < image_shape[1] and 0 <= z < image_shape[2]:
                painted_image[y, x, z] = True

    # Selection loop
    loop_impl = _choose_vertices_loop_numba or _choose_vertices_loop_python
    loop_impl(
        painted_image,
        vertex_positions.astype(np.float64),
        scale_indices.astype(np.int32),
        all_offsets,
        template_starts,
        template_ends,
        chosen_mask,
        start_index,
        min(end_index, n_vertices),
        image_shape,
        rounding_mode=policy.rounding_mode,
    )

    return chosen_mask


__all__ = [
    "choose_vertices_matlab_style",
    "crop_vertices_matlab_style",
    "ellipsoid_offsets",
    "matlab_vertex_candidates",
    "matlab_vertex_candidates_in_chunk",
    "vertex_neighborhood_slices",
    "vertex_window_apothem",
]
