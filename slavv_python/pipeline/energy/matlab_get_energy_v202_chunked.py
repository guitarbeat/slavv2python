"""MATLAB port: ``get_energy_V202.m`` — octave-chunked exact energy field.

Role: chunk lattice traversal, per-octave scale winner reduction, ``interp3``
upsampling, and cross-chunk master merge. Also hosts ``get_chunking_lattice_V190``
and ``get_starts_and_counts_V200`` helpers from the same MATLAB lineage.

MATLAB source: ``external/Vectorization-Public/source/get_energy_V202.m``
Uses: ``matlab_energy_filter_v200.py`` for per-scale filter math
"""

# The per-octave Joblib worker is intentionally defined inside the octave loop so
# each worker captures the current MATLAB mesh context.
# ruff: noqa: B023

from __future__ import annotations

import gc
import json
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from joblib import Parallel, delayed

try:
    from numba import njit, prange
except ImportError:
    njit = None
    prange = range

from slavv_python.pipeline.energy import matlab_energy_filter_v200 as native_hessian
from slavv_python.pipeline.energy.matlab_principal_energy import compute_principal_energy

if TYPE_CHECKING:
    from collections.abc import Callable


def get_chunking_lattice_v190(
    strel_size_in_pixels: np.ndarray,
    max_voxels_per_node: int | float,
    size_of_image: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Replicate MATLAB get_chunking_lattice_V190."""
    target_voxel_per_chunk = float(max_voxels_per_node)
    target_chunk_characteristic_length = target_voxel_per_chunk ** (1.0 / 3.0)

    unit_volume_voxel_aspect_ratio = strel_size_in_pixels / (
        np.prod(strel_size_in_pixels) ** (1.0 / 3.0)
    )
    target_chunk_dimensions = target_chunk_characteristic_length * unit_volume_voxel_aspect_ratio

    chunk_lattice_dimensions = _matlab_uint16_cast(
        np.maximum(size_of_image / target_chunk_dimensions, 1.0)
    )

    number_of_chunks_in_lattice = int(np.prod(chunk_lattice_dimensions))
    return chunk_lattice_dimensions, number_of_chunks_in_lattice


def _matlab_uint16_cast(values: np.ndarray) -> np.ndarray:
    """Match MATLAB's positive double -> uint16 conversion."""
    rounded = np.floor(np.asarray(values, dtype=float) + 0.5)
    return cast("np.ndarray", np.clip(rounded, 0, np.iinfo(np.uint16).max).astype(np.uint16))


def get_starts_and_counts_v200(
    chunk_lattice_dimensions: np.ndarray,
    chunk_overlap_in_pixels: np.ndarray,
    size_of_image: np.ndarray,
    resolution_factors: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Replicate MATLAB get_starts_and_counts_V200 with uint16 saturation arithmetic."""

    def get_borders(size, lat):
        pts = np.linspace(0, float(size), int(lat) + 1)
        return _matlab_uint16_cast(pts)

    y_writing_borders = get_borders(size_of_image[0], chunk_lattice_dimensions[0])
    x_writing_borders = get_borders(size_of_image[1], chunk_lattice_dimensions[1])
    z_writing_borders = get_borders(size_of_image[2], chunk_lattice_dimensions[2])

    def sat_sub(a, b):
        return np.clip(a.astype(np.int32) - b.astype(np.int32), 0, 65535).astype(np.uint16)

    def sat_add(a, b):
        return np.clip(a.astype(np.int32) + b.astype(np.int32), 0, 65535).astype(np.uint16)

    y_reading_starts = sat_sub(y_writing_borders[:-1], chunk_overlap_in_pixels[0]) + 1
    x_reading_starts = sat_sub(x_writing_borders[:-1], chunk_overlap_in_pixels[1]) + 1
    z_reading_starts = sat_sub(z_writing_borders[:-1], chunk_overlap_in_pixels[2]) + 1

    y_reading_ends = np.minimum(
        sat_add(y_writing_borders[1:], chunk_overlap_in_pixels[0]), int(size_of_image[0])
    )
    x_reading_ends = np.minimum(
        sat_add(x_writing_borders[1:], chunk_overlap_in_pixels[1]), int(size_of_image[1])
    )
    z_reading_ends = np.minimum(
        sat_add(z_writing_borders[1:], chunk_overlap_in_pixels[2]), int(size_of_image[2])
    )

    y_reading_counts = sat_sub(y_reading_ends, y_reading_starts - 1)
    x_reading_counts = sat_sub(x_reading_ends, x_reading_starts - 1)
    z_reading_counts = sat_sub(z_reading_ends, z_reading_starts - 1)

    def adjust_last_count(counts, rf):
        counts[-1] = 1 + int(rf) * int(np.floor((float(counts[-1]) - 1.0) / float(rf)))
        return counts

    y_reading_counts = adjust_last_count(y_reading_counts, resolution_factors[0])
    x_reading_counts = adjust_last_count(x_reading_counts, resolution_factors[1])
    z_reading_counts = adjust_last_count(z_reading_counts, resolution_factors[2])

    y_reading_starts[-1] = sat_sub(y_reading_ends[-1], y_reading_counts[-1] - 1)
    x_reading_starts[-1] = sat_sub(x_reading_ends[-1], x_reading_counts[-1] - 1)
    z_reading_starts[-1] = sat_sub(z_reading_ends[-1], z_reading_counts[-1] - 1)

    y_writing_starts = y_writing_borders[:-1].astype(np.int32) + 1
    x_writing_starts = x_writing_borders[:-1].astype(np.int32) + 1
    z_writing_starts = z_writing_borders[:-1].astype(np.int32) + 1

    y_writing_ends = y_writing_borders[1:].astype(np.int32)
    x_writing_ends = x_writing_borders[1:].astype(np.int32)
    z_writing_ends = z_writing_borders[1:].astype(np.int32)

    y_writing_counts = y_writing_ends - y_writing_starts + 1
    x_writing_counts = x_writing_ends - x_writing_starts + 1
    z_writing_counts = z_writing_ends - z_writing_starts + 1

    y_offsets = sat_sub(y_writing_starts, y_reading_starts)
    x_offsets = sat_sub(x_writing_starts, x_reading_starts)
    z_offsets = sat_sub(z_writing_starts, z_reading_starts)

    return (
        y_reading_starts.astype(float),
        x_reading_starts.astype(float),
        z_reading_starts.astype(float),
        y_reading_counts.astype(float),
        x_reading_counts.astype(float),
        z_reading_counts.astype(float),
        y_writing_starts.astype(float),
        x_writing_starts.astype(float),
        z_writing_starts.astype(float),
        y_writing_counts.astype(float),
        x_writing_counts.astype(float),
        z_writing_counts.astype(float),
        y_offsets.astype(float),
        x_offsets.astype(float),
        z_offsets.astype(float),
    )


def _interp3_matlab_linear_inf(
    volume: np.ndarray,
    coords: np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    cval: float = 0.0,
) -> np.ndarray:
    """Linear interp3-compatible interpolation that propagates positive-weight Inf. Supports sparse coords."""
    y = np.asarray(coords[0], dtype=np.float64)
    x = np.asarray(coords[1], dtype=np.float64)
    z = np.asarray(coords[2], dtype=np.float64)

    out_shape = np.broadcast(y, x, z).shape

    y0 = np.floor(y).astype(np.int64)
    x0 = np.floor(x).astype(np.int64)
    z0 = np.floor(z).astype(np.int64)
    fy = y - y0
    fx = x - x0
    fz = z - z0

    out = np.zeros(out_shape, dtype=np.float64)
    has_inf = np.zeros(out_shape, dtype=bool)
    shape_y, shape_x, shape_z = volume.shape

    for dy in (0, 1):
        iy = y0 + dy
        wy = fy if dy else 1.0 - fy
        valid_y = (iy >= 0) & (iy < shape_y)
        for dx in (0, 1):
            ix = x0 + dx
            wx = fx if dx else 1.0 - fx
            valid_xy = valid_y & (ix >= 0) & (ix < shape_x)
            for dz in (0, 1):
                iz = z0 + dz
                wz = fz if dz else 1.0 - fz
                weight = wy * wx * wz
                positive_weight = weight > 0.0
                valid = valid_xy & (iz >= 0) & (iz < shape_z)
                valid_positive = valid & positive_weight

                if np.any(valid_positive):
                    iy_b = np.broadcast_to(iy, out_shape)[valid_positive]
                    ix_b = np.broadcast_to(ix, out_shape)[valid_positive]
                    iz_b = np.broadcast_to(iz, out_shape)[valid_positive]
                    w_b = np.broadcast_to(weight, out_shape)[valid_positive]

                    vals = volume[iy_b, ix_b, iz_b]
                    has_inf_mask = np.isposinf(vals)
                    has_inf[valid_positive] |= has_inf_mask
                    finite_mask = np.isfinite(vals)

                    update_mask = valid_positive.copy()
                    update_mask[valid_positive] = finite_mask

                    out[update_mask] += w_b[finite_mask] * vals[finite_mask]

                invalid_positive = (~valid) & positive_weight
                if cval is not None and np.any(invalid_positive):
                    w_inv = np.broadcast_to(weight, out_shape)[invalid_positive]
                    out[invalid_positive] += w_inv * cval

    out[has_inf] = np.inf
    inside = (
        (y >= 0.0)
        & (y <= float(shape_y - 1))
        & (x >= 0.0)
        & (x <= float(shape_x - 1))
        & (z >= 0.0)
        & (z <= float(shape_z - 1))
    )
    if cval is not None and np.any(~inside):
        out[~inside] = cval
    return cast("np.ndarray", out)


def _matlab_coarse_local_slices(
    *,
    offsets: tuple[int, int, int],
    write_counts: tuple[int, int, int],
    strides: tuple[int, int, int],
    padded_shape: tuple[int, ...],
) -> tuple[slice, slice, slice]:
    """Return zero-based [Y, X, Z] coarse slices matching ``get_energy_V202`` local_ranges."""
    slices: list[slice] = []
    for offset, write_count, stride, padded_extent in zip(
        offsets, write_counts, strides, padded_shape, strict=True
    ):
        local_start = max(0, int(np.floor(offset / stride)))
        local_stop = 1 + int(np.ceil((offset + write_count - 1) / stride))
        if local_stop > int(padded_extent):
            raise ValueError(
                "MATLAB requested coarse support exceeds padded FFT grid: "
                f"stop={local_stop}, padded_extent={padded_extent}, "
                f"offset={offset}, write_count={write_count}, stride={stride}"
            )
        slices.append(slice(local_start, local_stop))
    return slices[0], slices[1], slices[2]


_OVERRIDES_PATH = Path(__file__).with_name("matlab_linspace_overrides.json")
_RANDOM_OVERRIDES_PATH = Path(__file__).with_name("matlab_random_linspace_reference.json")


def _load_linspace_override_payload(path: Path) -> dict[tuple[int, int, int, int], np.ndarray]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    meshes: dict[tuple[int, int, int, int], np.ndarray] = {}
    for key, values in payload.items():
        offset_s, stride_s, count_s, local_start_s = key.split(",")
        meshes[(int(offset_s), int(stride_s), int(count_s), int(local_start_s))] = np.asarray(
            values, dtype=np.float64
        )
    return meshes


@lru_cache(maxsize=1)
def _matlab_linspace_override_meshes() -> dict[tuple[int, int, int, int], np.ndarray]:
    """Full MATLAB R2019a meshes for contexts that diverge from the raw formula."""
    meshes = _load_linspace_override_payload(_OVERRIDES_PATH)
    meshes.update(_load_linspace_override_payload(_RANDOM_OVERRIDES_PATH))
    return meshes


def _snap_mesh_to_grid_integers(mesh: np.ndarray) -> np.ndarray:
    """Snap upsample-mesh coordinates that are within ~1 ULP of an integer to it.

    A fine voxel aligned with a coarse sample maps to an exact integer coordinate;
    MATLAB ``linspace`` lands on it exactly, but the float barycentric formula drifts
    ~1e-15. At a coarse-cell boundary that drift floors ``interp3``'s base cell into
    the neighbor (which may be invalid/Inf), collapsing a valid energy to 0 -- the
    octave-3/4 ``scale_indices`` divergence. Mesh steps are ``i/stride`` (stride <= 20
    here), so genuine non-integer samples sit >= 1/20 = 0.05 from any integer; a 1e-9
    snap is therefore unambiguous and a no-op for rf==1 (octave-1) meshes, which are
    already integer-exact. See docs/solutions/parity/canonical-energy-high-octave-divergence.md.
    """
    nearest = np.round(mesh)
    snap = np.abs(mesh - nearest) < 1e-9
    if np.any(snap):
        mesh = mesh.copy()
        mesh[snap] = nearest[snap]
    return mesh


def _matlab_zero_based_linspace_raw(
    offset: int, stride: int, count: int, local_start: int
) -> np.ndarray:
    """Base linspace formula without MATLAB R2019a ULP corrections."""
    if count <= 0:
        return np.empty(0, dtype=np.float64)
    x1 = 1.0 + float(offset) / float(stride) - float(local_start)
    x2 = x1 + float(count - 1) / float(stride)
    if count == 1:
        return _snap_mesh_to_grid_integers(np.array([x2 - 1.0], dtype=np.float64))
    i: np.ndarray = np.arange(count, dtype=np.float64)
    mesh = ((count - 1 - i) * x1 + i * x2) / (count - 1) - 1.0
    return _snap_mesh_to_grid_integers(cast("np.ndarray", mesh))


def _matlab_zero_based_linspace(
    offset: int, stride: int, count: int, local_start: int
) -> np.ndarray:
    """Return MATLAB ``linspace(1 + offset/rf - local_start, ..., count)`` in zero-based coordinates."""
    key = (offset, stride, count, local_start)
    override = _matlab_linspace_override_meshes().get(key)
    if override is not None:
        if override.shape[0] != count:
            raise ValueError(f"linspace override length mismatch for {key}")
        return override.copy()
    return _matlab_zero_based_linspace_raw(offset, stride, count, local_start)


def compute_exact_parity_energy_chunked(
    image: np.ndarray,
    config: dict[str, Any],
    progress_callback: Callable[[int, int, int, int], None] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Compute exact-route energy using MATLAB octave chunk and mesh rules."""

    n_jobs = max(1, int(config.get("n_jobs", 2)))
    image_shape = np.asarray(image.shape, dtype=float)

    energy_3d = np.zeros(image.shape, dtype=np.float64)
    scale_indices = np.full(image.shape, -1, dtype=np.int16)

    octave_at_scales = config["octave_at_scales"]
    octave_range = np.unique(octave_at_scales)

    microns_per_voxel = np.asarray(config["microns_per_voxel"], dtype=float)
    pixels_per_sigma_PSF = np.asarray(config["pixels_per_sigma_PSF"], dtype=float)
    lumen_radius_microns = np.asarray(config["lumen_radius_microns"], dtype=float)

    total_chunk_units = 0
    for planned_octave in octave_range:
        planned_scales = np.where(octave_at_scales == planned_octave)[0]
        if len(planned_scales) == 0:
            continue
        planned_rf = np.asarray(config["scale_resolution_factors"][planned_scales[0]], dtype=float)
        planned_shape = np.array([image_shape[1], image_shape[2], image_shape[0]], dtype=float)
        planned_rf_matlab = np.array([planned_rf[1], planned_rf[2], planned_rf[0]], dtype=float)
        planned_microns = microns_per_voxel * planned_rf
        _, planned_chunks = get_chunking_lattice_v190(
            1.0 / np.array([planned_microns[1], planned_microns[2], planned_microns[0]]),
            float(config["max_voxels"]),
            np.round(planned_shape / planned_rf_matlab),
        )
        total_chunk_units += int(planned_chunks)
    completed_chunk_units = 0

    for current_octave in octave_range:
        scale_indices_at_octave = np.where(octave_at_scales == current_octave)[0]
        if len(scale_indices_at_octave) == 0:
            continue

        rf = np.asarray(config["scale_resolution_factors"][scale_indices_at_octave[0]], dtype=float)
        # get_starts_and_counts_V200 uses MATLAB axis order (Y, X, Z); working image is (Z, Y, X).
        matlab_image_shape = np.array(
            [image_shape[1], image_shape[2], image_shape[0]],
            dtype=float,
        )
        rf_matlab = np.array([rf[1], rf[2], rf[0]], dtype=float)

        largest_scale_idx = scale_indices_at_octave[-1]
        largest_radius_microns = lumen_radius_microns[largest_scale_idx]
        largest_pixels_per_radius = largest_radius_microns / microns_per_voxel

        approx_size = np.round(matlab_image_shape / rf_matlab)
        microns_per_pixel = microns_per_voxel * rf
        microns_per_pixel_matlab = np.array(
            [microns_per_pixel[1], microns_per_pixel[2], microns_per_pixel[0]],
            dtype=float,
        )
        voxel_aspect_ratio = 1.0 / microns_per_pixel_matlab

        chunk_lattice_dimensions, number_of_chunks = get_chunking_lattice_v190(
            voxel_aspect_ratio,
            float(config["max_voxels"]),
            approx_size,
        )

        chunk_overlap_vector = np.ceil(
            6.0 * np.sqrt(pixels_per_sigma_PSF**2 + largest_pixels_per_radius**2)
        ).astype(np.int32)
        chunk_overlap_matlab = chunk_overlap_vector[[1, 2, 0]]

        res_starts_counts = get_starts_and_counts_v200(
            chunk_lattice_dimensions,
            chunk_overlap_matlab,
            matlab_image_shape,
            rf_matlab,
        )

        y_read_starts = res_starts_counts[0]
        x_read_starts = res_starts_counts[1]
        z_read_starts = res_starts_counts[2]

        y_read_counts = res_starts_counts[3]
        x_read_counts = res_starts_counts[4]
        z_read_counts = res_starts_counts[5]

        y_write_starts = res_starts_counts[6]
        x_write_starts = res_starts_counts[7]
        z_write_starts = res_starts_counts[8]

        y_write_counts = res_starts_counts[9]
        x_write_counts = res_starts_counts[10]
        z_write_counts = res_starts_counts[11]

        y_offset = res_starts_counts[12]
        x_offset = res_starts_counts[13]
        z_offset = res_starts_counts[14]

        def _process_chunk(
            chunk_idx: int,
        ) -> tuple[int, tuple[slice, slice, slice, np.ndarray, np.ndarray]]:
            # Fortran unraveling matching MATLAB ind2sub on (Y, X, Z) lattice.
            y_idx, x_idx, z_idx = np.unravel_index(chunk_idx, chunk_lattice_dimensions, order="F")

            py_z_start = int(z_read_starts[z_idx]) - 1
            py_y_start = int(y_read_starts[y_idx]) - 1
            py_x_start = int(x_read_starts[x_idx]) - 1

            py_z_count = int(z_read_counts[z_idx])
            py_y_count = int(y_read_counts[y_idx])
            py_x_count = int(x_read_counts[x_idx])

            # ``rf`` is in the raw [Z, Y, X] input frame. The chunk is
            # transposed to MATLAB [Y, X, Z] only after this strided slice.
            stride_z = int(rf[0])
            stride_y = int(rf[1])
            stride_x = int(rf[2])

            original_chunk_zyx = image[
                py_z_start : py_z_start + py_z_count : stride_z,
                py_y_start : py_y_start + py_y_count : stride_y,
                py_x_start : py_x_start + py_x_count : stride_x,
            ]
            # Transpose the chunk from [Z, Y, X] to [Y, X, Z] to match MATLAB's internal memory layout.
            original_chunk = np.transpose(original_chunk_zyx, (1, 2, 0)).copy(order="F")
            original_chunk = original_chunk.astype(np.float64, copy=False)

            padded_chunk = native_hessian._fourier_transform_input(original_chunk)
            padded_shape = padded_chunk.shape
            chunk_dft = np.fft.fftn(padded_chunk.astype(np.float64, copy=False))

            # Pre-compute pixel frequency meshes for the padded chunk once
            pixel_freq_meshes = native_hessian._pixel_frequency_meshes(padded_chunk.shape)

            w_count_z = int(z_write_counts[z_idx])
            w_count_y = int(y_write_counts[y_idx])
            w_count_x = int(x_write_counts[x_idx])

            off_z = int(z_offset[z_idx])
            off_y = int(y_offset[y_idx])
            off_x = int(x_offset[x_idx])

            y_local, x_local, z_local = _matlab_coarse_local_slices(
                offsets=(off_y, off_x, off_z),
                write_counts=(w_count_y, w_count_x, w_count_z),
                strides=(stride_y, stride_x, stride_z),
                padded_shape=padded_shape,
            )
            l_start_y = y_local.start or 0
            l_start_x = x_local.start or 0
            l_start_z = z_local.start or 0

            mesh_y = _matlab_zero_based_linspace(off_y, stride_y, w_count_y, l_start_y)
            mesh_x = _matlab_zero_based_linspace(off_x, stride_x, w_count_x, l_start_x)
            mesh_z = _matlab_zero_based_linspace(off_z, stride_z, w_count_z, l_start_z)

            # [Y, X, Z] mesh for interpolation (sparse to save memory)
            mesh_coords: tuple[np.ndarray, np.ndarray, np.ndarray] = np.meshgrid(
                mesh_y, mesh_x, mesh_z, indexing="ij", sparse=True
            )
            coords_grid = mesh_coords

            # Accumulators in [Y, X, Z] order with Fortran contiguity
            chunk_best_energy: np.ndarray = np.full(
                (w_count_y, w_count_x, w_count_z), 0.0, dtype=np.float64, order="F"
            )
            chunk_best_scale_sub_idx: np.ndarray = np.full(
                (w_count_y, w_count_x, w_count_z), -1, dtype=np.int16, order="F"
            )

            for s_sub_idx, s_idx in enumerate(scale_indices_at_octave):
                radius_of_lumen_in_microns = lumen_radius_microns[s_idx]
                pixels_per_sigma_psf_at_oct = pixels_per_sigma_PSF / rf

                matching_kernel_dft, derivative_weights = native_hessian._matching_kernel_dft(
                    pixel_freq_meshes,
                    radius_of_lumen_in_microns=radius_of_lumen_in_microns,
                    microns_per_pixel=microns_per_pixel_matlab,
                    pixels_per_sigma_psf=pixels_per_sigma_psf_at_oct[[1, 2, 0]],
                    gaussian_to_ideal_ratio=float(config["gaussian_to_ideal_ratio"]),
                    spherical_to_annular_ratio=float(config["spherical_to_annular_ratio"]),
                )

                filtered_chunk_dft = matching_kernel_dft * chunk_dft
                del matching_kernel_dft

                coarse_shape = (
                    y_local.stop - y_local.start,
                    x_local.stop - x_local.start,
                    z_local.stop - z_local.start,
                )

                # All intermediates are [Y, X, Z] with Fortran order
                # Compute derivative kernels one-by-one to minimize peak memory
                curvatures_local = np.empty((6, *coarse_shape), dtype=np.float64, order="F")
                for k_idx in range(6):
                    k_dft = native_hessian._derivative_kernel_dft_single(
                        pixel_freq_meshes, derivative_weights, k_idx, is_curvature=True
                    )
                    full_ifft = native_hessian._ifftn_matlab_symmetric(k_dft * filtered_chunk_dft)
                    curvatures_local[k_idx] = full_ifft[y_local, x_local, z_local]
                    del k_dft, full_ifft

                gradient_local = np.empty((3, *coarse_shape), dtype=np.float64, order="F")
                for k_idx in range(3):
                    k_dft = native_hessian._derivative_kernel_dft_single(
                        pixel_freq_meshes, derivative_weights, k_idx, is_curvature=False
                    )
                    full_ifft = native_hessian._ifftn_matlab_symmetric(k_dft * filtered_chunk_dft)
                    gradient_local[k_idx] = full_ifft[y_local, x_local, z_local]
                    del k_dft, full_ifft

                # Explicitly delete DFT products
                del filtered_chunk_dft

                laplacian_chunk = curvatures_local[0] + curvatures_local[1] + curvatures_local[2]
                valid_voxels = laplacian_chunk < 0
                coarse_energy = np.full(coarse_shape, np.inf, dtype=np.float64)

                if np.any(valid_voxels):
                    grad_valid = np.stack(
                        [
                            gradient_local[0][valid_voxels],
                            gradient_local[1][valid_voxels],
                            gradient_local[2][valid_voxels],
                        ],
                        axis=1,
                    )
                    curvatures_valid = np.stack(
                        [
                            curvatures_local[0][valid_voxels],
                            curvatures_local[1][valid_voxels],
                            curvatures_local[2][valid_voxels],
                            curvatures_local[3][valid_voxels],
                            curvatures_local[4][valid_voxels],
                            curvatures_local[5][valid_voxels],
                        ],
                        axis=1,
                    )
                    energy_valid = compute_principal_energy(
                        grad_valid,
                        curvatures_valid,
                        energy_sign=float(config.get("energy_sign", -1.0)),
                    )
                    coarse_energy[valid_voxels] = energy_valid

                    del grad_valid, curvatures_valid, energy_valid
                # Free local cropped arrays
                del curvatures_local, gradient_local

                coarse_energy[~np.isfinite(coarse_energy)] = np.inf
                coarse_energy[coarse_energy >= 0] = np.inf

                upsampled = _interp3_matlab_linear_inf(coarse_energy, coords_grid)
                del coarse_energy
                upsampled[(~np.isfinite(upsampled)) | (upsampled >= 0)] = 0.0

                if s_sub_idx == 0:
                    chunk_best_energy = upsampled
                    chunk_best_scale_sub_idx = np.zeros_like(chunk_best_scale_sub_idx)
                else:
                    is_better = upsampled < chunk_best_energy
                    chunk_best_energy = np.where(is_better, upsampled, chunk_best_energy)
                    chunk_best_scale_sub_idx[is_better] = s_sub_idx

                del upsampled

            # Energy and scale are accumulated in [Y, X, Z] order internally.
            # We transpose them back to [Z, Y, X] to match the master volume's axis order.
            chunk_energy_min = chunk_best_energy.transpose(2, 0, 1)
            chunk_scale_min = chunk_best_scale_sub_idx.transpose(2, 0, 1)
            chunk_scale_min[chunk_energy_min >= 0.0] = -1

            prev_scales_count = int(np.sum(octave_at_scales < current_octave))
            valid_scale = chunk_scale_min >= 0
            chunk_scale_min = chunk_scale_min.astype(np.int16)
            chunk_scale_min[valid_scale] += prev_scales_count

            py_z_w_start = int(z_write_starts[z_idx]) - 1
            py_y_w_start = int(y_write_starts[y_idx]) - 1
            py_x_w_start = int(x_write_starts[x_idx]) - 1

            slice_z = slice(py_z_w_start, py_z_w_start + w_count_z)
            slice_y = slice(py_y_w_start, py_y_w_start + w_count_y)
            slice_x = slice(py_x_w_start, py_x_w_start + w_count_x)

            # One collection per chunk (not per scale) bounds cyclic garbage
            # without serializing every scale iteration under the GIL.
            gc.collect()
            return chunk_idx, (slice_z, slice_y, slice_x, chunk_energy_min, chunk_scale_min)

        if n_jobs == 1:
            for c_idx in range(number_of_chunks):
                _, (slice_z, slice_y, slice_x, chunk_energy, chunk_scale) = _process_chunk(c_idx)
                master_energy = energy_3d[slice_z, slice_y, slice_x]
                is_better = chunk_energy < master_energy
                energy_3d[slice_z, slice_y, slice_x] = np.where(
                    is_better, chunk_energy, master_energy
                )
                scale_indices[slice_z, slice_y, slice_x] = np.where(
                    is_better, chunk_scale, scale_indices[slice_z, slice_y, slice_x]
                )
                completed_chunk_units += 1
                if progress_callback is not None:
                    progress_callback(
                        completed_chunk_units, total_chunk_units, int(current_octave), c_idx
                    )
        else:
            chunk_results = Parallel(n_jobs=n_jobs, prefer="threads", verbose=10)(
                delayed(_process_chunk)(c_idx) for c_idx in range(number_of_chunks)
            )

            for _, (slice_z, slice_y, slice_x, chunk_energy, chunk_scale) in chunk_results:
                master_energy = energy_3d[slice_z, slice_y, slice_x]
                is_better = chunk_energy < master_energy
                energy_3d[slice_z, slice_y, slice_x] = np.where(
                    is_better, chunk_energy, master_energy
                )
                scale_indices[slice_z, slice_y, slice_x] = np.where(
                    is_better, chunk_scale, scale_indices[slice_z, slice_y, slice_x]
                )
                completed_chunk_units += 1
                if progress_callback is not None:
                    progress_callback(
                        completed_chunk_units, total_chunk_units, int(current_octave), -1
                    )

    energy_3d[energy_3d >= 0.0] = 0.0
    energy_3d[~np.isfinite(energy_3d)] = 0.0
    scale_indices[energy_3d >= 0.0] = -1
    return energy_3d.astype(np.float64, copy=False), scale_indices.astype(np.int16), None


__all__ = [
    "_interp3_matlab_linear_inf",
    "_matlab_coarse_local_slices",
    "_matlab_zero_based_linspace",
    "_matlab_zero_based_linspace_raw",
    "compute_exact_parity_energy_chunked",
    "get_chunking_lattice_v190",
    "get_starts_and_counts_v200",
]
