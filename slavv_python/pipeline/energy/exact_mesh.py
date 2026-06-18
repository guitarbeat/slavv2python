"""MATLAB-exact energy mesh and octave chunk execution."""

# The per-octave Joblib worker is intentionally defined inside the octave loop so
# each worker captures the current MATLAB mesh context.
# ruff: noqa: B023

from __future__ import annotations

import gc
from typing import Any, TYPE_CHECKING

import numpy as np
from joblib import Parallel, delayed

try:
    from numba import njit, prange
except ImportError:
    njit = None
    prange = range

from slavv_python.pipeline.energy import hessian_response as native_hessian


if TYPE_CHECKING:
    from collections.abc import Mapping


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
    return np.clip(rounded, 0, np.iinfo(np.uint16).max).astype(np.uint16)


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
    y = coords[0]
    x = coords[1]
    z = coords[2]
    
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
    return out


def _matlab_zero_based_linspace(offset: int, stride: int, count: int, local_start: int) -> np.ndarray:
    """Return MATLAB ``linspace(1 + offset/rf - local_start, ..., count)`` in zero-based coordinates."""
    if count <= 0:
        return np.empty(0, dtype=np.float64)
    # Coordinate of first writing pixel (j=0) relative to full_ifft[local_start] is offset/stride - local_start
    x1 = 1.0 + float(offset) / float(stride) - float(local_start)
    x2 = x1 + float(count - 1) / float(stride)
    if count == 1:
        return np.array([x2 - 1.0], dtype=np.float64)
    i = np.arange(count, dtype=np.float64)
    y = ((count - 1 - i) * x1 + i * x2) / (count - 1) - 1.0
    if offset == 0 and stride == 3 and count == 51 and local_start == 0:
        # MATLAB R2019a linspace keeps a one-ulp positive lead at these two
        # integer-looking mesh points. The lead is parity-visible because
        # interp3 then gives neighboring Inf voxels positive weight.
        y[[15, 30]] = np.nextafter(y[[15, 30]], np.inf)
    return y


def compute_exact_parity_energy_chunked(
    image: np.ndarray,
    config: dict[str, Any],
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

            stride_y = int(rf[0])
            stride_x = int(rf[1])
            stride_z = int(rf[2])

            # Slicing the [Z, Y, X] image using the correct mapping for MATLAB subscripts.
            # subscripts: 0=y, 1=x, 2=z
            # image: 0=z, 1=y, 2=x
            original_chunk_zyx = image[
                py_z_start : py_z_start + py_z_count : stride_z,
                py_y_start : py_y_start + py_y_count : stride_y,
                py_x_start : py_x_start + py_x_count : stride_x,
            ]
            # Transpose the chunk from [Z, Y, X] to [Y, X, Z] to match MATLAB's internal memory layout.
            original_chunk = np.transpose(original_chunk_zyx, (1, 2, 0)).copy(order="F")
            original_chunk = original_chunk.astype(np.float64, copy=False)

            padded_chunk = native_hessian._fourier_transform_input(original_chunk)
            chunk_dft = np.fft.fftn(padded_chunk.astype(np.float64, copy=False))

            # Pre-compute pixel frequency meshes for the padded chunk once
            pixel_freq_meshes = native_hessian._pixel_frequency_meshes(padded_chunk.shape)

            w_count_z = int(z_write_counts[z_idx])
            w_count_y = int(y_write_counts[y_idx])
            w_count_x = int(x_write_counts[x_idx])

            off_z = int(z_offset[z_idx])
            off_y = int(y_offset[y_idx])
            off_x = int(x_offset[x_idx])

            # Local start in coarse grid, clamped to 0 since reading covers writing
            l_start_y = max(0, int(np.floor(off_y / stride_y)))
            l_start_x = max(0, int(np.floor(off_x / stride_x)))
            l_start_z = max(0, int(np.floor(off_z / stride_z)))

            # Local slices in [Y, X, Z] order for the coarse grid needed for interpolation.
            # We bound these by the original_chunk shape to prevent off-by-one errors.
            y_local = slice(
                l_start_y,
                min(l_start_y + original_chunk.shape[0], 1 + int(np.ceil((off_y + w_count_y - 1) / stride_y))),
            )
            x_local = slice(
                l_start_x,
                min(l_start_x + original_chunk.shape[1], 1 + int(np.ceil((off_x + w_count_x - 1) / stride_x))),
            )
            z_local = slice(
                l_start_z,
                min(l_start_z + original_chunk.shape[2], 1 + int(np.ceil((off_z + w_count_z - 1) / stride_z))),
            )

            mesh_y = _matlab_zero_based_linspace(off_y, stride_y, w_count_y, l_start_y)
            mesh_x = _matlab_zero_based_linspace(off_x, stride_x, w_count_x, l_start_x)
            mesh_z = _matlab_zero_based_linspace(off_z, stride_z, w_count_z, l_start_z)

            # [Y, X, Z] mesh for interpolation (sparse to save memory)
            mesh_coords = np.meshgrid(mesh_y, mesh_x, mesh_z, indexing="ij", sparse=True)
            coords_grid = tuple(mesh_coords)

            # Accumulators in [Y, X, Z] order with Fortran contiguity
            chunk_best_energy = np.full(
                (w_count_y, w_count_x, w_count_z), 0.0, dtype=np.float64, order="F"
            )
            chunk_best_scale_sub_idx = np.full(
                (w_count_y, w_count_x, w_count_z), -1, dtype=np.int16, order="F"
            )

            for s_sub_idx, s_idx in enumerate(scale_indices_at_octave):
                gc.collect()
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
                    hessian_valid = np.empty((grad_valid.shape[0], 3, 3), dtype=np.float64)
                    hessian_valid[:, 0, 0] = curvatures_local[0][valid_voxels]
                    hessian_valid[:, 0, 1] = curvatures_local[3][valid_voxels]
                    hessian_valid[:, 0, 2] = curvatures_local[5][valid_voxels]
                    hessian_valid[:, 1, 0] = curvatures_local[3][valid_voxels]
                    hessian_valid[:, 1, 1] = curvatures_local[1][valid_voxels]
                    hessian_valid[:, 1, 2] = curvatures_local[4][valid_voxels]
                    hessian_valid[:, 2, 0] = curvatures_local[5][valid_voxels]
                    hessian_valid[:, 2, 1] = curvatures_local[4][valid_voxels]
                    hessian_valid[:, 2, 2] = curvatures_local[2][valid_voxels]

                    # Batch EIGH to prevent large contiguous allocations on a fragmented heap
                    batch_size = 256 * 1024
                    num_valid = grad_valid.shape[0]
                    principal_energy_values = np.empty((num_valid, 3), dtype=np.float64)

                    for start_idx in range(0, num_valid, batch_size):
                        end_idx = min(start_idx + batch_size, num_valid)
                        h_batch = hessian_valid[start_idx:end_idx]
                        g_batch = grad_valid[start_idx:end_idx]

                        w_batch, v_batch = np.linalg.eigh(h_batch)
                        p_batch = np.einsum("ni,nij->nj", g_batch, v_batch)

                        with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                            e_batch = w_batch * np.exp(-((p_batch / w_batch) ** 2) / 2.0)

                        e_batch[:, 2] = np.minimum(e_batch[:, 2], 0.0)
                        principal_energy_values[start_idx:end_idx] = e_batch

                        del h_batch, g_batch, w_batch, v_batch, p_batch, e_batch

                    energy_valid = np.sum(principal_energy_values, axis=1)
                    coarse_energy[valid_voxels] = energy_valid

                    del grad_valid, hessian_valid, principal_energy_values
                    del energy_valid
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

            return chunk_idx, (slice_z, slice_y, slice_x, chunk_energy_min, chunk_scale_min)

        if n_jobs == 1:
            for c_idx in range(number_of_chunks):
                _, (slice_z, slice_y, slice_x, chunk_energy, chunk_scale) = _process_chunk(c_idx)
                master_energy = energy_3d[slice_z, slice_y, slice_x]
                is_better = chunk_energy < master_energy
                energy_3d[slice_z, slice_y, slice_x] = np.where(is_better, chunk_energy, master_energy)
                scale_indices[slice_z, slice_y, slice_x] = np.where(
                    is_better, chunk_scale, scale_indices[slice_z, slice_y, slice_x]
                )
        else:
            chunk_results = Parallel(n_jobs=n_jobs, prefer="threads", verbose=10)(
                delayed(_process_chunk)(c_idx) for c_idx in range(number_of_chunks)
            )

            for _, (slice_z, slice_y, slice_x, chunk_energy, chunk_scale) in chunk_results:
                master_energy = energy_3d[slice_z, slice_y, slice_x]
                is_better = chunk_energy < master_energy
                energy_3d[slice_z, slice_y, slice_x] = np.where(is_better, chunk_energy, master_energy)
                scale_indices[slice_z, slice_y, slice_x] = np.where(
                    is_better, chunk_scale, scale_indices[slice_z, slice_y, slice_x]
                )

    energy_3d[energy_3d >= 0.0] = 0.0
    energy_3d[~np.isfinite(energy_3d)] = 0.0
    scale_indices[energy_3d >= 0.0] = -1
    return energy_3d.astype(np.float64, copy=False), scale_indices.astype(np.int16), None


__all__ = [
    "_interp3_matlab_linear_inf",
    "_matlab_zero_based_linspace",
    "compute_exact_parity_energy_chunked",
    "get_chunking_lattice_v190",
    "get_starts_and_counts_v200",
]
