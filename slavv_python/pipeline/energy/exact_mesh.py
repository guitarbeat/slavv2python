"""MATLAB-exact energy mesh and octave chunk execution."""

# The per-octave Joblib worker is intentionally defined inside the octave loop so
# each worker captures the current MATLAB mesh context.
# ruff: noqa: B023

from __future__ import annotations

from typing import Any

import numpy as np
from joblib import Parallel, delayed

from slavv_python.pipeline.energy import hessian_response as native_hessian


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
    coords: np.ndarray,
    *,
    cval: float = 0.0,
) -> np.ndarray:
    """Linear interp3-compatible interpolation that propagates positive-weight Inf."""
    y = coords[0]
    x = coords[1]
    z = coords[2]
    y0 = np.floor(y).astype(np.int64)
    x0 = np.floor(x).astype(np.int64)
    z0 = np.floor(z).astype(np.int64)
    fy = y - y0
    fx = x - x0
    fz = z - z0

    out = np.zeros_like(y, dtype=np.float64)
    has_inf = np.zeros_like(y, dtype=bool)
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

                values = np.full(out.shape, cval, dtype=np.float64)
                valid_positive = valid & positive_weight
                if np.any(valid_positive):
                    values[valid_positive] = volume[
                        iy[valid_positive],
                        ix[valid_positive],
                        iz[valid_positive],
                    ]
                    has_inf |= valid_positive & np.isposinf(values)
                    finite = valid_positive & np.isfinite(values)
                    out[finite] += weight[finite] * values[finite]

                invalid_positive = (~valid) & positive_weight
                if cval and np.any(invalid_positive):
                    out[invalid_positive] += weight[invalid_positive] * cval

    out[has_inf] = np.inf
    return out


def _matlab_zero_based_linspace(offset: int, stride: int, count: int) -> np.ndarray:
    """Return MATLAB ``linspace(1+offset/rf, ..., count)`` in zero-based coordinates."""
    if count <= 0:
        return np.empty(0, dtype=np.float64)
    x1 = 1.0 + float(offset % stride) / float(stride)
    x2 = x1 + float(count - 1) / float(stride)
    if count == 1:
        return np.array([x2 - 1.0], dtype=np.float64)
    i = np.arange(count, dtype=np.float64)
    y = ((count - 1 - i) * x1 + i * x2) / (count - 1) - 1.0
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

            stride_z = int(rf[0])
            stride_y = int(rf[1])
            stride_x = int(rf[2])

            original_chunk_zyx = image[
                py_z_start : py_z_start + py_z_count : stride_z,
                py_y_start : py_y_start + py_y_count : stride_y,
                py_x_start : py_x_start + py_x_count : stride_x,
            ]
            original_chunk = np.ascontiguousarray(original_chunk_zyx.transpose(1, 2, 0))

            padded_chunk = native_hessian._fourier_transform_input(original_chunk)
            chunk_dft = np.fft.fftn(padded_chunk.astype(np.float64, copy=False))
            pixel_freq_meshes = native_hessian._pixel_frequency_meshes(padded_chunk.shape)

            w_count_z = int(z_write_counts[z_idx])
            w_count_y = int(y_write_counts[y_idx])
            w_count_x = int(x_write_counts[x_idx])

            off_z = int(z_offset[z_idx])
            off_y = int(y_offset[y_idx])
            off_x = int(x_offset[x_idx])

            z_local = slice(
                int(np.floor(off_z / stride_z)),
                1 + int(np.ceil((off_z + w_count_z - 1) / stride_z)),
            )
            y_local = slice(
                int(np.floor(off_y / stride_y)),
                1 + int(np.ceil((off_y + w_count_y - 1) / stride_y)),
            )
            x_local = slice(
                int(np.floor(off_x / stride_x)),
                1 + int(np.ceil((off_x + w_count_x - 1) / stride_x)),
            )

            mesh_z = _matlab_zero_based_linspace(off_z, stride_z, w_count_z)
            mesh_y = _matlab_zero_based_linspace(off_y, stride_y, w_count_y)
            mesh_x = _matlab_zero_based_linspace(off_x, stride_x, w_count_x)

            mesh_coords = np.meshgrid(mesh_y, mesh_x, mesh_z, indexing="ij")
            coords_grid = np.stack(mesh_coords, axis=0)

            pixel_freq_meshes = native_hessian._pixel_frequency_meshes(chunk_dft.shape)
            base_kernels = native_hessian._precompute_base_derivative_kernels_dft(pixel_freq_meshes)

            chunk_best_energy = np.full((w_count_y, w_count_x, w_count_z), 0.0, dtype=np.float64)
            chunk_best_scale_sub_idx = np.full((w_count_y, w_count_x, w_count_z), -1, dtype=np.int16)

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

                curvatures_kernels_dft, gradient_kernels_dft = (
                    native_hessian._derivative_kernels_dft(
                        pixel_freq_meshes,
                        derivative_weights,
                        base_kernels=base_kernels,
                    )
                )

                filtered_chunk_dft = matching_kernel_dft * chunk_dft
                curvatures_chunk = np.empty(
                    (curvatures_kernels_dft.shape[0], *chunk_dft.shape),
                    dtype=np.float64,
                )
                for kernel_index, curvature_kernel_dft in enumerate(curvatures_kernels_dft):
                    curvatures_chunk[kernel_index] = native_hessian._ifftn_matlab_symmetric(
                        curvature_kernel_dft * filtered_chunk_dft,
                    )

                gradient_chunk = np.empty(
                    (gradient_kernels_dft.shape[0], *chunk_dft.shape),
                    dtype=np.float64,
                )
                for kernel_index, gradient_kernel_dft in enumerate(gradient_kernels_dft):
                    gradient_chunk[kernel_index] = native_hessian._ifftn_matlab_symmetric(
                        gradient_kernel_dft * filtered_chunk_dft,
                    )

                # Explicitly delete DFT products
                del filtered_chunk_dft

                # Crop to local mesh bounds (these are the smaller working arrays)
                curvatures_local = curvatures_chunk[:, y_local, x_local, z_local].copy()
                gradient_local = gradient_chunk[:, y_local, x_local, z_local].copy()

                # Free large FFT results
                del curvatures_chunk
                del gradient_chunk

                laplacian_chunk = curvatures_local[0] + curvatures_local[1] + curvatures_local[2]
                valid_voxels = laplacian_chunk < 0
                coarse_shape = curvatures_local.shape[1:4]
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

                    principal_curvature_values, principal_curvature_vectors = np.linalg.eigh(
                        hessian_valid
                    )
                    principal_projections = np.einsum(
                        "ni,nij->nj",
                        grad_valid,
                        principal_curvature_vectors,
                    )
                    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
                        principal_energy_values = principal_curvature_values * np.exp(
                            -((principal_projections / principal_curvature_values) ** 2) / 2.0
                        )
                    principal_energy_values[:, 2] = np.minimum(principal_energy_values[:, 2], 0.0)
                    energy_valid = np.sum(principal_energy_values, axis=1)
                    coarse_energy[valid_voxels] = energy_valid

                # Free local cropped arrays
                del curvatures_local
                del gradient_local

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

            chunk_energy_min = chunk_best_energy.transpose(2, 0, 1)
            chunk_scale_min = chunk_best_scale_sub_idx.transpose(2, 0, 1)
            chunk_scale_min[chunk_energy_min >= 0.0] = -1

            import gc

            gc.collect()

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
