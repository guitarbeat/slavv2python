"""One-voxel exact-route Energy probes for MATLAB parity isolation."""

from __future__ import annotations

from typing import Any

import numpy as np

from slavv_python.pipeline.energy import hessian_response as native_hessian
from slavv_python.pipeline.energy.exact_mesh import (
    _interp3_matlab_linear_inf,
    _matlab_coarse_local_slices,
    _matlab_zero_based_linspace,
    get_chunking_lattice_v190,
    get_starts_and_counts_v200,
)
from slavv_python.pipeline.energy.math import compute_principal_energy


def resolve_write_chunk_idx_for_voxel(
    config: dict[str, Any],
    *,
    voxel_zyx: tuple[int, int, int],
    target_rf_zyx: tuple[int, int, int],
) -> int:
    """Return the chunk index whose write window contains ``voxel_zyx`` at ``target_rf_zyx``."""
    octave_at_scales = np.asarray(config["octave_at_scales"])
    scale_resolution_factors = np.asarray(config["scale_resolution_factors"])
    current_octave = _resolve_octave_for_rf(
        octave_at_scales, scale_resolution_factors, target_rf_zyx
    )
    scale_indices_at_octave = np.where(octave_at_scales == current_octave)[0]

    image_shape = np.asarray(config.get("image_shape"), dtype=float)
    if image_shape.size != 3:
        raise ValueError("config must include image_shape=(Z, Y, X) for chunk resolution")
    matlab_image_shape = np.array([image_shape[1], image_shape[2], image_shape[0]], dtype=float)
    rf = np.asarray(scale_resolution_factors[scale_indices_at_octave[0]], dtype=float)
    rf_matlab = np.array([rf[1], rf[2], rf[0]], dtype=float)

    microns_per_voxel = np.asarray(config["microns_per_voxel"], dtype=float)
    pixels_per_sigma_PSF = np.asarray(config["pixels_per_sigma_PSF"], dtype=float)
    lumen_radius_microns = np.asarray(config["lumen_radius_microns"], dtype=float)
    largest_scale_idx = scale_indices_at_octave[-1]
    largest_pixels_per_radius = lumen_radius_microns[largest_scale_idx] / microns_per_voxel

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
    chunk_overlap_matlab = np.ceil(
        6.0 * np.sqrt(pixels_per_sigma_PSF**2 + largest_pixels_per_radius**2)
    ).astype(np.int32)[[1, 2, 0]]
    starts_counts = get_starts_and_counts_v200(
        chunk_lattice_dimensions,
        chunk_overlap_matlab,
        matlab_image_shape,
        rf_matlab,
    )
    y_write_starts = starts_counts[6]
    x_write_starts = starts_counts[7]
    z_write_starts = starts_counts[8]
    y_write_counts = starts_counts[9]
    x_write_counts = starts_counts[10]
    z_write_counts = starts_counts[11]

    py_z, py_y, py_x = voxel_zyx
    for chunk_idx in range(int(number_of_chunks)):
        y_idx, x_idx, z_idx = np.unravel_index(chunk_idx, chunk_lattice_dimensions, order="F")
        py_z_w_start = int(z_write_starts[z_idx]) - 1
        py_y_w_start = int(y_write_starts[y_idx]) - 1
        py_x_w_start = int(x_write_starts[x_idx]) - 1
        w_count_z = int(z_write_counts[z_idx])
        w_count_y = int(y_write_counts[y_idx])
        w_count_x = int(x_write_counts[x_idx])
        if (
            py_z_w_start <= py_z < py_z_w_start + w_count_z
            and py_y_w_start <= py_y < py_y_w_start + w_count_y
            and py_x_w_start <= py_x < py_x_w_start + w_count_x
        ):
            return int(chunk_idx)

    raise ValueError(
        f"voxel {voxel_zyx} is not covered by any write window at rf_zyx={target_rf_zyx}"
    )


def _resolve_octave_for_rf(
    octave_at_scales: np.ndarray,
    scale_resolution_factors: np.ndarray,
    target_rf_zyx: tuple[int, int, int],
) -> int:
    target = np.asarray(target_rf_zyx, dtype=np.int16)
    for current_octave in np.unique(octave_at_scales):
        scale_indices = np.where(octave_at_scales == current_octave)[0]
        if len(scale_indices) == 0:
            continue
        rf = np.asarray(scale_resolution_factors[scale_indices[0]], dtype=np.int16)
        if np.array_equal(rf, target):
            return int(current_octave)
    raise ValueError(f"no consolidated octave matches rf_zyx={target_rf_zyx}")


def _coarse_interp_corners(
    coarse_energy: np.ndarray,
    mesh_y: float,
    mesh_x: float,
    mesh_z: float,
) -> dict[str, Any]:
    """Record mesh coordinates and eight corner coarse samples for interp3 diagnosis."""
    y0 = int(np.floor(mesh_y))
    x0 = int(np.floor(mesh_x))
    z0 = int(np.floor(mesh_z))
    shape_y, shape_x, shape_z = coarse_energy.shape
    corners: dict[str, float | None] = {}
    for name, (dy, dx, dz) in {
        "c000": (0, 0, 0),
        "c100": (1, 0, 0),
        "c010": (0, 1, 0),
        "c110": (1, 1, 0),
        "c001": (0, 0, 1),
        "c101": (1, 0, 1),
        "c011": (0, 1, 1),
        "c111": (1, 1, 1),
    }.items():
        iy, ix, iz = y0 + dy, x0 + dx, z0 + dz
        if 0 <= iy < shape_y and 0 <= ix < shape_x and 0 <= iz < shape_z:
            value = float(coarse_energy[iy, ix, iz])
            corners[name] = None if not np.isfinite(value) else value
        else:
            corners[name] = None
    return {
        "mesh_yxz_1based": [mesh_y + 1.0, mesh_x + 1.0, mesh_z + 1.0],
        "mesh_yxz_0based": [mesh_y, mesh_x, mesh_z],
        "corner_base_0based": [y0, x0, z0],
        "corner_weights": [
            float((mesh_y - y0) * (mesh_x - x0) * (mesh_z - z0)),
            float((1.0 - (mesh_y - y0)) * (mesh_x - x0) * (mesh_z - z0)),
        ],
        "corners": corners,
    }


def probe_exact_energy_voxel_at_octave(
    image: np.ndarray,
    config: dict[str, Any],
    *,
    voxel_zyx: tuple[int, int, int],
    target_rf_zyx: tuple[int, int, int],
    chunk_idx: int = 0,
) -> dict[str, Any]:
    """Compute per-scale upsampled energy at one voxel for a single consolidated octave."""
    octave_at_scales = np.asarray(config["octave_at_scales"])
    scale_resolution_factors = np.asarray(config["scale_resolution_factors"])
    current_octave = _resolve_octave_for_rf(
        octave_at_scales, scale_resolution_factors, target_rf_zyx
    )
    scale_indices_at_octave = np.where(octave_at_scales == current_octave)[0]

    image_shape = np.asarray(image.shape, dtype=float)
    matlab_image_shape = np.array([image_shape[1], image_shape[2], image_shape[0]], dtype=float)
    rf = np.asarray(scale_resolution_factors[scale_indices_at_octave[0]], dtype=float)
    rf_matlab = np.array([rf[1], rf[2], rf[0]], dtype=float)

    microns_per_voxel = np.asarray(config["microns_per_voxel"], dtype=float)
    pixels_per_sigma_PSF = np.asarray(config["pixels_per_sigma_PSF"], dtype=float)
    lumen_radius_microns = np.asarray(config["lumen_radius_microns"], dtype=float)

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

    chunk_lattice_dimensions, _ = get_chunking_lattice_v190(
        voxel_aspect_ratio,
        float(config["max_voxels"]),
        approx_size,
    )
    chunk_overlap_matlab = np.ceil(
        6.0 * np.sqrt(pixels_per_sigma_PSF**2 + largest_pixels_per_radius**2)
    ).astype(np.int32)[[1, 2, 0]]
    starts_counts = get_starts_and_counts_v200(
        chunk_lattice_dimensions,
        chunk_overlap_matlab,
        matlab_image_shape,
        rf_matlab,
    )

    y_idx, x_idx, z_idx = np.unravel_index(chunk_idx, chunk_lattice_dimensions, order="F")
    py_z, py_y, py_x = voxel_zyx
    y_write_starts = starts_counts[6]
    x_write_starts = starts_counts[7]
    z_write_starts = starts_counts[8]
    y_write_counts = starts_counts[9]
    x_write_counts = starts_counts[10]
    z_write_counts = starts_counts[11]

    py_z_w_start = int(z_write_starts[z_idx]) - 1
    py_y_w_start = int(y_write_starts[y_idx]) - 1
    py_x_w_start = int(x_write_starts[x_idx]) - 1

    wy = py_y - py_y_w_start
    wx = py_x - py_x_w_start
    wz = py_z - py_z_w_start

    if not (
        0 <= wy < int(y_write_counts[y_idx])
        and 0 <= wx < int(x_write_counts[x_idx])
        and 0 <= wz < int(z_write_counts[z_idx])
    ):
        raise ValueError(
            f"voxel {voxel_zyx} outside chunk {chunk_idx} write window "
            f"y={slice(py_y_w_start, py_y_w_start + int(y_write_counts[y_idx]))} "
            f"x={slice(py_x_w_start, py_x_w_start + int(x_write_counts[x_idx]))} "
            f"z={slice(py_z_w_start, py_z_w_start + int(z_write_counts[z_idx]))}"
        )

    stride_z, stride_y, stride_x = int(rf[0]), int(rf[1]), int(rf[2])
    py_z_read = int(starts_counts[2][z_idx]) - 1
    py_y_read = int(starts_counts[0][y_idx]) - 1
    py_x_read = int(starts_counts[1][x_idx]) - 1
    py_z_count = int(starts_counts[5][z_idx])
    py_y_count = int(starts_counts[3][y_idx])
    py_x_count = int(starts_counts[4][x_idx])

    original_chunk_zyx = image[
        py_z_read : py_z_read + py_z_count : stride_z,
        py_y_read : py_y_read + py_y_count : stride_y,
        py_x_read : py_x_read + py_x_count : stride_x,
    ]
    original_chunk = (
        np.transpose(original_chunk_zyx, (1, 2, 0)).copy(order="F").astype(np.float64, copy=False)
    )
    padded_chunk = native_hessian._fourier_transform_input(original_chunk)
    padded_shape = padded_chunk.shape
    chunk_dft = np.fft.fftn(padded_chunk.astype(np.float64, copy=False))
    pixel_freq_meshes = native_hessian._pixel_frequency_meshes(padded_chunk.shape)

    off_y = int(starts_counts[12][y_idx])
    off_x = int(starts_counts[13][x_idx])
    off_z = int(starts_counts[14][z_idx])
    w_count_y = int(y_write_counts[y_idx])
    w_count_x = int(x_write_counts[x_idx])
    w_count_z = int(z_write_counts[z_idx])

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

    mesh_at_voxel = {
        "mesh_y": float(mesh_y[wy]),
        "mesh_x": float(mesh_x[wx]),
        "mesh_z": float(mesh_z[wz]),
    }

    prev_scales_count = int(np.sum(octave_at_scales < current_octave))
    per_scale: list[dict[str, Any]] = []
    best_energy = np.inf
    best_global_scale = -1

    for s_sub_idx, s_idx in enumerate(scale_indices_at_octave):
        s_idx = int(s_idx)
        matching_kernel_dft, derivative_weights = native_hessian._matching_kernel_dft(
            pixel_freq_meshes,
            radius_of_lumen_in_microns=float(lumen_radius_microns[s_idx]),
            microns_per_pixel=microns_per_pixel_matlab,
            pixels_per_sigma_psf=(pixels_per_sigma_PSF / rf)[[1, 2, 0]],
            gaussian_to_ideal_ratio=float(config["gaussian_to_ideal_ratio"]),
            spherical_to_annular_ratio=float(config["spherical_to_annular_ratio"]),
        )
        filtered_chunk_dft = matching_kernel_dft * chunk_dft

        coarse_shape = (
            y_local.stop - y_local.start,
            x_local.stop - x_local.start,
            z_local.stop - z_local.start,
        )
        curvatures_local = np.empty((6, *coarse_shape), dtype=np.float64, order="F")
        for k_idx in range(6):
            k_dft = native_hessian._derivative_kernel_dft_single(
                pixel_freq_meshes, derivative_weights, k_idx, is_curvature=True
            )
            full_ifft = native_hessian._ifftn_matlab_symmetric(k_dft * filtered_chunk_dft)
            curvatures_local[k_idx] = full_ifft[y_local, x_local, z_local]

        gradient_local = np.empty((3, *coarse_shape), dtype=np.float64, order="F")
        for k_idx in range(3):
            k_dft = native_hessian._derivative_kernel_dft_single(
                pixel_freq_meshes, derivative_weights, k_idx, is_curvature=False
            )
            full_ifft = native_hessian._ifftn_matlab_symmetric(k_dft * filtered_chunk_dft)
            gradient_local[k_idx] = full_ifft[y_local, x_local, z_local]

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
            coarse_energy[valid_voxels] = compute_principal_energy(
                grad_valid,
                curvatures_valid,
                energy_sign=float(config.get("energy_sign", -1.0)),
            )
        coarse_energy[~np.isfinite(coarse_energy)] = np.inf
        coarse_energy[coarse_energy >= 0] = np.inf

        upsampled = _interp3_matlab_linear_inf(
            coarse_energy,
            (
                np.array([[mesh_at_voxel["mesh_y"]]], dtype=np.float64),
                np.array([[mesh_at_voxel["mesh_x"]]], dtype=np.float64),
                np.array([[mesh_at_voxel["mesh_z"]]], dtype=np.float64),
            ),
        )
        upsampled_val = float(np.asarray(upsampled, dtype=np.float64).ravel()[0])
        if (not np.isfinite(upsampled_val)) or upsampled_val >= 0.0:
            upsampled_val = 0.0

        global_scale = prev_scales_count + s_sub_idx
        record = {
            "global_scale": global_scale,
            "s_sub_idx": int(s_sub_idx),
            "lumen_radius_microns": float(lumen_radius_microns[s_idx]),
            "upsampled_energy": upsampled_val,
            "interp3": _coarse_interp_corners(
                coarse_energy,
                mesh_at_voxel["mesh_y"],
                mesh_at_voxel["mesh_x"],
                mesh_at_voxel["mesh_z"],
            ),
        }
        per_scale.append(record)
        if upsampled_val < best_energy and upsampled_val < 0.0:
            best_energy = upsampled_val
            best_global_scale = global_scale

    return {
        "voxel_zyx": list(voxel_zyx),
        "consolidated_octave": current_octave,
        "rf_zyx": [int(v) for v in rf],
        "rf_matlab_yxz": [int(v) for v in rf_matlab],
        "chunk_lattice_dimensions_yxz": [int(v) for v in chunk_lattice_dimensions],
        "chunk_idx": int(chunk_idx),
        "write_index_yxz": [int(y_idx), int(x_idx), int(z_idx)],
        "target_write_offset_yxz": [wy, wx, wz],
        "write_window_zyx": {
            "starts": [py_z_w_start, py_y_w_start, py_x_w_start],
            "counts": [w_count_z, w_count_y, w_count_x],
        },
        "offsets_yxz": [off_y, off_x, off_z],
        "strided_read_shape_yxz": list(original_chunk.shape),
        "padded_shape_yxz": list(padded_shape),
        "coarse_local_slices_yxz": [
            [y_local.start, y_local.stop],
            [x_local.start, x_local.stop],
            [z_local.start, z_local.stop],
        ],
        "coarse_shape_yxz": [
            y_local.stop - y_local.start,
            x_local.stop - x_local.start,
            z_local.stop - z_local.start,
        ],
        "mesh_at_voxel": mesh_at_voxel,
        "prev_scales_count": prev_scales_count,
        "per_scale": per_scale,
        "octave_winner": {
            "global_scale": int(best_global_scale),
            "upsampled_energy": float(best_energy if np.isfinite(best_energy) else 0.0),
        },
    }


__all__ = [
    "probe_exact_energy_voxel_at_octave",
    "resolve_write_chunk_idx_for_voxel",
]
