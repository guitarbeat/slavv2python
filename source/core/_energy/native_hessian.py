from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.special import jv

_WORST_RESOLUTION_TO_DOWNSAMPLE = 1.0 / 2.5


def matlab_octave_resolution_factors(
    lumen_radius_microns: np.ndarray,
    microns_per_voxel: np.ndarray,
    scales_per_octave: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return MATLAB-style octave ids and per-scale integer downsampling factors."""
    number_of_scales = len(lumen_radius_microns)
    scale_subscripts = np.arange(1, number_of_scales + 1, dtype=float)
    octave_at_scales = np.ceil(scale_subscripts / scales_per_octave / 3.0).astype(np.int16)

    resolution_factors_by_octave: dict[int, np.ndarray] = {}
    for current_octave in np.unique(octave_at_scales):
        smallest_scale_at_octave = min(
            number_of_scales,
            int(np.floor((int(current_octave) - 1) * scales_per_octave * 3.0)) + 1,
        )
        resolutions_at_octave = np.minimum(
            microns_per_voxel / float(lumen_radius_microns[smallest_scale_at_octave - 1]),
            np.full(3, _WORST_RESOLUTION_TO_DOWNSAMPLE, dtype=float),
        )
        resolution_factors = np.maximum(
            np.rint(_WORST_RESOLUTION_TO_DOWNSAMPLE / resolutions_at_octave).astype(np.int16),
            1,
        )
        resolution_factors_by_octave[int(current_octave)] = resolution_factors

    scale_resolution_factors = np.stack(
        [resolution_factors_by_octave[int(octave)] for octave in octave_at_scales],
        axis=0,
    )
    return octave_at_scales, scale_resolution_factors


def required_scale_stack(config: dict[str, Any]) -> bool:
    """Return whether the full 4D scale stack is needed for the configured projection."""
    return bool(config["return_all_scales"]) or str(config["energy_projection_mode"]) == "paper"


def project_energy_stack(
    energy_4d: np.ndarray,
    *,
    energy_sign: float,
    projection_mode: str,
    spherical_to_annular_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Project a per-scale energy stack into final energy and scale-index volumes."""
    if energy_sign >= 0:
        energy_3d = np.max(energy_4d, axis=3)
        scale_indices = np.argmax(energy_4d, axis=3).astype(np.int16)
        return energy_3d.astype(np.float32, copy=False), scale_indices

    if projection_mode == "matlab":
        energy_3d = np.min(energy_4d, axis=3)
        scale_indices = np.argmin(energy_4d, axis=3).astype(np.int16)
        return energy_3d.astype(np.float32, copy=False), scale_indices

    annular_indices = np.argmin(energy_4d, axis=3).astype(np.int16)
    annular_energy = np.take_along_axis(
        energy_4d,
        annular_indices[..., None],
        axis=3,
    )[..., 0]

    negative_weights = np.where(np.isfinite(energy_4d) & (energy_4d < 0), -energy_4d, 0.0)
    scale_axis = np.arange(energy_4d.shape[3], dtype=np.float32)
    weighted_sum = np.sum(negative_weights * scale_axis.reshape((1, 1, 1, -1)), axis=3)
    total_weight = np.sum(negative_weights, axis=3)

    spherical_indices = np.divide(
        weighted_sum,
        total_weight,
        out=annular_indices.astype(np.float32),
        where=total_weight > 0,
    )
    blended_indices = spherical_to_annular_ratio * spherical_indices + (
        1.0 - spherical_to_annular_ratio
    ) * annular_indices.astype(np.float32)
    sampled_indices = np.clip(np.rint(blended_indices), 0, energy_4d.shape[3] - 1).astype(np.int16)
    sampled_energy = np.take_along_axis(energy_4d, sampled_indices[..., None], axis=3)[..., 0]

    fallback_mask = ~np.isfinite(sampled_energy)
    if np.any(fallback_mask):
        sampled_energy = sampled_energy.copy()
        sampled_indices = sampled_indices.copy()
        sampled_energy[fallback_mask] = annular_energy[fallback_mask]
        sampled_indices[fallback_mask] = annular_indices[fallback_mask]

    return sampled_energy.astype(np.float32, copy=False), sampled_indices


def compute_native_hessian_energy(
    image: np.ndarray,
    config: dict[str, Any],
    scale_idx: int,
) -> np.ndarray:
    """Compute one scale of the MATLAB-style matched-filter Hessian energy."""
    debug_outputs = _compute_native_hessian_scale_debug(image, config, scale_idx)
    return debug_outputs["energy"]


def _compute_native_hessian_scale_debug(
    image: np.ndarray,
    config: dict[str, Any],
    scale_idx: int,
) -> dict[str, np.ndarray]:
    """Return one scale of native Hessian intermediates on the working grid."""
    resolution_factor = np.asarray(config["scale_resolution_factors"][scale_idx], dtype=np.int16)
    radius_microns = float(config["lumen_radius_microns"][scale_idx])
    microns_per_pixel = np.asarray(config["microns_per_voxel"], dtype=float) * resolution_factor
    pixels_per_sigma_psf = (
        np.asarray(config["pixels_per_sigma_PSF"], dtype=float) / resolution_factor
    )

    working_image = _downsample_volume(image, resolution_factor)
    debug_outputs = _matched_hessian_intermediates(
        working_image.astype(np.float32, copy=False),
        radius_of_lumen_in_microns=radius_microns,
        microns_per_pixel=microns_per_pixel,
        pixels_per_sigma_psf=pixels_per_sigma_psf,
        gaussian_to_ideal_ratio=float(config["gaussian_to_ideal_ratio"]),
        spherical_to_annular_ratio=float(config["spherical_to_annular_ratio"]),
    )
    return {
        "resolution_factor": resolution_factor,
        "laplacian": debug_outputs["laplacian"],
        "valid_voxels": debug_outputs["valid_voxels"],
        "energy": _upsample_volume(debug_outputs["energy"], image.shape, resolution_factor),
    }


def _downsample_volume(image: np.ndarray, resolution_factor: np.ndarray) -> np.ndarray:
    factor_y, factor_x, factor_z = (int(value) for value in resolution_factor)
    if factor_y == factor_x == factor_z == 1:
        return image
    return image[::factor_y, ::factor_x, ::factor_z]


def _upsample_volume(
    volume: np.ndarray,
    output_shape: tuple[int, int, int],
    resolution_factor: np.ndarray,
) -> np.ndarray:
    factor_y, factor_x, factor_z = (float(value) for value in resolution_factor)
    if factor_y == factor_x == factor_z == 1.0 and volume.shape == output_shape:
        return volume.astype(np.float32, copy=False)

    coord_y = np.arange(output_shape[0], dtype=np.float32) / factor_y
    coord_x = np.arange(output_shape[1], dtype=np.float32) / factor_x
    coord_z = np.arange(output_shape[2], dtype=np.float32) / factor_z
    mesh = np.meshgrid(coord_y, coord_x, coord_z, indexing="ij")
    coordinates = np.asarray(mesh, dtype=np.float32)
    upsampled = map_coordinates(
        volume.astype(np.float32, copy=False),
        coordinates,
        order=1,
        mode="nearest",
        prefilter=False,
    )
    return upsampled.astype(np.float32, copy=False)


def _matched_hessian_energy(
    image: np.ndarray,
    *,
    radius_of_lumen_in_microns: float,
    microns_per_pixel: np.ndarray,
    pixels_per_sigma_psf: np.ndarray,
    gaussian_to_ideal_ratio: float,
    spherical_to_annular_ratio: float,
) -> np.ndarray:
    return _matched_hessian_intermediates(
        image,
        radius_of_lumen_in_microns=radius_of_lumen_in_microns,
        microns_per_pixel=microns_per_pixel,
        pixels_per_sigma_psf=pixels_per_sigma_psf,
        gaussian_to_ideal_ratio=gaussian_to_ideal_ratio,
        spherical_to_annular_ratio=spherical_to_annular_ratio,
    )["energy"]


def _matched_hessian_intermediates(
    image: np.ndarray,
    *,
    radius_of_lumen_in_microns: float,
    microns_per_pixel: np.ndarray,
    pixels_per_sigma_psf: np.ndarray,
    gaussian_to_ideal_ratio: float,
    spherical_to_annular_ratio: float,
) -> dict[str, np.ndarray]:
    image = image.astype(np.float32, copy=False)
    original_shape = image.shape
    padded_image = _fourier_transform_input(image)
    chunk_dft = np.fft.fftn(padded_image.astype(np.float64, copy=False))
    pixel_freq_meshes = _pixel_frequency_meshes(padded_image.shape)
    matching_kernel_dft, derivative_weights = _matching_kernel_dft(
        pixel_freq_meshes,
        radius_of_lumen_in_microns=radius_of_lumen_in_microns,
        microns_per_pixel=microns_per_pixel,
        pixels_per_sigma_psf=pixels_per_sigma_psf,
        gaussian_to_ideal_ratio=gaussian_to_ideal_ratio,
        spherical_to_annular_ratio=spherical_to_annular_ratio,
    )
    curvatures_kernels_dft, gradient_kernels_dft = _derivative_kernels_dft(
        pixel_freq_meshes,
        derivative_weights,
    )

    curvatures_chunk = np.fft.ifftn(
        curvatures_kernels_dft * matching_kernel_dft[None, ...] * chunk_dft[None, ...],
        axes=(-3, -2, -1),
    ).real
    gradient_chunk = np.fft.ifftn(
        gradient_kernels_dft * matching_kernel_dft[None, ...] * chunk_dft[None, ...],
        axes=(-3, -2, -1),
    ).real

    curvatures_chunk = curvatures_chunk[
        :, : original_shape[0], : original_shape[1], : original_shape[2]
    ]
    gradient_chunk = gradient_chunk[
        :, : original_shape[0], : original_shape[1], : original_shape[2]
    ]

    laplacian_chunk = curvatures_chunk[0] + curvatures_chunk[1] + curvatures_chunk[2]
    valid_voxels = laplacian_chunk < 0
    energy_chunk = np.full(image.shape, np.inf, dtype=np.float32)
    if not np.any(valid_voxels):
        return {
            "laplacian": laplacian_chunk.astype(np.float32, copy=False),
            "valid_voxels": valid_voxels,
            "energy": energy_chunk,
        }

    grad_valid = np.stack(
        [
            gradient_chunk[0][valid_voxels],
            gradient_chunk[1][valid_voxels],
            gradient_chunk[2][valid_voxels],
        ],
        axis=1,
    )
    hessian_valid = np.empty((grad_valid.shape[0], 3, 3), dtype=np.float64)
    hessian_valid[:, 0, 0] = curvatures_chunk[0][valid_voxels]
    hessian_valid[:, 0, 1] = curvatures_chunk[3][valid_voxels]
    hessian_valid[:, 0, 2] = curvatures_chunk[5][valid_voxels]
    hessian_valid[:, 1, 0] = curvatures_chunk[3][valid_voxels]
    hessian_valid[:, 1, 1] = curvatures_chunk[1][valid_voxels]
    hessian_valid[:, 1, 2] = curvatures_chunk[4][valid_voxels]
    hessian_valid[:, 2, 0] = curvatures_chunk[5][valid_voxels]
    hessian_valid[:, 2, 1] = curvatures_chunk[4][valid_voxels]
    hessian_valid[:, 2, 2] = curvatures_chunk[2][valid_voxels]

    principal_curvature_values, principal_curvature_vectors = np.linalg.eigh(hessian_valid)
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

    energy_chunk[valid_voxels] = energy_valid.astype(np.float32, copy=False)
    energy_chunk[~np.isfinite(energy_chunk)] = np.inf
    energy_chunk[energy_chunk >= 0] = np.inf
    return {
        "laplacian": laplacian_chunk.astype(np.float32, copy=False),
        "valid_voxels": valid_voxels,
        "energy": energy_chunk,
    }


def _fourier_transform_input(image: np.ndarray) -> np.ndarray:
    size_of_image = np.asarray(image.shape, dtype=np.int64)
    next_even_image_size = 2 * np.ceil((size_of_image + 1) / 2.0).astype(np.int64)
    pad_width = [
        (0, int(padded - current)) for current, padded in zip(size_of_image, next_even_image_size)
    ]
    if all(after == 0 for _, after in pad_width):
        return image
    return np.pad(image, pad_width, mode="symmetric")


def _pixel_frequency_meshes(
    shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pixel_frequencies = [np.fft.fftfreq(length) for length in shape]
    return tuple(
        np.meshgrid(
            pixel_frequencies[0],
            pixel_frequencies[1],
            pixel_frequencies[2],
            indexing="ij",
        )
    )


def _matching_kernel_dft(
    pixel_freq_meshes: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    radius_of_lumen_in_microns: float,
    microns_per_pixel: np.ndarray,
    pixels_per_sigma_psf: np.ndarray,
    gaussian_to_ideal_ratio: float,
    spherical_to_annular_ratio: float,
) -> tuple[np.ndarray, np.ndarray]:
    y_pixel_freq_mesh, x_pixel_freq_mesh, z_pixel_freq_mesh = pixel_freq_meshes
    y_micron_freq_mesh = y_pixel_freq_mesh / microns_per_pixel[0]
    x_micron_freq_mesh = x_pixel_freq_mesh / microns_per_pixel[1]
    z_micron_freq_mesh = z_pixel_freq_mesh / microns_per_pixel[2]

    microns_per_sigma_psf = pixels_per_sigma_psf * microns_per_pixel
    gaussian_lengths = gaussian_to_ideal_ratio * radius_of_lumen_in_microns + np.zeros(3)
    annular_pulse_lengths_squared = (
        1.0 - gaussian_to_ideal_ratio**2
    ) * radius_of_lumen_in_microns**2 + microns_per_sigma_psf**2
    sphere_pulse_lengths_squared = annular_pulse_lengths_squared.copy()

    radial_freq_mesh_gaussian = np.sqrt(
        (y_micron_freq_mesh * gaussian_lengths[0]) ** 2
        + (x_micron_freq_mesh * gaussian_lengths[1]) ** 2
        + (z_micron_freq_mesh * gaussian_lengths[2]) ** 2
    )
    gaussian_kernel_dft = np.exp(-2.0 * np.pi**2 * radial_freq_mesh_gaussian**2)

    radial_angular_freq_mesh_sphere = (
        2.0
        * np.pi
        * np.sqrt(
            y_micron_freq_mesh**2 * sphere_pulse_lengths_squared[0]
            + x_micron_freq_mesh**2 * sphere_pulse_lengths_squared[1]
            + z_micron_freq_mesh**2 * sphere_pulse_lengths_squared[2]
        )
    )
    spherical_pulse_kernel_dft = np.ones_like(radial_angular_freq_mesh_sphere, dtype=np.float64)
    nonzero_sphere = radial_angular_freq_mesh_sphere != 0
    sphere_argument = radial_angular_freq_mesh_sphere[nonzero_sphere]
    spherical_pulse_kernel_dft[nonzero_sphere] = np.sqrt(np.pi / 2.0 / sphere_argument) * (
        jv(2.5, sphere_argument) + jv(0.5, sphere_argument)
    )

    radial_angular_freq_mesh_annular = (
        2.0
        * np.pi
        * np.sqrt(
            y_micron_freq_mesh**2 * annular_pulse_lengths_squared[0]
            + x_micron_freq_mesh**2 * annular_pulse_lengths_squared[1]
            + z_micron_freq_mesh**2 * annular_pulse_lengths_squared[2]
        )
    )
    annular_pulse_kernel_dft = np.cos(radial_angular_freq_mesh_annular)
    matching_kernel_dft = gaussian_kernel_dft * (
        (1.0 - spherical_to_annular_ratio) * annular_pulse_kernel_dft
        + spherical_to_annular_ratio * spherical_pulse_kernel_dft
    )
    derivative_weights_from_blurring = gaussian_lengths / microns_per_pixel
    return matching_kernel_dft, derivative_weights_from_blurring


def _derivative_kernels_dft(
    pixel_freq_meshes: tuple[np.ndarray, np.ndarray, np.ndarray],
    derivative_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y_pixel_freq_mesh, x_pixel_freq_mesh, z_pixel_freq_mesh = pixel_freq_meshes
    curvatures_kernels_dft = np.zeros((6, *y_pixel_freq_mesh.shape), dtype=np.float64)
    gradient_kernels_dft = np.zeros((3, *y_pixel_freq_mesh.shape), dtype=np.complex128)

    curvatures_kernels_dft[0] = derivative_weights[0] ** 2 * (
        np.cos(2.0 * np.pi * y_pixel_freq_mesh) - 1.0
    )
    curvatures_kernels_dft[1] = derivative_weights[1] ** 2 * (
        np.cos(2.0 * np.pi * x_pixel_freq_mesh) - 1.0
    )
    curvatures_kernels_dft[2] = derivative_weights[2] ** 2 * (
        np.cos(2.0 * np.pi * z_pixel_freq_mesh) - 1.0
    )

    yx_freq = y_pixel_freq_mesh * x_pixel_freq_mesh
    xz_freq = x_pixel_freq_mesh * z_pixel_freq_mesh
    zy_freq = z_pixel_freq_mesh * y_pixel_freq_mesh
    curvatures_kernels_dft[3] = (
        derivative_weights[0]
        * derivative_weights[1]
        * (np.cos(2.0 * np.pi * np.sqrt(np.abs(yx_freq))) - 1.0)
        * np.sign(yx_freq)
        / 4.0
    )
    curvatures_kernels_dft[4] = (
        derivative_weights[1]
        * derivative_weights[2]
        * (np.cos(2.0 * np.pi * np.sqrt(np.abs(xz_freq))) - 1.0)
        * np.sign(xz_freq)
        / 4.0
    )
    curvatures_kernels_dft[5] = (
        derivative_weights[2]
        * derivative_weights[0]
        * (np.cos(2.0 * np.pi * np.sqrt(np.abs(zy_freq))) - 1.0)
        * np.sign(zy_freq)
        / 4.0
    )

    gradient_kernels_dft[0] = (
        1j * derivative_weights[0] * np.sin(2.0 * np.pi * y_pixel_freq_mesh) / 2.0
    )
    gradient_kernels_dft[1] = (
        1j * derivative_weights[1] * np.sin(2.0 * np.pi * x_pixel_freq_mesh) / 2.0
    )
    gradient_kernels_dft[2] = (
        1j * derivative_weights[2] * np.sin(2.0 * np.pi * z_pixel_freq_mesh) / 2.0
    )
    return curvatures_kernels_dft, gradient_kernels_dft
