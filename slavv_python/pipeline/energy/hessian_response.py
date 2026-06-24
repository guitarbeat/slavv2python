"""Preferred internal name for the native Hessian response backend."""

import gc
import logging
from typing import Any, cast

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.special import jv

from slavv_python.pipeline.energy.math import compute_principal_energy
from slavv_python.pipeline.energy.policy import EnergyPolicy

_WORST_RESOLUTION_TO_DOWNSAMPLE = 1.0 / 2.5
logger = logging.getLogger(__name__)


def matlab_octave_resolution_factors(
    lumen_radius_microns: np.ndarray,
    microns_per_voxel: np.ndarray,
    scales_per_octave: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return MATLAB-style octave ids and per-scale integer downsampling factors."""
    number_of_scales = len(lumen_radius_microns)
    scale_subscripts: np.ndarray = np.arange(1, number_of_scales + 1, dtype=float)
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
            np.floor(_WORST_RESOLUTION_TO_DOWNSAMPLE / resolutions_at_octave + 0.5).astype(
                np.int16
            ),
            1,
        )
        logger.info(f"Octave {current_octave} resolution factors: {resolution_factors}")
        resolution_factors_by_octave[int(current_octave)] = resolution_factors

    # Replicate MATLAB unique(rf_list, 'rows') which sorts lexicographically
    initial_octave_range = np.unique(octave_at_scales)
    initial_rf_list = np.stack(
        [resolution_factors_by_octave[int(o)] for o in initial_octave_range], axis=0
    )

    unique_rf, ic = np.unique(initial_rf_list, axis=0, return_inverse=True)

    # Map original octaves to the consolidated unique octave IDs
    consolidated_octave_at_scales = np.array(
        [ic[int(octave_at_scales[i]) - 1] + 1 for i in range(number_of_scales)],
        dtype=np.int16,
    )

    scale_resolution_factors = unique_rf[ic[octave_at_scales - 1]]

    return consolidated_octave_at_scales, scale_resolution_factors


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
        return energy_3d.astype(np.float64, copy=False), scale_indices

    if projection_mode == "matlab":
        energy_3d = np.min(energy_4d, axis=3)
        scale_indices = np.argmin(energy_4d, axis=3).astype(np.int16)
        return energy_3d.astype(np.float64, copy=False), scale_indices

    annular_indices = np.argmin(energy_4d, axis=3).astype(np.int16)
    annular_energy = np.take_along_axis(
        energy_4d,
        annular_indices[..., None],
        axis=3,
    )[..., 0]

    negative_weights = np.where(np.isfinite(energy_4d) & (energy_4d < 0), -energy_4d, 0.0)
    scale_axis = np.arange(energy_4d.shape[3], dtype=np.float64)
    weighted_sum = np.sum(negative_weights * scale_axis.reshape((1, 1, 1, -1)), axis=3)
    total_weight = np.sum(negative_weights, axis=3)

    spherical_indices = np.divide(
        weighted_sum,
        total_weight,
        out=annular_indices.astype(np.float64),
        where=total_weight > 0,
    )
    blended_indices = spherical_to_annular_ratio * spherical_indices + (
        1.0 - spherical_to_annular_ratio
    ) * annular_indices.astype(np.float64)
    sampled_indices = np.clip(np.rint(blended_indices), 0, energy_4d.shape[3] - 1).astype(np.int16)
    sampled_energy = np.take_along_axis(energy_4d, sampled_indices[..., None], axis=3)[..., 0]

    fallback_mask = ~np.isfinite(sampled_energy)
    if np.any(fallback_mask):
        sampled_energy = sampled_energy.copy()
        sampled_indices = sampled_indices.copy()
        sampled_energy[fallback_mask] = annular_energy[fallback_mask]
        sampled_indices[fallback_mask] = annular_indices[fallback_mask]

    return sampled_energy.astype(np.float64, copy=False), sampled_indices


def compute_native_hessian_energy(
    image: np.ndarray,
    config: dict[str, Any],
    scale_idx: int,
) -> np.ndarray:
    """Compute one scale of the MATLAB-style matched-filter Hessian energy."""
    policy = EnergyPolicy.from_params(config)
    debug_outputs = _compute_native_hessian_scale_debug(image, config, scale_idx, policy=policy)
    return debug_outputs["energy"]


def _compute_native_hessian_scale_debug(
    image: np.ndarray,
    config: dict[str, Any],
    scale_idx: int,
    policy: EnergyPolicy | None = None,
) -> dict[str, np.ndarray]:
    """Return one scale of native Hessian intermediates on the working grid."""
    policy = policy or EnergyPolicy.from_params(config)
    resolution_factor = np.asarray(config["scale_resolution_factors"][scale_idx], dtype=np.int16)
    radius_microns = float(config["lumen_radius_microns"][scale_idx])
    microns_per_pixel: np.ndarray = (
        np.asarray(config["microns_per_voxel"], dtype=float) * resolution_factor
    )
    pixels_per_sigma_psf = (
        np.asarray(config["pixels_per_sigma_PSF"], dtype=float) / resolution_factor
    )

    working_image = _downsample_volume(image, resolution_factor, policy=policy)
    debug_outputs = _matched_hessian_intermediates(
        working_image.astype(policy.precision, copy=False),
        radius_of_lumen_in_microns=radius_microns,
        microns_per_pixel=microns_per_pixel,
        pixels_per_sigma_psf=pixels_per_sigma_psf,
        gaussian_to_ideal_ratio=float(config["gaussian_to_ideal_ratio"]),
        spherical_to_annular_ratio=float(config["spherical_to_annular_ratio"]),
        policy=policy,
    )
    return {
        "resolution_factor": resolution_factor,
        "laplacian": debug_outputs["laplacian"],
        "valid_voxels": debug_outputs["valid_voxels"],
        "energy": _upsample_volume(debug_outputs["energy"], image.shape, resolution_factor),
    }


def _downsample_volume(
    image: np.ndarray,
    resolution_factor: np.ndarray,
    policy: EnergyPolicy | None = None,
) -> np.ndarray:
    """Downsample with configurable stride phase alignment."""
    factors = [int(value) for value in resolution_factor]
    if factors[0] == factors[1] == factors[2] == 1:
        return image

    alignment = policy.downsample_alignment if policy else "paper"
    if alignment == "matlab":
        # MATLAB ``get_starts_and_counts_V200`` adjusts phase per axis.
        starts = [(image.shape[axis] - 1) % factors[axis] for axis in range(3)]
    else:
        starts = [0, 0, 0]

    return cast(
        "np.ndarray",
        image[
            starts[0] :: factors[0],
            starts[1] :: factors[1],
            starts[2] :: factors[2],
        ],
    )


def _upsample_volume(
    volume: np.ndarray,
    output_shape: tuple[int, int, int],
    resolution_factor: np.ndarray,
) -> np.ndarray:
    factor_y, factor_x, factor_z = (float(value) for value in resolution_factor)
    if factor_y == factor_x == factor_z == 1.0 and volume.shape == output_shape:
        return cast("np.ndarray", volume.astype(np.float64, copy=False))

    coord_y = np.arange(output_shape[0], dtype=np.float64) / factor_y
    coord_x = np.arange(output_shape[1], dtype=np.float64) / factor_x
    coord_z = np.arange(output_shape[2], dtype=np.float64) / factor_z
    mesh: tuple[np.ndarray, np.ndarray, np.ndarray] = np.meshgrid(
        coord_y, coord_x, coord_z, indexing="ij"
    )
    coordinates = np.asarray(mesh, dtype=np.float64)

    source: np.ndarray = volume.astype(np.float64, copy=False)
    finite_mask = np.isfinite(source)
    filled = np.where(finite_mask, source, 0.0).astype(np.float64, copy=False)
    weights = finite_mask.astype(np.float64, copy=False)

    value_sum = map_coordinates(
        filled,
        coordinates,
        order=1,
        mode="nearest",
        prefilter=False,
    )
    weight_sum = map_coordinates(
        weights,
        coordinates,
        order=1,
        mode="nearest",
        prefilter=False,
    )
    upsampled: np.ndarray = np.full(output_shape, np.inf, dtype=np.float64)
    valid = weight_sum > 0.0
    upsampled[valid] = (value_sum[valid] / weight_sum[valid]).astype(np.float64, copy=False)
    return cast("np.ndarray", upsampled)


def _matched_hessian_energy(
    image: np.ndarray,
    *,
    radius_of_lumen_in_microns: float,
    microns_per_pixel: np.ndarray,
    pixels_per_sigma_psf: np.ndarray,
    gaussian_to_ideal_ratio: float,
    spherical_to_annular_ratio: float,
    policy: EnergyPolicy | None = None,
) -> np.ndarray:
    return _matched_hessian_intermediates(
        image,
        radius_of_lumen_in_microns=radius_of_lumen_in_microns,
        microns_per_pixel=microns_per_pixel,
        pixels_per_sigma_psf=pixels_per_sigma_psf,
        gaussian_to_ideal_ratio=gaussian_to_ideal_ratio,
        spherical_to_annular_ratio=spherical_to_annular_ratio,
        policy=policy,
    )["energy"]


def _matched_hessian_intermediates(
    image: np.ndarray,
    *,
    radius_of_lumen_in_microns: float,
    microns_per_pixel: np.ndarray,
    pixels_per_sigma_psf: np.ndarray,
    gaussian_to_ideal_ratio: float,
    spherical_to_annular_ratio: float,
    energy_sign: float = -1.0,
    policy: EnergyPolicy | None = None,
) -> dict[str, np.ndarray]:
    dtype = policy.precision if policy else np.float64
    image = image.astype(dtype, copy=False)
    original_shape = image.shape
    padded_image = _fourier_transform_input(image)
    chunk_dft = np.fft.fftn(padded_image.astype(dtype, copy=False))
    pixel_freq_meshes = _pixel_frequency_meshes(padded_image.shape)
    matching_kernel_dft, derivative_weights = _matching_kernel_dft(
        pixel_freq_meshes,
        radius_of_lumen_in_microns=radius_of_lumen_in_microns,
        microns_per_pixel=microns_per_pixel,
        pixels_per_sigma_psf=pixels_per_sigma_psf,
        gaussian_to_ideal_ratio=gaussian_to_ideal_ratio,
        spherical_to_annular_ratio=spherical_to_annular_ratio,
    )

    filtered_chunk_dft = (matching_kernel_dft * chunk_dft).astype(np.complex128, copy=False)
    del matching_kernel_dft

    # Compute curvature components one-by-one to avoid 4D stacks.
    c0 = _ifftn_matlab_symmetric(
        _derivative_kernel_dft_single(
            pixel_freq_meshes, derivative_weights, 0, is_curvature=True
        ).astype(np.float64, copy=False)
        * filtered_chunk_dft
    )[: original_shape[0], : original_shape[1], : original_shape[2]]

    c1 = _ifftn_matlab_symmetric(
        _derivative_kernel_dft_single(
            pixel_freq_meshes, derivative_weights, 1, is_curvature=True
        ).astype(np.float64, copy=False)
        * filtered_chunk_dft
    )[: original_shape[0], : original_shape[1], : original_shape[2]]

    c2 = _ifftn_matlab_symmetric(
        _derivative_kernel_dft_single(
            pixel_freq_meshes, derivative_weights, 2, is_curvature=True
        ).astype(np.float64, copy=False)
        * filtered_chunk_dft
    )[: original_shape[0], : original_shape[1], : original_shape[2]]

    laplacian_chunk = c0 + c1 + c2
    valid_voxels = (laplacian_chunk < 0) if energy_sign < 0 else (laplacian_chunk > 0)
    energy_chunk = np.full(image.shape, np.inf if energy_sign < 0 else -np.inf, dtype=dtype)

    if not np.any(valid_voxels):
        del c0, c1, c2, filtered_chunk_dft
        return {
            "laplacian": laplacian_chunk.astype(dtype, copy=False),
            "valid_voxels": valid_voxels,
            "energy": energy_chunk,
        }

    num_valid = np.count_nonzero(valid_voxels)
    curvatures_valid: np.ndarray = np.empty((num_valid, 6), dtype=dtype)
    curvatures_valid[:, 0] = c0[valid_voxels].astype(dtype, copy=False)
    curvatures_valid[:, 1] = c1[valid_voxels].astype(dtype, copy=False)
    curvatures_valid[:, 2] = c2[valid_voxels].astype(dtype, copy=False)
    del c0, c1, c2

    for i, col_idx in enumerate([3, 4, 5], start=3):
        curvatures_valid[:, col_idx] = _ifftn_matlab_symmetric(
            _derivative_kernel_dft_single(
                pixel_freq_meshes, derivative_weights, i, is_curvature=True
            ).astype(np.float64, copy=False)
            * filtered_chunk_dft
        )[: original_shape[0], : original_shape[1], : original_shape[2]][valid_voxels].astype(
            dtype, copy=False
        )

    grad_valid: np.ndarray = np.empty((num_valid, 3), dtype=dtype)
    for i in range(3):
        grad_valid[:, i] = _ifftn_matlab_symmetric(
            _derivative_kernel_dft_single(
                pixel_freq_meshes, derivative_weights, i, is_curvature=False
            ).astype(np.float64, copy=False)
            * filtered_chunk_dft
        )[: original_shape[0], : original_shape[1], : original_shape[2]][valid_voxels].astype(
            dtype, copy=False
        )

    del filtered_chunk_dft
    gc.collect()

    energy_valid = compute_principal_energy(
        grad_valid, curvatures_valid, energy_sign=energy_sign, dtype=dtype
    )

    del grad_valid, curvatures_valid
    energy_chunk[valid_voxels] = energy_valid

    return {
        "laplacian": laplacian_chunk.astype(dtype, copy=False),
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
    return cast("np.ndarray", np.pad(image, pad_width, mode="symmetric"))


def _ifftn_matlab_symmetric(spectrum: np.ndarray) -> np.ndarray:
    """Match MATLAB ``ifftn(..., 'symmetric')`` for conjugate-pair rounding drift.

    Memory-efficient implementation: avoids full 3D integer meshgrids by computing
    partner subscripts only for the masked voxels, keeping peak allocation proportional
    to mask.sum() rather than the full volume.  This prevents heap-fragmentation OOM
    on large chunks where three full (Y,X,Z) int64 meshgrids would all be live at once.
    """
    spectrum_arr = np.asarray(spectrum)
    ny, nx, nz = spectrum_arr.shape

    # 1-D partner-index arrays (cyclic flip: partner(i) = (-i) % N).
    # Fortran (column-major) linear index of each voxel is:
    #   lin(y,x,z) = y + ny*x + ny*nx*z
    # Partner linear index:
    #   plin(y,x,z) = py + ny*px + ny*nx*pz   where py=(-y)%ny, etc.
    p_idx_y = (-np.arange(ny)) % ny  # shape (ny,)
    p_idx_x = (-np.arange(nx)) % nx  # shape (nx,)
    p_idx_z = (-np.arange(nz)) % nz  # shape (nz,)

    # Fortran-order linear index for every voxel — compact 1-D arithmetic, no meshgrid.
    #   lin(y,x,z)  = y  + ny*x  + ny*nx*z
    #   plin(y,x,z) = py + ny*px + ny*nx*pz
    # Broadcast over axes using outer products on the three 1-D vectors.
    lin_y = np.arange(ny)
    lin_x = np.arange(nx)
    lin_z = np.arange(nz)

    # Compute partner linear index for each voxel via outer-product broadcasting.
    # Each term has shape compatible with (ny, nx, nz):
    #   plin = p_idx_y[:,None,None] + ny*p_idx_x[None,:,None] + ny*nx*p_idx_z[None,None,:]
    plin = (
        p_idx_y[:, None, None] + ny * p_idx_x[None, :, None] + ny * nx * p_idx_z[None, None, :]
    )  # shape (ny, nx, nz), dtype int64
    del p_idx_y, p_idx_x, p_idx_z

    # Own Fortran-order linear index.
    lin = (
        lin_y[:, None, None] + ny * lin_x[None, :, None] + ny * nx * lin_z[None, None, :]
    )  # shape (ny, nx, nz), dtype int64
    del lin_y, lin_x, lin_z

    # Mask: voxels whose partner has a *smaller* linear index — these need overwriting.
    mask = lin > plin  # shape (ny, nx, nz), bool
    self_partner_mask = lin == plin  # shape (ny, nx, nz), bool
    del lin, plin

    symmetric_spectrum = spectrum_arr.copy()

    if np.any(mask):
        # Decode subscripts only for masked voxels — avoids full meshgrids.
        flat_mask = mask.ravel(order="F")  # Fortran order matches lin computation
        masked_lin_flat = np.where(flat_mask)[0]  # 1-D Fortran linear indices of masked voxels

        # Decode (y, x, z) subscripts from Fortran linear index.
        iy = masked_lin_flat % ny
        tmp = masked_lin_flat // ny
        ix = tmp % nx
        iz = tmp // nx
        del tmp, masked_lin_flat, flat_mask

        # Compute partner subscripts for masked voxels only.
        py = (-iy) % ny
        px = (-ix) % nx
        pz = (-iz) % nz

        symmetric_spectrum[iy, ix, iz] = np.conj(spectrum_arr[py, px, pz])
        del iy, ix, iz, py, px, pz

    del mask

    if np.any(self_partner_mask):
        symmetric_spectrum[self_partner_mask] = symmetric_spectrum[self_partner_mask].real
    del self_partner_mask

    result = np.fft.ifftn(symmetric_spectrum).real
    del symmetric_spectrum
    return cast("np.ndarray", result)


def _pixel_frequency_meshes(
    shape: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pixel_frequencies = [np.fft.fftfreq(length) for length in shape]
    y_mesh: np.ndarray
    x_mesh: np.ndarray
    z_mesh: np.ndarray
    y_mesh, x_mesh, z_mesh = np.meshgrid(
        pixel_frequencies[0],
        pixel_frequencies[1],
        pixel_frequencies[2],
        indexing="ij",
        sparse=True,
    )
    return y_mesh, x_mesh, z_mesh


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

    microns_per_sigma_psf: np.ndarray = pixels_per_sigma_psf * microns_per_pixel
    gaussian_lengths: np.ndarray = gaussian_to_ideal_ratio * radius_of_lumen_in_microns + np.zeros(
        3
    )
    annular_pulse_lengths_squared: np.ndarray = np.asarray(
        (1.0 - gaussian_to_ideal_ratio**2) * radius_of_lumen_in_microns**2
        + microns_per_sigma_psf**2,
        dtype=np.float64,
    )
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

    # Compute Bessel sum in chunks to keep peak memory footprint minimal
    res = np.empty_like(sphere_argument)
    chunk_size = 1000000
    for i in range(0, len(sphere_argument), chunk_size):
        s = slice(i, i + chunk_size)
        c = sphere_argument[s]
        # Match MATLAB ``energy_filter_V200`` evaluation order:
        # (pi/2./radial).^0.5 .* (besselj(2.5,radial)+besselj(0.5,radial))
        res[s] = np.sqrt(np.pi / 2.0 / c) * (jv(2.5, c) + jv(0.5, c))

    spherical_pulse_kernel_dft[nonzero_sphere] = res
    del res, sphere_argument

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
    derivative_weights_from_blurring: np.ndarray = gaussian_lengths / microns_per_pixel
    return matching_kernel_dft, derivative_weights_from_blurring


def _precompute_base_derivative_kernels_dft(
    pixel_freq_meshes: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Pre-compute scale-independent parts of the derivative kernels.

    Args:
        pixel_freq_meshes: tuple of (y, x, z) pixel frequency meshes.

    Returns:
        tuple of (base_curvatures, base_gradients)
    """
    y_pixel_freq_mesh, x_pixel_freq_mesh, z_pixel_freq_mesh = pixel_freq_meshes
    # Determine full shape from sparse meshes for pre-allocation
    shape = np.broadcast_shapes(
        y_pixel_freq_mesh.shape, x_pixel_freq_mesh.shape, z_pixel_freq_mesh.shape
    )

    # Base derivative kernels (unweighted)
    base_curvatures = np.zeros((6, *shape), dtype=np.float64)
    base_gradients = np.zeros((3, *shape), dtype=np.complex128)

    base_curvatures[0] = np.cos(2.0 * np.pi * y_pixel_freq_mesh) - 1.0
    base_curvatures[1] = np.cos(2.0 * np.pi * x_pixel_freq_mesh) - 1.0
    base_curvatures[2] = np.cos(2.0 * np.pi * z_pixel_freq_mesh) - 1.0

    yx_freq: np.ndarray = y_pixel_freq_mesh * x_pixel_freq_mesh
    xz_freq: np.ndarray = x_pixel_freq_mesh * z_pixel_freq_mesh
    zy_freq: np.ndarray = z_pixel_freq_mesh * y_pixel_freq_mesh

    base_curvatures[3] = (
        (np.cos(2.0 * np.pi * np.sqrt(np.abs(yx_freq))) - 1.0) * np.sign(yx_freq) / 4.0
    )
    base_curvatures[4] = (
        (np.cos(2.0 * np.pi * np.sqrt(np.abs(xz_freq))) - 1.0) * np.sign(xz_freq) / 4.0
    )
    base_curvatures[5] = (
        (np.cos(2.0 * np.pi * np.sqrt(np.abs(zy_freq))) - 1.0) * np.sign(zy_freq) / 4.0
    )

    base_gradients[0] = 1j * np.sin(2.0 * np.pi * y_pixel_freq_mesh) / 2.0
    base_gradients[1] = 1j * np.sin(2.0 * np.pi * x_pixel_freq_mesh) / 2.0
    base_gradients[2] = 1j * np.sin(2.0 * np.pi * z_pixel_freq_mesh) / 2.0

    return base_curvatures, base_gradients


def _derivative_kernel_dft_single(
    pixel_freq_meshes: tuple[np.ndarray, np.ndarray, np.ndarray],
    derivative_weights: np.ndarray,
    kernel_index: int,
    is_curvature: bool = True,
) -> np.ndarray:
    """Compute a single scale-dependent derivative kernel to save memory."""
    y_mesh, x_mesh, z_mesh = pixel_freq_meshes
    if is_curvature:
        if kernel_index == 0:
            return cast(
                "np.ndarray",
                (derivative_weights[0] ** 2) * (np.cos(2.0 * np.pi * y_mesh) - 1.0),
            )
        if kernel_index == 1:
            return cast(
                "np.ndarray",
                (derivative_weights[1] ** 2) * (np.cos(2.0 * np.pi * x_mesh) - 1.0),
            )
        if kernel_index == 2:
            return cast(
                "np.ndarray",
                (derivative_weights[2] ** 2) * (np.cos(2.0 * np.pi * z_mesh) - 1.0),
            )

        if kernel_index == 3:
            yx_freq = y_mesh * x_mesh
            return cast(
                "np.ndarray",
                derivative_weights[0]
                * derivative_weights[1]
                * (np.cos(2.0 * np.pi * np.sqrt(np.abs(yx_freq))) - 1.0)
                * np.sign(yx_freq)
                / 4.0,
            )
        if kernel_index == 4:
            xz_freq = x_mesh * z_mesh
            return cast(
                "np.ndarray",
                derivative_weights[1]
                * derivative_weights[2]
                * (np.cos(2.0 * np.pi * np.sqrt(np.abs(xz_freq))) - 1.0)
                * np.sign(xz_freq)
                / 4.0,
            )
        if kernel_index == 5:
            zy_freq = z_mesh * y_mesh
            return cast(
                "np.ndarray",
                derivative_weights[2]
                * derivative_weights[0]
                * (np.cos(2.0 * np.pi * np.sqrt(np.abs(zy_freq))) - 1.0)
                * np.sign(zy_freq)
                / 4.0,
            )
    else:
        if kernel_index == 0:
            return cast(
                "np.ndarray",
                1j * derivative_weights[0] * np.sin(2.0 * np.pi * y_mesh) / 2.0,
            )
        if kernel_index == 1:
            return cast(
                "np.ndarray",
                1j * derivative_weights[1] * np.sin(2.0 * np.pi * x_mesh) / 2.0,
            )
        if kernel_index == 2:
            return cast(
                "np.ndarray",
                1j * derivative_weights[2] * np.sin(2.0 * np.pi * z_mesh) / 2.0,
            )

    raise ValueError(f"Invalid kernel_index {kernel_index}")


def _derivative_kernels_dft(
    pixel_freq_meshes: tuple[np.ndarray, np.ndarray, np.ndarray],
    derivative_weights: np.ndarray,
    base_kernels: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute scale-dependent derivative kernels.

    Args:
        pixel_freq_meshes: tuple of (y, x, z) pixel frequency meshes.
        derivative_weights: (3,) array of weights for y, x, z derivatives.
        base_kernels: Optional pre-computed base kernels from _precompute_base_derivative_kernels_dft.

    Returns:
        tuple of (curvatures_kernels_dft, gradient_kernels_dft)
    """
    if base_kernels is not None:
        base_curvatures, base_gradients = base_kernels
        curvatures_kernels_dft = np.empty_like(base_curvatures)
        curvatures_kernels_dft[0] = (derivative_weights[0] ** 2) * base_curvatures[0]
        curvatures_kernels_dft[1] = (derivative_weights[1] ** 2) * base_curvatures[1]
        curvatures_kernels_dft[2] = (derivative_weights[2] ** 2) * base_curvatures[2]
        curvatures_kernels_dft[3] = (
            derivative_weights[0] * derivative_weights[1] * base_curvatures[3]
        )
        curvatures_kernels_dft[4] = (
            derivative_weights[1] * derivative_weights[2] * base_curvatures[4]
        )
        curvatures_kernels_dft[5] = (
            derivative_weights[2] * derivative_weights[0] * base_curvatures[5]
        )

        gradient_kernels_dft = np.empty_like(base_gradients)
        gradient_kernels_dft[0] = derivative_weights[0] * base_gradients[0]
        gradient_kernels_dft[1] = derivative_weights[1] * base_gradients[1]
        gradient_kernels_dft[2] = derivative_weights[2] * base_gradients[2]

        return curvatures_kernels_dft, gradient_kernels_dft

    # Fallback to full computation
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

    yx_freq: np.ndarray = y_pixel_freq_mesh * x_pixel_freq_mesh
    xz_freq: np.ndarray = x_pixel_freq_mesh * z_pixel_freq_mesh
    zy_freq: np.ndarray = z_pixel_freq_mesh * y_pixel_freq_mesh
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


__all__ = [
    "_compute_native_hessian_scale_debug",
    "_derivative_kernels_dft",
    "_downsample_volume",
    "_fourier_transform_input",
    "_ifftn_matlab_symmetric",
    "_matched_hessian_energy",
    "_matched_hessian_intermediates",
    "_matching_kernel_dft",
    "_pixel_frequency_meshes",
    "_upsample_volume",
    "compute_native_hessian_energy",
    "matlab_octave_resolution_factors",
    "project_energy_stack",
    "required_scale_stack",
]
