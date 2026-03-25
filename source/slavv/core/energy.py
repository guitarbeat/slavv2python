"""
Energy field calculations for SLAVV.
Includes Hessian-based vessel enhancement (Frangi/Sato) and Numba-accelerated gradient computation.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.ndimage import gaussian_filter
from skimage import feature

if TYPE_CHECKING:
    from slavv.runtime import StageController

try:
    from skimage.filters import frangi, sato
except ImportError:
    try:
        from skimage.filters import frangi

        sato = None
    except ImportError:
        frangi = None
        sato = None

# Optional Numba acceleration
# Currently disabled: Numba 0.43.1 is incompatible with NumPy 1.21.6+
# To enable, upgrade Numba to 0.50+ or set _NUMBA_AVAILABLE = True after verifying compatibility
_NUMBA_AVAILABLE = False
try:
    from numba import njit
except ImportError:
    njit = None

logger = logging.getLogger(__name__)

# --- Numba Accelerated Gradient Computation ---

if _NUMBA_AVAILABLE:

    @njit
    def compute_gradient_impl(energy, pos_int, mpv):
        """Compute local energy gradient via central differences (Numba accelerated)."""
        dim_y = energy.shape[0]
        dim_x = energy.shape[1]
        dim_z = energy.shape[2]

        if dim_y < 3 or dim_x < 3 or dim_z < 3:
            return np.zeros(3, dtype=np.float64)

        # Manual clamping to [1, shape-2]
        pos_y = int(pos_int[0])
        if pos_y < 1:
            pos_y = 1
        elif pos_y > dim_y - 2:
            pos_y = dim_y - 2

        pos_x = int(pos_int[1])
        if pos_x < 1:
            pos_x = 1
        elif pos_x > dim_x - 2:
            pos_x = dim_x - 2

        pos_z = int(pos_int[2])
        if pos_z < 1:
            pos_z = 1
        elif pos_z > dim_z - 2:
            pos_z = dim_z - 2

        grad = np.zeros(3, dtype=np.float64)
        # Use explicit indexing for speed and clarity
        grad[0] = (energy[pos_y + 1, pos_x, pos_z] - energy[pos_y - 1, pos_x, pos_z]) / (
            2.0 * mpv[0]
        )
        grad[1] = (energy[pos_y, pos_x + 1, pos_z] - energy[pos_y, pos_x - 1, pos_z]) / (
            2.0 * mpv[1]
        )
        grad[2] = (energy[pos_y, pos_x, pos_z + 1] - energy[pos_y, pos_x, pos_z - 1]) / (
            2.0 * mpv[2]
        )

        return grad

else:

    def compute_gradient_impl(energy, pos_int, microns_per_voxel):
        """Compute local energy gradient via central differences."""
        grad = np.zeros(3, dtype=float)

        # Unpack position and shape
        pos_y, pos_x, pos_z = pos_int
        shape_y, shape_x, shape_z = energy.shape

        # Check for small volume
        if shape_y < 3 or shape_x < 3 or shape_z < 3:
            return np.zeros(3, dtype=float)

        # Manual clamping to [1, shape-2] to prevent out-of-bounds access
        if pos_y < 1:
            pos_y = 1
        elif pos_y > shape_y - 2:
            pos_y = shape_y - 2

        if pos_x < 1:
            pos_x = 1
        elif pos_x > shape_x - 2:
            pos_x = shape_x - 2

        if pos_z < 1:
            pos_z = 1
        elif pos_z > shape_z - 2:
            pos_z = shape_z - 2

        # Direct indexing for speed (avoids allocations and tuple overhead)
        grad[0] = (energy[pos_y + 1, pos_x, pos_z] - energy[pos_y - 1, pos_x, pos_z]) / (
            2.0 * microns_per_voxel[0]
        )
        grad[1] = (energy[pos_y, pos_x + 1, pos_z] - energy[pos_y, pos_x - 1, pos_z]) / (
            2.0 * microns_per_voxel[1]
        )
        grad[2] = (energy[pos_y, pos_x, pos_z + 1] - energy[pos_y, pos_x, pos_z - 1]) / (
            2.0 * microns_per_voxel[2]
        )

        return grad

# --- Helper Functions ---


def spherical_structuring_element(radius: int, microns_per_voxel: np.ndarray) -> np.ndarray:
    """
    Create a 3D spherical structuring element accounting for voxel spacing.

    The ``radius`` is interpreted in voxel units along the smallest physical
    dimension.  For anisotropic data the resulting footprint becomes an
    ellipsoid so that voxels within ``radius`` microns of the origin are
    included regardless of spacing differences along ``y, x, z``.
    """
    microns_per_voxel = np.asarray(microns_per_voxel, dtype=float)
    r_phys = float(radius) * microns_per_voxel.min()
    ranges = [
        np.arange(-int(np.ceil(r_phys / s)), int(np.ceil(r_phys / s)) + 1)
        for s in microns_per_voxel
    ]
    yy, xx, zz = np.meshgrid(*ranges, indexing="ij")
    dist2 = (
        (yy * microns_per_voxel[0]) ** 2
        + (xx * microns_per_voxel[1]) ** 2
        + (zz * microns_per_voxel[2]) ** 2
    )
    return dist2 <= r_phys**2


def _matlab_lumen_radius_range(
    radius_smallest: float, radius_largest: float, scales_per_octave: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return MATLAB-aligned scale ordinates and lumen radii.

    MATLAB defines ``scales_per_octave`` per doubling of the vessel radius cubed
    and pads the requested range by one scale on each side so 4D extrema can be
    detected during vertex extraction.
    """
    largest_per_smallest_volume_ratio = (radius_largest / radius_smallest) ** 3
    final_scale = int(np.round(np.log2(largest_per_smallest_volume_ratio) * scales_per_octave))
    scale_ordinates = np.arange(-1, final_scale + 2, dtype=float)
    scale_factors = 2 ** (scale_ordinates / scales_per_octave / 3.0)
    return scale_ordinates, radius_smallest * scale_factors


def _matched_filter_derivative(
    image: np.ndarray,
    sigma_object: np.ndarray,
    sigma_background: np.ndarray | None,
    spherical_to_annular_ratio: float,
    order: tuple[int, int, int],
    microns_per_voxel: np.ndarray,
) -> np.ndarray:
    """Evaluate a matched-kernel derivative in physical units."""
    derivative = gaussian_filter(image, sigma=tuple(sigma_object), order=order)
    spacing_scale = np.prod(np.power(microns_per_voxel, order))
    if spacing_scale > 0:
        derivative = derivative / spacing_scale
    sigma_object_physical = np.asarray(sigma_object, dtype=float) * microns_per_voxel
    object_normalization = np.prod(np.power(sigma_object_physical, order))
    if object_normalization > 0:
        derivative = derivative * object_normalization
    if sigma_background is not None and spherical_to_annular_ratio < 1.0:
        background = gaussian_filter(image, sigma=tuple(sigma_background), order=order)
        if spacing_scale > 0:
            background = background / spacing_scale
        sigma_background_physical = np.asarray(sigma_background, dtype=float) * microns_per_voxel
        background_normalization = np.prod(np.power(sigma_background_physical, order))
        if background_normalization > 0:
            background = background * background_normalization
        derivative = spherical_to_annular_ratio * derivative + (
            1.0 - spherical_to_annular_ratio
        ) * (derivative - background)
    return derivative.astype(np.float32, copy=False)


def _matlab_hessian_energy(
    image: np.ndarray,
    sigma_object: np.ndarray,
    sigma_background: np.ndarray | None,
    spherical_to_annular_ratio: float,
    microns_per_voxel: np.ndarray,
    energy_sign: float,
) -> np.ndarray:
    """Approximate MATLAB's curvature-weighted Hessian energy response."""
    grad_y = _matched_filter_derivative(
        image,
        sigma_object,
        sigma_background,
        spherical_to_annular_ratio,
        (1, 0, 0),
        microns_per_voxel,
    )
    grad_x = _matched_filter_derivative(
        image,
        sigma_object,
        sigma_background,
        spherical_to_annular_ratio,
        (0, 1, 0),
        microns_per_voxel,
    )
    grad_z = _matched_filter_derivative(
        image,
        sigma_object,
        sigma_background,
        spherical_to_annular_ratio,
        (0, 0, 1),
        microns_per_voxel,
    )
    h_yy = _matched_filter_derivative(
        image,
        sigma_object,
        sigma_background,
        spherical_to_annular_ratio,
        (2, 0, 0),
        microns_per_voxel,
    )
    h_xx = _matched_filter_derivative(
        image,
        sigma_object,
        sigma_background,
        spherical_to_annular_ratio,
        (0, 2, 0),
        microns_per_voxel,
    )
    h_zz = _matched_filter_derivative(
        image,
        sigma_object,
        sigma_background,
        spherical_to_annular_ratio,
        (0, 0, 2),
        microns_per_voxel,
    )
    h_yx = _matched_filter_derivative(
        image,
        sigma_object,
        sigma_background,
        spherical_to_annular_ratio,
        (1, 1, 0),
        microns_per_voxel,
    )
    h_xz = _matched_filter_derivative(
        image,
        sigma_object,
        sigma_background,
        spherical_to_annular_ratio,
        (0, 1, 1),
        microns_per_voxel,
    )
    h_yz = _matched_filter_derivative(
        image,
        sigma_object,
        sigma_background,
        spherical_to_annular_ratio,
        (1, 0, 1),
        microns_per_voxel,
    )

    laplacian = h_yy + h_xx + h_zz
    if energy_sign < 0:
        valid = laplacian < 0
        energy = np.full(image.shape, np.inf, dtype=np.float32)
    else:
        valid = laplacian > 0
        energy = np.full(image.shape, -np.inf, dtype=np.float32)
    if not np.any(valid):
        return energy

    grad_valid = np.stack([grad_y[valid], grad_x[valid], grad_z[valid]], axis=1).astype(np.float64)
    hessian_valid = np.empty((grad_valid.shape[0], 3, 3), dtype=np.float64)
    hessian_valid[:, 0, 0] = h_yy[valid]
    hessian_valid[:, 0, 1] = h_yx[valid]
    hessian_valid[:, 0, 2] = h_yz[valid]
    hessian_valid[:, 1, 0] = h_yx[valid]
    hessian_valid[:, 1, 1] = h_xx[valid]
    hessian_valid[:, 1, 2] = h_xz[valid]
    hessian_valid[:, 2, 0] = h_yz[valid]
    hessian_valid[:, 2, 1] = h_xz[valid]
    hessian_valid[:, 2, 2] = h_zz[valid]

    eigvals, eigvecs = np.linalg.eigh(hessian_valid)
    projections = np.einsum("ni,nik->nk", grad_valid, eigvecs)
    denom = np.where(np.abs(eigvals) > 1e-12, eigvals, np.where(eigvals >= 0, 1e-12, -1e-12))
    principal_energy = eigvals * np.exp(-0.5 * np.square(projections / denom))

    if energy_sign < 0:
        principal_energy[:, 2] = np.minimum(principal_energy[:, 2], 0.0)
        energy_valid = principal_energy.sum(axis=1)
        energy_valid[~np.isfinite(energy_valid)] = np.inf
        energy_valid[energy_valid >= 0] = np.inf
    else:
        principal_energy[:, 0] = np.maximum(principal_energy[:, 0], 0.0)
        energy_valid = principal_energy.sum(axis=1)
        energy_valid[~np.isfinite(energy_valid)] = -np.inf
        energy_valid[energy_valid <= 0] = -np.inf

    energy[valid] = energy_valid.astype(np.float32, copy=False)
    return energy


def calculate_energy_field(
    image: np.ndarray, params: dict[str, Any], get_chunking_lattice_func=None
) -> dict[str, Any]:
    """
    Calculate multi-scale energy field using Hessian-based filtering.

    This implements the energy calculation from ``get_energy_V202`` in
    MATLAB, including PSF prefiltering and configurable Gaussian/annular
    ratios. Set ``energy_method='frangi'`` or ``'sato'`` in ``params`` to use
    scikit-image's :func:`~skimage.filters.frangi` or
    :func:`~skimage.filters.sato` vesselness filters as alternative backends.
    """
    logger.info("Calculating energy field")

    # Extract parameters with defaults from MATLAB
    microns_per_voxel = params.get("microns_per_voxel", [1.0, 1.0, 1.0])
    radius_smallest = params.get("radius_of_smallest_vessel_in_microns", 1.5)
    radius_largest = params.get("radius_of_largest_vessel_in_microns", 50.0)
    scales_per_octave = params.get("scales_per_octave", 1.5)
    gaussian_to_ideal_ratio = params.get("gaussian_to_ideal_ratio", 1.0)
    spherical_to_annular_ratio = params.get("spherical_to_annular_ratio", 1.0)
    approximating_PSF = params.get("approximating_PSF", True)
    energy_sign = params.get("energy_sign", -1.0)  # -1 for bright vessels
    return_all_scales = params.get("return_all_scales", False)
    energy_method = params.get("energy_method", "hessian")

    # Cache voxel spacing for anisotropy handling
    voxel_size = np.array(microns_per_voxel, dtype=float)

    # PSF calculation (from MATLAB implementation)
    if approximating_PSF:
        numerical_aperture = params.get("numerical_aperture", 0.95)
        excitation_wavelength = params.get("excitation_wavelength_in_microns", 1.3)
        sample_index_of_refraction = params.get("sample_index_of_refraction", 1.33)

        # PSF calculation based on Zipfel et al.
        if numerical_aperture <= 0.7:
            coefficient, exponent = 0.320, 1.0
        else:
            coefficient, exponent = 0.325, 0.91

        microns_per_sigma_PSF = [
            excitation_wavelength / (2**0.5) * coefficient / (numerical_aperture**exponent),
            excitation_wavelength / (2**0.5) * coefficient / (numerical_aperture**exponent),
            excitation_wavelength
            / (2**0.5)
            * 0.532
            / (
                sample_index_of_refraction
                - (sample_index_of_refraction**2 - numerical_aperture**2) ** 0.5
            ),
        ]
    else:
        microns_per_sigma_PSF = [0.0, 0.0, 0.0]

    microns_per_sigma_PSF = np.array(microns_per_sigma_PSF, dtype=float)
    pixels_per_sigma_PSF = microns_per_sigma_PSF / voxel_size

    # Calculate scale range following MATLAB ordination.
    _, lumen_radius_microns = _matlab_lumen_radius_range(
        radius_smallest,
        radius_largest,
        scales_per_octave,
    )
    n_scales = len(lumen_radius_microns)

    # Convert radii to pixels per axis then average for scalar pixel radii
    lumen_radius_pixels_axes = lumen_radius_microns[:, None] / voxel_size[None, :]
    lumen_radius_pixels = lumen_radius_pixels_axes.mean(axis=1)

    # Use float32 image once for all scales
    image = image.astype(np.float32)

    total_voxels = int(np.prod(image.shape))
    max_voxels = int(params.get("max_voxels_per_node_energy", 1e5))
    if total_voxels > max_voxels:
        max_sigma = (lumen_radius_microns[-1] / voxel_size) / max(gaussian_to_ideal_ratio, 1e-12)
        if approximating_PSF:
            max_sigma = np.sqrt(max_sigma**2 + pixels_per_sigma_PSF**2)
        margin = int(np.ceil(np.max(max_sigma)))
        lattice = get_chunking_lattice_func(image.shape, max_voxels, margin)
        if return_all_scales:
            energy_4d = np.zeros((*image.shape, n_scales), dtype=np.float32)
        else:
            energy_3d = np.empty(image.shape, dtype=np.float32)
            scale_indices = np.empty(image.shape, dtype=np.int16)
        for chunk_slice, out_slice, inner_slice in lattice:
            chunk_img = image[chunk_slice]
            sub_params = params.copy()
            sub_params["max_voxels_per_node_energy"] = chunk_img.size + 1
            sub_params["return_all_scales"] = return_all_scales
            chunk_data = calculate_energy_field(chunk_img, sub_params, get_chunking_lattice_func)
            if return_all_scales:
                energy_4d[(*out_slice, slice(None))] = chunk_data["energy_4d"][
                    (*inner_slice, slice(None))
                ]
            else:
                energy_3d[out_slice] = chunk_data["energy"][inner_slice]
                scale_indices[out_slice] = chunk_data["scale_indices"][inner_slice]
        if return_all_scales:
            if energy_sign < 0:
                energy_3d = np.min(energy_4d, axis=3)
                scale_indices = np.argmin(energy_4d, axis=3).astype(np.int16)
            else:
                energy_3d = np.max(energy_4d, axis=3)
                scale_indices = np.argmax(energy_4d, axis=3).astype(np.int16)
            return {
                "energy": energy_3d,
                "scale_indices": scale_indices,
                "lumen_radius_microns": lumen_radius_microns,
                "lumen_radius_pixels": lumen_radius_pixels,
                "lumen_radius_pixels_axes": lumen_radius_pixels_axes,
                "pixels_per_sigma_PSF": pixels_per_sigma_PSF,
                "microns_per_sigma_PSF": microns_per_sigma_PSF,
                "energy_sign": energy_sign,
                "energy_4d": energy_4d,
                "image_shape": image.shape,
            }
        return {
            "energy": energy_3d,
            "scale_indices": scale_indices,
            "lumen_radius_microns": lumen_radius_microns,
            "lumen_radius_pixels": lumen_radius_pixels,
            "lumen_radius_pixels_axes": lumen_radius_pixels_axes,
            "pixels_per_sigma_PSF": pixels_per_sigma_PSF,
            "microns_per_sigma_PSF": microns_per_sigma_PSF,
            "energy_sign": energy_sign,
            "image_shape": image.shape,
        }

    if energy_method == "sato" and sato is None:
        logger.warning(
            "Sato filter unavailable (requires scikit-image>=0.19). Falling back to Hessian."
        )
        energy_method = "hessian"

    if energy_method == "frangi" and frangi is None:
        logger.warning("Frangi filter unavailable. Falling back to Hessian.")
        energy_method = "hessian"

    if energy_method in ("frangi", "sato"):
        if return_all_scales:
            energy_4d = np.zeros((*image.shape, n_scales), dtype=np.float32)
        if energy_sign < 0:
            energy_3d = np.full(image.shape, np.inf, dtype=np.float32)
        else:
            energy_3d = np.full(image.shape, -np.inf, dtype=np.float32)
        scale_indices = np.zeros(image.shape, dtype=np.int16)

        for scale_idx, sigma in enumerate(lumen_radius_pixels):
            if energy_method == "frangi":
                vesselness = frangi(
                    image,
                    sigmas=[sigma],
                    black_ridges=(energy_sign > 0),
                )
            else:
                vesselness = sato(
                    image,
                    sigmas=[sigma],
                    black_ridges=(energy_sign > 0),
                )
            energy_scale = energy_sign * vesselness.astype(np.float32)
            if return_all_scales:
                energy_4d[..., scale_idx] = energy_scale
            mask = energy_scale < energy_3d if energy_sign < 0 else energy_scale > energy_3d
            energy_3d[mask] = energy_scale[mask]
            scale_indices[mask] = scale_idx
    else:
        # Multi-scale energy calculation with per-scale PSF weighting
        if return_all_scales:
            energy_4d = np.zeros((*image.shape, n_scales), dtype=np.float32)
        if energy_sign < 0:
            energy_3d = np.full(image.shape, np.inf, dtype=np.float32)
        else:
            energy_3d = np.full(image.shape, -np.inf, dtype=np.float32)
        scale_indices = np.zeros(image.shape, dtype=np.int16)

        for scale_idx, _ in enumerate(lumen_radius_pixels):
            # Calculate Gaussian sigmas at this scale using physical voxel spacing
            radius_microns = lumen_radius_microns[scale_idx]
            sigma_scale = (radius_microns / voxel_size) / max(gaussian_to_ideal_ratio, 1e-12)
            sigma_scale = np.asarray(sigma_scale, dtype=float)

            if approximating_PSF:
                sigma_object = np.sqrt(sigma_scale**2 + pixels_per_sigma_PSF**2)
            else:
                sigma_object = sigma_scale

            if spherical_to_annular_ratio < 1.0:
                annular_scale = sigma_scale * 1.5
                if approximating_PSF:
                    sigma_background = np.sqrt(annular_scale**2 + pixels_per_sigma_PSF**2)
                else:
                    sigma_background = annular_scale
            else:
                sigma_background = None

            energy_scale = _matlab_hessian_energy(
                image,
                sigma_object,
                sigma_background,
                spherical_to_annular_ratio,
                voxel_size,
                energy_sign,
            )
            if return_all_scales:
                energy_4d[:, :, :, scale_idx] = energy_scale
            if energy_sign < 0:
                mask = energy_scale < energy_3d
            else:
                mask = energy_scale > energy_3d
            energy_3d[mask] = energy_scale[mask]
            scale_indices[mask] = scale_idx
            continue

            # Spherical (Gaussian) component
            smoothed_object = gaussian_filter(image, sigma=tuple(sigma_object))

            # Annular (Difference of Gaussians) component
            # When spherical_to_annular_ratio=1: 100% Gaussian, 0% DoG
            # When spherical_to_annular_ratio=0: 0% Gaussian, 100% DoG
            if spherical_to_annular_ratio < 1.0:
                # Need to compute DoG component
                # Use a larger sigma for the background (annular means larger scale)
                annular_scale = sigma_scale * 1.5  # MATLAB uses a factor > 1 for annular
                if approximating_PSF:
                    sigma_background = np.sqrt(annular_scale**2 + pixels_per_sigma_PSF**2)
                else:
                    sigma_background = annular_scale
                smoothed_background = gaussian_filter(image, sigma=tuple(sigma_background))
                dog = smoothed_object - smoothed_background

                # Linear combination: (1-ratio)*DoG + ratio*Gaussian
                smoothed = (
                    1.0 - spherical_to_annular_ratio
                ) * dog + spherical_to_annular_ratio * smoothed_object
            else:
                # Pure Gaussian (no DoG component)
                smoothed = smoothed_object

            # Calculate Hessian eigenvalues with PSF-weighted sigma
            hessian = feature.hessian_matrix(smoothed, sigma=tuple(sigma_object))

            # Memory-efficient eigenvalue computation: compute directly without
            # storing all three eigenvalue arrays at full precision
            # Extract Hessian elements (6 components for symmetric 3x3)
            Hxx, Hxy, Hxz, Hyy, Hyz, Hzz = hessian

            # Compute eigenvalues in batches by z-slice to reduce peak memory
            shape_3d = smoothed.shape
            lambda1 = np.empty(shape_3d, dtype=np.float32)
            lambda2 = np.empty(shape_3d, dtype=np.float32)
            lambda3 = np.empty(shape_3d, dtype=np.float32)

            for z_idx in range(shape_3d[0]):
                # Build 3x3 Hessian matrices for this z-slice
                H = np.array(
                    [
                        [Hxx[z_idx], Hxy[z_idx], Hxz[z_idx]],
                        [Hxy[z_idx], Hyy[z_idx], Hyz[z_idx]],
                        [Hxz[z_idx], Hyz[z_idx], Hzz[z_idx]],
                    ]
                )  # Shape: (3, 3, Y, X)
                # Transpose to (Y, X, 3, 3) for eigvalsh
                H = np.moveaxis(H, [0, 1], [-2, -1])
                # Compute eigenvalues for this slice
                eigs = np.linalg.eigvalsh(H)  # Shape: (Y, X, 3), sorted ascending
                # Store in descending order (largest first) like skimage
                lambda1[z_idx] = eigs[..., 2]
                lambda2[z_idx] = eigs[..., 1]
                lambda3[z_idx] = eigs[..., 0]

            # Free Hessian memory
            del Hxx, Hxy, Hxz, Hyy, Hyz, Hzz, hessian

            # Frangi-like vesselness measure
            vesselness = np.zeros_like(lambda1)

            # Diagnostic logging for eigenvalue distributions
            if scale_idx == 0:  # Log only for first scale to avoid spam
                logger.debug(f"Scale {scale_idx}: radius={radius_microns:.2f}µm")
                logger.debug(f"  lambda1 range: [{lambda1.min():.6f}, {lambda1.max():.6f}]")
                logger.debug(f"  lambda2 range: [{lambda2.min():.6f}, {lambda2.max():.6f}]")
                logger.debug(f"  lambda3 range: [{lambda3.min():.6f}, {lambda3.max():.6f}]")
                logger.debug(f"  lambda2 < 0: {(lambda2 < 0).sum():,} voxels")
                logger.debug(f"  lambda3 < 0: {(lambda3 < 0).sum():,} voxels")
                logger.debug(f"  Both < 0: {((lambda2 < 0) & (lambda3 < 0)).sum():,} voxels")

            # For bright vessels (energy_sign < 0): we want tubular structures with
            # at least one negative eigenvalue in the cross-section (lambda2 or lambda3)
            # The original condition was too strict requiring BOTH to be negative
            if energy_sign < 0:
                mask = (lambda2 < 0) | (lambda3 < 0)  # At least one negative (more permissive)
            else:
                mask = (lambda2 > 0) | (lambda3 > 0)  # At least one positive for dark vessels

            if np.any(mask):
                # Ratios for tubular structure detection
                Ra = np.abs(lambda2[mask]) / (np.abs(lambda3[mask]) + 1e-12)
                Rb = np.abs(lambda1[mask]) / (
                    np.sqrt(np.abs(lambda2[mask] * lambda3[mask])) + 1e-12
                )
                S = np.sqrt(lambda1[mask] ** 2 + lambda2[mask] ** 2 + lambda3[mask] ** 2)

                # Vesselness response (Frangi-inspired)
                alpha, beta, c = 0.5, 0.5, np.max(S) + 1e-12
                vesselness[mask] = (
                    (1.0 - np.exp(-(Ra**2) / (2 * (alpha**2))))
                    * np.exp(-(Rb**2) / (2 * (beta**2)))
                    * (1.0 - np.exp(-(S**2) / (2 * (c**2))))
                )

            energy_scale = energy_sign * vesselness
            if return_all_scales:
                energy_4d[:, :, :, scale_idx] = energy_scale
            if energy_sign < 0:
                mask = energy_scale < energy_3d
            else:
                mask = energy_scale > energy_3d
            energy_3d[mask] = energy_scale[mask]
            scale_indices[mask] = scale_idx

    result = {
        "energy": energy_3d,
        "scale_indices": scale_indices,
        "lumen_radius_microns": lumen_radius_microns,
        "lumen_radius_pixels": lumen_radius_pixels,
        "lumen_radius_pixels_axes": lumen_radius_pixels_axes,
        "pixels_per_sigma_PSF": pixels_per_sigma_PSF,
        "microns_per_sigma_PSF": microns_per_sigma_PSF,
        "energy_sign": energy_sign,
        "image_shape": image.shape,
    }
    if return_all_scales:
        result["energy_4d"] = energy_4d
    return result


def _prepare_energy_config(image: np.ndarray, params: dict[str, Any]) -> dict[str, Any]:
    """Pre-compute scale and PSF metadata for resumable energy evaluation."""
    microns_per_voxel = np.array(params.get("microns_per_voxel", [1.0, 1.0, 1.0]), dtype=float)
    radius_smallest = float(params.get("radius_of_smallest_vessel_in_microns", 1.5))
    radius_largest = float(params.get("radius_of_largest_vessel_in_microns", 50.0))
    scales_per_octave = float(params.get("scales_per_octave", 1.5))
    gaussian_to_ideal_ratio = float(params.get("gaussian_to_ideal_ratio", 1.0))
    spherical_to_annular_ratio = float(params.get("spherical_to_annular_ratio", 1.0))
    approximating_PSF = bool(params.get("approximating_PSF", True))
    energy_sign = float(params.get("energy_sign", -1.0))
    energy_method = params.get("energy_method", "hessian")
    return_all_scales = bool(params.get("return_all_scales", False))
    max_voxels = int(params.get("max_voxels_per_node_energy", 1e5))

    if approximating_PSF:
        numerical_aperture = params.get("numerical_aperture", 0.95)
        excitation_wavelength = params.get("excitation_wavelength_in_microns", 1.3)
        sample_index_of_refraction = params.get("sample_index_of_refraction", 1.33)
        if numerical_aperture <= 0.7:
            coefficient, exponent = 0.320, 1.0
        else:
            coefficient, exponent = 0.325, 0.91
        microns_per_sigma_PSF = np.array(
            [
                excitation_wavelength / (2**0.5) * coefficient / (numerical_aperture**exponent),
                excitation_wavelength / (2**0.5) * coefficient / (numerical_aperture**exponent),
                excitation_wavelength
                / (2**0.5)
                * 0.532
                / (
                    sample_index_of_refraction
                    - (sample_index_of_refraction**2 - numerical_aperture**2) ** 0.5
                ),
            ],
            dtype=float,
        )
    else:
        microns_per_sigma_PSF = np.zeros(3, dtype=float)

    pixels_per_sigma_PSF = microns_per_sigma_PSF / microns_per_voxel
    scale_ordinates, lumen_radius_microns = _matlab_lumen_radius_range(
        radius_smallest,
        radius_largest,
        scales_per_octave,
    )
    lumen_radius_pixels_axes = lumen_radius_microns[:, None] / microns_per_voxel[None, :]
    lumen_radius_pixels = lumen_radius_pixels_axes.mean(axis=1)

    max_sigma = (lumen_radius_microns[-1] / microns_per_voxel) / max(
        gaussian_to_ideal_ratio,
        1e-12,
    )
    if approximating_PSF:
        max_sigma = np.sqrt(max_sigma**2 + pixels_per_sigma_PSF**2)
    margin = int(np.ceil(np.max(max_sigma)))

    if energy_method == "sato" and sato is None:
        logger.warning(
            "Sato filter unavailable (requires scikit-image>=0.19). Falling back to Hessian."
        )
        energy_method = "hessian"
    if energy_method == "frangi" and frangi is None:
        logger.warning("Frangi filter unavailable. Falling back to Hessian.")
        energy_method = "hessian"

    return {
        "image_shape": tuple(image.shape),
        "image_dtype": str(image.dtype),
        "microns_per_voxel": microns_per_voxel,
        "gaussian_to_ideal_ratio": gaussian_to_ideal_ratio,
        "spherical_to_annular_ratio": spherical_to_annular_ratio,
        "approximating_PSF": approximating_PSF,
        "energy_sign": energy_sign,
        "energy_method": energy_method,
        "return_all_scales": return_all_scales,
        "max_voxels": max_voxels,
        "margin": margin,
        "scale_ordinates": scale_ordinates,
        "lumen_radius_microns": lumen_radius_microns,
        "lumen_radius_pixels": lumen_radius_pixels,
        "lumen_radius_pixels_axes": lumen_radius_pixels_axes,
        "pixels_per_sigma_PSF": pixels_per_sigma_PSF,
        "microns_per_sigma_PSF": microns_per_sigma_PSF,
    }


def _compute_energy_scale(image: np.ndarray, config: dict[str, Any], scale_idx: int) -> np.ndarray:
    """Compute a single-scale energy response for a chunk."""
    image = image.astype(np.float32, copy=False)
    energy_method = config["energy_method"]
    energy_sign = config["energy_sign"]
    sigma_scale = config["lumen_radius_microns"][scale_idx] / config["microns_per_voxel"]
    sigma_scale = sigma_scale / max(config["gaussian_to_ideal_ratio"], 1e-12)
    sigma_scale = np.asarray(sigma_scale, dtype=float)

    if config["approximating_PSF"]:
        sigma_object = np.sqrt(sigma_scale**2 + config["pixels_per_sigma_PSF"] ** 2)
    else:
        sigma_object = sigma_scale

    if energy_method in ("frangi", "sato"):
        sigma = float(config["lumen_radius_pixels"][scale_idx])
        if energy_method == "frangi":
            vesselness = frangi(image, sigmas=[sigma], black_ridges=(energy_sign > 0))
        else:
            vesselness = sato(image, sigmas=[sigma], black_ridges=(energy_sign > 0))
        return energy_sign * vesselness.astype(np.float32)

    if config["spherical_to_annular_ratio"] < 1.0:
        annular_scale = sigma_scale * 1.5
        if config["approximating_PSF"]:
            sigma_background = np.sqrt(annular_scale**2 + config["pixels_per_sigma_PSF"] ** 2)
        else:
            sigma_background = annular_scale
    else:
        sigma_background = None

    return _matlab_hessian_energy(
        image,
        sigma_object,
        sigma_background,
        config["spherical_to_annular_ratio"],
        config["microns_per_voxel"],
        energy_sign,
    )

    smoothed_object = gaussian_filter(image, sigma=tuple(sigma_object))
    if config["spherical_to_annular_ratio"] < 1.0:
        annular_scale = sigma_scale * 1.5
        if config["approximating_PSF"]:
            sigma_background = np.sqrt(annular_scale**2 + config["pixels_per_sigma_PSF"] ** 2)
        else:
            sigma_background = annular_scale
        smoothed_background = gaussian_filter(image, sigma=tuple(sigma_background))
        dog = smoothed_object - smoothed_background
        smoothed = (1.0 - config["spherical_to_annular_ratio"]) * dog + config[
            "spherical_to_annular_ratio"
        ] * smoothed_object
    else:
        smoothed = smoothed_object

    hessian = feature.hessian_matrix(smoothed, sigma=tuple(sigma_object))
    Hxx, Hxy, Hxz, Hyy, Hyz, Hzz = hessian
    shape_3d = smoothed.shape
    lambda1 = np.empty(shape_3d, dtype=np.float32)
    lambda2 = np.empty(shape_3d, dtype=np.float32)
    lambda3 = np.empty(shape_3d, dtype=np.float32)
    for z_idx in range(shape_3d[0]):
        H = np.array(
            [
                [Hxx[z_idx], Hxy[z_idx], Hxz[z_idx]],
                [Hxy[z_idx], Hyy[z_idx], Hyz[z_idx]],
                [Hxz[z_idx], Hyz[z_idx], Hzz[z_idx]],
            ]
        )
        H = np.moveaxis(H, [0, 1], [-2, -1])
        eigs = np.linalg.eigvalsh(H)
        lambda1[z_idx] = eigs[..., 2]
        lambda2[z_idx] = eigs[..., 1]
        lambda3[z_idx] = eigs[..., 0]

    vesselness = np.zeros_like(lambda1)
    if energy_sign < 0:
        mask = (lambda2 < 0) | (lambda3 < 0)
    else:
        mask = (lambda2 > 0) | (lambda3 > 0)
    if np.any(mask):
        Ra = np.abs(lambda2[mask]) / (np.abs(lambda3[mask]) + 1e-12)
        Rb = np.abs(lambda1[mask]) / (np.sqrt(np.abs(lambda2[mask] * lambda3[mask])) + 1e-12)
        S = np.sqrt(lambda1[mask] ** 2 + lambda2[mask] ** 2 + lambda3[mask] ** 2)
        alpha, beta, c = 0.5, 0.5, np.max(S) + 1e-12
        vesselness[mask] = (
            (1.0 - np.exp(-(Ra**2) / (2 * (alpha**2))))
            * np.exp(-(Rb**2) / (2 * (beta**2)))
            * (1.0 - np.exp(-(S**2) / (2 * (c**2))))
        )
    return (energy_sign * vesselness).astype(np.float32)


def calculate_energy_field_resumable(
    image: np.ndarray,
    params: dict[str, Any],
    stage_controller: StageController,
    get_chunking_lattice_func=None,
) -> dict[str, Any]:
    """Compute energy with resumable chunk/scale units backed by memmaps."""
    config = _prepare_energy_config(image, params)
    config_hash = hashlib.sha256(
        json.dumps(
            {
                "params": {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in config.items()
                    if k not in {"image_shape", "image_dtype"}
                },
                "shape": list(config["image_shape"]),
            },
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()

    total_voxels = int(np.prod(image.shape))
    max_voxels = int(config["max_voxels"])
    if total_voxels > max_voxels:
        lattice = get_chunking_lattice_func(image.shape, max_voxels, int(config["margin"]))
    else:
        lattice = [
            (
                (slice(0, image.shape[0]), slice(0, image.shape[1]), slice(0, image.shape[2])),
                (slice(0, image.shape[0]), slice(0, image.shape[1]), slice(0, image.shape[2])),
                (slice(0, image.shape[0]), slice(0, image.shape[1]), slice(0, image.shape[2])),
            )
        ]

    energy_path = stage_controller.artifact_path("best_energy.npy")
    scale_path = stage_controller.artifact_path("best_scale.npy")
    energy4d_path = stage_controller.artifact_path("energy_4d.npy")
    state = stage_controller.load_state()
    completed_units = set(state.get("completed_units", []))
    if state.get("config_hash") not in (None, config_hash):
        completed_units = set()
        for stale_path in (energy_path, scale_path, energy4d_path):
            if stale_path.exists():
                stale_path.unlink()

    n_scales = len(config["lumen_radius_microns"])
    total_units = len(lattice) * n_scales
    resumed = bool(completed_units)
    stage_controller.begin(
        detail="Computing resumable energy field",
        units_total=total_units,
        units_completed=len(completed_units),
        substage="scale_chunks",
        resumed=resumed,
    )

    if energy_path.exists():
        best_energy = np.lib.format.open_memmap(energy_path, mode="r+")
    else:
        best_energy = np.lib.format.open_memmap(
            energy_path,
            mode="w+",
            dtype=np.float32,
            shape=image.shape,
        )
        fill_value = np.inf if config["energy_sign"] < 0 else -np.inf
        best_energy[...] = fill_value

    if scale_path.exists():
        best_scale = np.lib.format.open_memmap(scale_path, mode="r+")
    else:
        best_scale = np.lib.format.open_memmap(
            scale_path,
            mode="w+",
            dtype=np.int16,
            shape=image.shape,
        )
        best_scale[...] = -1

    energy_4d = None
    if config["return_all_scales"]:
        if energy4d_path.exists():
            energy_4d = np.lib.format.open_memmap(energy4d_path, mode="r+")
        else:
            energy_4d = np.lib.format.open_memmap(
                energy4d_path,
                mode="w+",
                dtype=np.float32,
                shape=(*image.shape, n_scales),
            )
            energy_4d[...] = 0.0

    for chunk_idx, (chunk_slice, out_slice, inner_slice) in enumerate(lattice):
        chunk_img = image[chunk_slice]
        for scale_idx in range(n_scales):
            unit_id = f"{chunk_idx}:{scale_idx}"
            if unit_id in completed_units:
                continue

            energy_scale = _compute_energy_scale(chunk_img, config, scale_idx)
            chunk_inner = energy_scale[inner_slice]
            target_view = best_energy[out_slice]
            if config["energy_sign"] < 0:
                mask = chunk_inner < target_view
            else:
                mask = chunk_inner > target_view
            target_view[mask] = chunk_inner[mask]
            best_energy[out_slice] = target_view
            scale_view = best_scale[out_slice]
            scale_view[mask] = scale_idx
            best_scale[out_slice] = scale_view
            if energy_4d is not None:
                energy_4d[(*out_slice, scale_idx)] = chunk_inner

            completed_units.add(unit_id)
            state = {
                "config_hash": config_hash,
                "completed_units": sorted(completed_units),
                "total_units": total_units,
                "n_chunks": len(lattice),
                "n_scales": n_scales,
            }
            stage_controller.save_state(state)
            stage_controller.update(
                units_total=total_units,
                units_completed=len(completed_units),
                detail=f"Energy chunk {chunk_idx + 1}/{len(lattice)}, scale {scale_idx + 1}/{n_scales}",
                substage="scale_chunks",
                resumed=resumed,
            )

    result = {
        "energy": np.asarray(best_energy),
        "scale_indices": np.asarray(best_scale),
        "lumen_radius_microns": config["lumen_radius_microns"],
        "lumen_radius_pixels": config["lumen_radius_pixels"],
        "lumen_radius_pixels_axes": config["lumen_radius_pixels_axes"],
        "pixels_per_sigma_PSF": config["pixels_per_sigma_PSF"],
        "microns_per_sigma_PSF": config["microns_per_sigma_PSF"],
        "energy_sign": config["energy_sign"],
        "image_shape": image.shape,
    }
    if energy_4d is not None:
        result["energy_4d"] = np.asarray(energy_4d)
    return result


def compute_gradient_fast(energy, p0, p1, p2, inv_mpv_2x):
    """
    Optimized gradient computation avoiding array allocations for position.
    inv_mpv_2x should be 1.0 / (2.0 * microns_per_voxel)
    """
    s0, s1, s2 = energy.shape

    if s0 < 3 or s1 < 3 or s2 < 3:
        return np.zeros(3, dtype=float)

    # Manual clamping to [1, shape-2] to prevent out-of-bounds access
    if p0 < 1:
        p0 = 1
    elif p0 > s0 - 2:
        p0 = s0 - 2

    if p1 < 1:
        p1 = 1
    elif p1 > s1 - 2:
        p1 = s1 - 2

    if p2 < 1:
        p2 = 1
    elif p2 > s2 - 2:
        p2 = s2 - 2

    # We still allocate grad, but we avoid pos_int allocation and unpacking
    grad = np.empty(3, dtype=float)
    grad[0] = (energy[p0 + 1, p1, p2] - energy[p0 - 1, p1, p2]) * inv_mpv_2x[0]
    grad[1] = (energy[p0, p1 + 1, p2] - energy[p0, p1 - 1, p2]) * inv_mpv_2x[1]
    grad[2] = (energy[p0, p1, p2 + 1] - energy[p0, p1, p2 - 1]) * inv_mpv_2x[2]

    return grad


__all__ = [
    "calculate_energy_field",
    "compute_gradient_fast",
    "compute_gradient_impl",
    "spherical_structuring_element",
]
