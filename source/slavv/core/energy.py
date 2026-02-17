
"""
Energy field calculations for SLAVV.
Includes Hessian-based vessel enhancement (Frangi/Sato) and Numba-accelerated gradient computation.
"""
import logging
from typing import Dict, Any, Tuple, Optional

import numpy as np
import scipy.ndimage as ndi
from scipy.ndimage import gaussian_filter
from skimage import feature
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
        
        # Manual clamping to [1, shape-2]
        pos_y = int(pos_int[0])
        if pos_y < 1: pos_y = 1
        elif pos_y > dim_y - 2: pos_y = dim_y - 2
        
        pos_x = int(pos_int[1])
        if pos_x < 1: pos_x = 1
        elif pos_x > dim_x - 2: pos_x = dim_x - 2
        
        pos_z = int(pos_int[2])
        if pos_z < 1: pos_z = 1
        elif pos_z > dim_z - 2: pos_z = dim_z - 2
        
        grad = np.zeros(3, dtype=np.float64)
        # Use explicit indexing for speed and clarity
        grad[0] = (energy[pos_y+1, pos_x, pos_z] - energy[pos_y-1, pos_x, pos_z]) / (2.0 * mpv[0])
        grad[1] = (energy[pos_y, pos_x+1, pos_z] - energy[pos_y, pos_x-1, pos_z]) / (2.0 * mpv[1])
        grad[2] = (energy[pos_y, pos_x, pos_z+1] - energy[pos_y, pos_x, pos_z-1]) / (2.0 * mpv[2])
        
        return grad

else:

    def compute_gradient_impl(energy, pos_int, microns_per_voxel):
        """Compute local energy gradient via central differences."""
        grad = np.zeros(3, dtype=float)

        # Unpack position and shape
        pos_y, pos_x, pos_z = pos_int
        shape_y, shape_x, shape_z = energy.shape

        # Manual clamping to [1, shape-2] to prevent out-of-bounds access
        if pos_y < 1: pos_y = 1
        elif pos_y > shape_y - 2: pos_y = shape_y - 2

        if pos_x < 1: pos_x = 1
        elif pos_x > shape_x - 2: pos_x = shape_x - 2

        if pos_z < 1: pos_z = 1
        elif pos_z > shape_z - 2: pos_z = shape_z - 2

        # Direct indexing for speed (avoids allocations and tuple overhead)
        grad[0] = (energy[pos_y+1, pos_x, pos_z] - energy[pos_y-1, pos_x, pos_z]) / (2.0 * microns_per_voxel[0])
        grad[1] = (energy[pos_y, pos_x+1, pos_z] - energy[pos_y, pos_x-1, pos_z]) / (2.0 * microns_per_voxel[1])
        grad[2] = (energy[pos_y, pos_x, pos_z+1] - energy[pos_y, pos_x, pos_z-1]) / (2.0 * microns_per_voxel[2])

        return grad

# --- Helper Functions ---

from ..utils import get_chunking_lattice

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


def solve_symmetric_eigenvalues_3x3(
    Hxx: np.ndarray, Hxy: np.ndarray, Hxz: np.ndarray,
    Hyy: np.ndarray, Hyz: np.ndarray, Hzz: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Analytically compute sorted eigenvalues for 3x3 symmetric matrices.

    Returns eigenvalues lambda1 >= lambda2 >= lambda3 (descending order).
    This function avoids constructing full 3x3 matrices and calling generic eigvalsh,
    providing significant speedup (10x+) for large volumes.

    Based on the trigonometric solution for the depressed cubic equation.
    """
    # Coefficients of characteristic polynomial: lambda^3 + c2*lambda^2 + c1*lambda + c0 = 0
    # x^3 - tr(A)*x^2 + (sum of 2x2 principal minors)*x - det(A) = 0
    c2 = -(Hxx + Hyy + Hzz)
    c1 = (Hxx*Hyy - Hxy*Hxy) + (Hxx*Hzz - Hxz*Hxz) + (Hyy*Hzz - Hyz*Hyz)
    c0 = -(Hxx*(Hyy*Hzz - Hyz*Hyz) - Hxy*(Hxy*Hzz - Hxz*Hyz) + Hxz*(Hxy*Hyz - Hxz*Hyy))

    # Depressed cubic t^3 + p*t + q = 0
    p = c1 - c2*c2/3.0
    q = 2.0*c2*c2*c2/27.0 - c1*c2/3.0 + c0

    # R = sqrt(-p/3)
    # Using np.maximum to ensure non-negative argument for sqrt
    R = np.sqrt(np.maximum(-p / 3.0, 0.0))

    # Avoid division by zero when R is 0 (triple root case)
    R_safe = np.where(R == 0, 1.0, R)
    val = -q / (2.0 * R_safe**3)

    # Clip for acos domain [-1, 1]
    val = np.clip(val, -1.0, 1.0)

    phi = np.arccos(val) / 3.0

    # Roots in t-space
    t1 = 2.0 * R * np.cos(phi)
    t2 = 2.0 * R * np.cos(phi + 2.0*np.pi/3.0)
    t3 = 2.0 * R * np.cos(phi + 4.0*np.pi/3.0)

    # Shift back to lambda-space
    offset = -c2 / 3.0
    l1 = t1 + offset
    l2 = t2 + offset
    l3 = t3 + offset

    # Sort eigenvalues (descending order l1 >= l2 >= l3)
    # Stacking is necessary for sorting, but we can do it efficiently
    # We want l1 to be largest, l3 smallest.
    # Current l1, l2, l3 are just roots, order depends on phi.

    # Stack along new last axis
    eigs = np.stack([l1, l2, l3], axis=-1)
    eigs.sort(axis=-1)

    # eigs is sorted ascending: [smallest, middle, largest]
    # We want descending: lambda1=largest, lambda2=middle, lambda3=smallest
    return eigs[..., 2], eigs[..., 1], eigs[..., 0]


def calculate_energy_field(image: np.ndarray, params: Dict[str, Any], get_chunking_lattice_func=None) -> Dict[str, Any]:
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
    microns_per_voxel = params.get('microns_per_voxel', [1.0, 1.0, 1.0])
    radius_smallest = params.get('radius_of_smallest_vessel_in_microns', 1.5)
    radius_largest = params.get('radius_of_largest_vessel_in_microns', 50.0)
    scales_per_octave = params.get('scales_per_octave', 1.5)
    gaussian_to_ideal_ratio = params.get('gaussian_to_ideal_ratio', 1.0)
    spherical_to_annular_ratio = params.get('spherical_to_annular_ratio', 1.0)
    approximating_PSF = params.get('approximating_PSF', True)
    energy_sign = params.get('energy_sign', -1.0)  # -1 for bright vessels
    return_all_scales = params.get('return_all_scales', False)
    energy_method = params.get('energy_method', 'hessian')

    # Cache voxel spacing for anisotropy handling
    voxel_size = np.array(microns_per_voxel, dtype=float)
    
    # PSF calculation (from MATLAB implementation)
    if approximating_PSF:
        numerical_aperture = params.get('numerical_aperture', 0.95)
        excitation_wavelength = params.get('excitation_wavelength_in_microns', 1.3)
        sample_index_of_refraction = params.get('sample_index_of_refraction', 1.33)
        
        # PSF calculation based on Zipfel et al.
        if numerical_aperture <= 0.7:
            coefficient, exponent = 0.320, 1.0
        else:
            coefficient, exponent = 0.325, 0.91
            
        microns_per_sigma_PSF = [
            excitation_wavelength / (2**0.5) * coefficient / (numerical_aperture**exponent),
            excitation_wavelength / (2**0.5) * coefficient / (numerical_aperture**exponent),
            excitation_wavelength / (2**0.5) * 0.532 / (
                sample_index_of_refraction - 
                (sample_index_of_refraction**2 - numerical_aperture**2)**0.5
            )
        ]
    else:
        microns_per_sigma_PSF = [0.0, 0.0, 0.0]
        
    microns_per_sigma_PSF = np.array(microns_per_sigma_PSF, dtype=float)
    pixels_per_sigma_PSF = microns_per_sigma_PSF / voxel_size
    
    # Calculate scale range following MATLAB ordination
    largest_per_smallest_ratio = radius_largest / radius_smallest
    final_scale = int(
        np.floor(np.log2(largest_per_smallest_ratio) * scales_per_octave)
    )
    scale_ordinates = np.arange(0, final_scale + 1)
    scale_factors = 2 ** (scale_ordinates / scales_per_octave)
    lumen_radius_microns = radius_smallest * scale_factors

    # Convert radii to pixels per axis then average for scalar pixel radii
    lumen_radius_pixels_axes = lumen_radius_microns[:, None] / voxel_size[None, :]
    lumen_radius_pixels = lumen_radius_pixels_axes.mean(axis=1)

    # Use float32 image once for all scales
    image = image.astype(np.float32)

    total_voxels = int(np.prod(image.shape))
    max_voxels = int(params.get("max_voxels_per_node_energy", 1e5))
    if total_voxels > max_voxels:
        max_sigma = (
            (lumen_radius_microns[-1] / voxel_size)
            / max(gaussian_to_ideal_ratio, 1e-12)
        )
        if approximating_PSF:
            max_sigma = np.sqrt(max_sigma**2 + pixels_per_sigma_PSF**2)
        margin = int(np.ceil(np.max(max_sigma)))
        lattice = get_chunking_lattice_func(image.shape, max_voxels, margin)
        if return_all_scales:
            energy_4d = np.zeros((*image.shape, len(scale_factors)), dtype=np.float32)
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
                energy_4d[out_slice + (slice(None),)] = chunk_data["energy_4d"][
                    inner_slice + (slice(None),)
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

    if energy_method == 'sato' and sato is None:
        logger.warning("Sato filter unavailable (requires scikit-image>=0.19). Falling back to Hessian.")
        energy_method = 'hessian'

    if energy_method == 'frangi' and frangi is None:
        logger.warning("Frangi filter unavailable. Falling back to Hessian.")
        energy_method = 'hessian'

    if energy_method in ('frangi', 'sato'):
        if return_all_scales:
            energy_4d = np.zeros((*image.shape, len(scale_factors)), dtype=np.float32)
        if energy_sign < 0:
            energy_3d = np.full(image.shape, np.inf, dtype=np.float32)
        else:
            energy_3d = np.full(image.shape, -np.inf, dtype=np.float32)
        scale_indices = np.zeros(image.shape, dtype=np.int16)

        for scale_idx, sigma in enumerate(lumen_radius_pixels):
            if energy_method == 'frangi':
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
            energy_4d = np.zeros((*image.shape, len(scale_factors)), dtype=np.float32)
        if energy_sign < 0:
            energy_3d = np.full(image.shape, np.inf, dtype=np.float32)
        else:
            energy_3d = np.full(image.shape, -np.inf, dtype=np.float32)
        scale_indices = np.zeros(image.shape, dtype=np.int16)

        for scale_idx, _ in enumerate(lumen_radius_pixels):
            # Calculate Gaussian sigmas at this scale using physical voxel spacing
            radius_microns = lumen_radius_microns[scale_idx]
            sigma_scale = (
                (radius_microns / voxel_size) / max(gaussian_to_ideal_ratio, 1e-12)
            )
            sigma_scale = np.asarray(sigma_scale, dtype=float)

            if approximating_PSF:
                sigma_object = np.sqrt(sigma_scale**2 + pixels_per_sigma_PSF**2)
            else:
                sigma_object = sigma_scale

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
                    sigma_background = np.sqrt(
                        annular_scale**2 + pixels_per_sigma_PSF**2
                    )
                else:
                    sigma_background = annular_scale
                smoothed_background = gaussian_filter(
                    image, sigma=tuple(sigma_background)
                )
                dog = smoothed_object - smoothed_background
                
                # Linear combination: (1-ratio)*DoG + ratio*Gaussian
                smoothed = (1.0 - spherical_to_annular_ratio) * dog + spherical_to_annular_ratio * smoothed_object
            else:
                # Pure Gaussian (no DoG component)
                smoothed = smoothed_object

            # Calculate Hessian eigenvalues with PSF-weighted sigma
            # Note: use_gaussian_derivatives=False removed for compatibility with older skimage
            hessian = feature.hessian_matrix(
                smoothed, sigma=tuple(sigma_object)
            )
            
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
                # Use analytical solver instead of constructing matrices and calling eigvalsh
                # This avoids significant overhead from array construction and generalized solver
                l1, l2, l3 = solve_symmetric_eigenvalues_3x3(
                    Hxx[z_idx], Hxy[z_idx], Hxz[z_idx],
                    Hyy[z_idx], Hyz[z_idx], Hzz[z_idx]
                )

                lambda1[z_idx] = l1
                lambda2[z_idx] = l2
                lambda3[z_idx] = l3
            
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
                Rb = np.abs(lambda1[mask]) / (np.sqrt(np.abs(lambda2[mask] * lambda3[mask])) + 1e-12)
                S = np.sqrt(lambda1[mask]**2 + lambda2[mask]**2 + lambda3[mask]**2)

                # Vesselness response (Frangi-inspired)
                alpha, beta, c = 0.5, 0.5, np.max(S) + 1e-12
                vesselness[mask] = (
                    (1.0 - np.exp(-(Ra**2) / (2 * (alpha**2)))) *
                    np.exp(-(Rb**2) / (2 * (beta**2))) *
                    (1.0 - np.exp(-(S**2) / (2 * (c**2))))
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
        'energy': energy_3d,
        'scale_indices': scale_indices,
        'lumen_radius_microns': lumen_radius_microns,
        "lumen_radius_pixels": lumen_radius_pixels,
        'lumen_radius_pixels_axes': lumen_radius_pixels_axes,
        'pixels_per_sigma_PSF': pixels_per_sigma_PSF,
        'microns_per_sigma_PSF': microns_per_sigma_PSF,
        'energy_sign': energy_sign,
        'image_shape': image.shape
    }
    if return_all_scales:
        result['energy_4d'] = energy_4d
    return result

def compute_gradient_fast(energy, p0, p1, p2, inv_mpv_2x):
    """
    Optimized gradient computation avoiding array allocations for position.
    inv_mpv_2x should be 1.0 / (2.0 * microns_per_voxel)
    """
    s0, s1, s2 = energy.shape

    # Manual clamping to [1, shape-2] to prevent out-of-bounds access
    if p0 < 1: p0 = 1
    elif p0 > s0 - 2: p0 = s0 - 2

    if p1 < 1: p1 = 1
    elif p1 > s1 - 2: p1 = s1 - 2

    if p2 < 1: p2 = 1
    elif p2 > s2 - 2: p2 = s2 - 2

    # We still allocate grad, but we avoid pos_int allocation and unpacking
    grad = np.empty(3, dtype=float)
    grad[0] = (energy[p0+1, p1, p2] - energy[p0-1, p1, p2]) * inv_mpv_2x[0]
    grad[1] = (energy[p0, p1+1, p2] - energy[p0, p1-1, p2]) * inv_mpv_2x[1]
    grad[2] = (energy[p0, p1, p2+1] - energy[p0, p1, p2-1]) * inv_mpv_2x[2]

    return grad


__all__ = [
    "calculate_energy_field",
    "spherical_structuring_element",
    "compute_gradient_impl",
    "compute_gradient_fast",
]
