"""Energy configuration helpers."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from slavv_python.core.energy import backends as backends
from slavv_python.core.energy.hessian_response import matlab_octave_resolution_factors

logger = logging.getLogger(__name__)


def _matlab_lumen_radius_range(
    radius_smallest: float, radius_largest: float, scales_per_octave: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return MATLAB-aligned scale ordinates and lumen radii."""
    largest_per_smallest_volume_ratio = (radius_largest / radius_smallest) ** 3
    final_scale = int(np.round(np.log2(largest_per_smallest_volume_ratio) * scales_per_octave))
    scale_ordinates: np.ndarray = np.arange(-1, final_scale + 2, dtype=float)
    scale_factors: np.ndarray = np.power(2.0, scale_ordinates / scales_per_octave / 3.0)
    return scale_ordinates, radius_smallest * scale_factors


def _prepare_energy_config(image: np.ndarray, params: dict[str, Any]) -> dict[str, Any]:
    """Pre-compute scale and PSF metadata for resumable energy evaluation."""
    microns_per_voxel = np.array(params.get("microns_per_voxel", [1.0, 1.0, 1.0]), dtype=float)
    radius_smallest = float(params.get("radius_of_smallest_vessel_in_microns", 1.5))
    radius_largest = float(params.get("radius_of_largest_vessel_in_microns", 50.0))
    scales_per_octave = float(params.get("scales_per_octave", 1.5))
    gaussian_to_ideal_ratio = float(params.get("gaussian_to_ideal_ratio", 1.0))
    spherical_to_annular_ratio = float(params.get("spherical_to_annular_ratio", 1.0))
    approximating_psf = bool(params.get("approximating_PSF", True))
    energy_sign = float(params.get("energy_sign", -1.0))
    energy_method = params.get("energy_method", "hessian")
    energy_projection_mode = str(params.get("energy_projection_mode", "matlab")).strip().lower()
    return_all_scales = bool(params.get("return_all_scales", False))
    max_voxels = int(params.get("max_voxels_per_node_energy", 1e5))
    if energy_method == "simpleitk_objectness":
        backends._require_simpleitk_backend()
        backends._warn_simpleitk_parameter_mismatches(params)
    if energy_method == "cupy_hessian":
        backends._require_cupy_backend()
        backends._warn_cupy_parameter_mismatches(params)

    if approximating_psf:
        numerical_aperture = params.get("numerical_aperture", 0.95)
        excitation_wavelength = params.get("excitation_wavelength_in_microns", 1.3)
        sample_index_of_refraction = params.get("sample_index_of_refraction", 1.33)
        if numerical_aperture <= 0.7:
            coefficient, exponent = 0.320, 1.0
        else:
            coefficient, exponent = 0.325, 0.91
        microns_per_sigma_psf = np.array(
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
        microns_per_sigma_psf = np.zeros(3, dtype=float)

    pixels_per_sigma_psf = microns_per_sigma_psf / microns_per_voxel
    scale_ordinates, lumen_radius_microns = _matlab_lumen_radius_range(
        radius_smallest,
        radius_largest,
        scales_per_octave,
    )
    octave_at_scales, scale_resolution_factors = matlab_octave_resolution_factors(
        lumen_radius_microns,
        microns_per_voxel,
        scales_per_octave,
    )
    lumen_radius_pixels_axes = lumen_radius_microns[:, None] / microns_per_voxel[None, :]
    lumen_radius_pixels = lumen_radius_pixels_axes.mean(axis=1)

    largest_pixels_per_radius = lumen_radius_microns[-1] / microns_per_voxel
    if approximating_psf:
        chunk_support = np.sqrt(pixels_per_sigma_psf**2 + largest_pixels_per_radius**2)
    else:
        chunk_support = largest_pixels_per_radius
    margin = int(np.ceil(np.max(6.0 * chunk_support)))

    if energy_method == "sato" and backends.sato is None:
        logger.warning(
            "Sato filter unavailable (requires scikit-image>=0.19). Falling back to Hessian."
        )
        energy_method = "hessian"
    if energy_method == "frangi" and backends.frangi is None:
        logger.warning("Frangi filter unavailable. Falling back to Hessian.")
        energy_method = "hessian"

    return {
        "image_shape": tuple(image.shape),
        "image_dtype": str(image.dtype),
        "microns_per_voxel": microns_per_voxel,
        "energy_storage_format": str(params.get("energy_storage_format", "auto")).strip(),
        "gaussian_to_ideal_ratio": gaussian_to_ideal_ratio,
        "spherical_to_annular_ratio": spherical_to_annular_ratio,
        "approximating_PSF": approximating_psf,
        "energy_sign": energy_sign,
        "energy_method": energy_method,
        "energy_projection_mode": energy_projection_mode,
        "return_all_scales": return_all_scales,
        "max_voxels": max_voxels,
        "margin": margin,
        "scale_ordinates": scale_ordinates,
        "octave_at_scales": octave_at_scales,
        "scale_resolution_factors": scale_resolution_factors,
        "lumen_radius_microns": lumen_radius_microns,
        "lumen_radius_pixels": lumen_radius_pixels,
        "lumen_radius_pixels_axes": lumen_radius_pixels_axes,
        "pixels_per_sigma_PSF": pixels_per_sigma_psf,
        "microns_per_sigma_PSF": microns_per_sigma_psf,
        "n_jobs": int(params.get("n_jobs", 1)),
    }


__all__ = ["_prepare_energy_config"]
