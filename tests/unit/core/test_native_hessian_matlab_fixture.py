from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.testing as npt

from slavv_python.core import SLAVVProcessor
from slavv_python.core.energy import hessian_response as native_hessian
from slavv_python.core.energy.config import _prepare_energy_config


def test_native_hessian_matches_small_matlab_reference_fixture():
    fixture = np.load(
        Path(__file__).resolve().parents[2]
        / "fixtures"
        / "energy"
        / "native_hessian_matlab_fixture.npz"
    )
    image = fixture["image"].astype(np.float32)
    params = {
        "radius_of_smallest_vessel_in_microns": float(fixture["radius_smallest"][0]),
        "radius_of_largest_vessel_in_microns": float(fixture["radius_largest"][0]),
        "scales_per_octave": float(fixture["scales_per_octave"][0]),
        "approximating_PSF": False,
        "gaussian_to_ideal_ratio": 1.0,
        "spherical_to_annular_ratio": 1.0,
        "energy_projection_mode": "matlab",
        "return_all_scales": True,
    }

    result = SLAVVProcessor().calculate_energy_field(image, params)

    expected_energy = fixture["expected_energy"].astype(np.float32)
    expected_energy_4d = fixture["expected_energy_4d"].astype(np.float32)
    expected_scale_indices = fixture["expected_scale_indices"].astype(np.int16)
    finite_energy_mask = np.isfinite(expected_energy) & np.isfinite(result["energy"])

    npt.assert_allclose(
        result["lumen_radius_microns"],
        fixture["lumen_radius_microns"].astype(np.float32),
        rtol=1e-7,
        atol=1e-7,
    )
    npt.assert_allclose(
        result["energy"][finite_energy_mask],
        expected_energy[finite_energy_mask],
        rtol=1e-4,
        atol=1e-4,
    )
    npt.assert_array_equal(
        result["scale_indices"],
        expected_scale_indices,
    )
    for scale_idx in range(expected_energy_4d.shape[3]):
        finite_scale_mask = np.isfinite(expected_energy_4d[..., scale_idx]) & np.isfinite(
            result["energy_4d"][..., scale_idx]
        )
        npt.assert_allclose(
            result["energy_4d"][..., scale_idx][finite_scale_mask],
            expected_energy_4d[..., scale_idx][finite_scale_mask],
            rtol=1e-4,
            atol=1e-4,
        )


def test_native_hessian_matches_matlab_laplacian_and_valid_mask_per_scale():
    fixture = np.load(
        Path(__file__).resolve().parents[2]
        / "fixtures"
        / "energy"
        / "native_hessian_matlab_fixture.npz"
    )
    image = fixture["image"].astype(np.float32)
    params = {
        "radius_of_smallest_vessel_in_microns": float(fixture["radius_smallest"][0]),
        "radius_of_largest_vessel_in_microns": float(fixture["radius_largest"][0]),
        "scales_per_octave": float(fixture["scales_per_octave"][0]),
        "approximating_PSF": False,
        "gaussian_to_ideal_ratio": 1.0,
        "spherical_to_annular_ratio": 1.0,
        "energy_projection_mode": "matlab",
        "return_all_scales": True,
    }
    config = _prepare_energy_config(image, params)

    expected_laplacian_4d = fixture["expected_laplacian_4d"].astype(np.float32)
    expected_valid_mask_4d = fixture["expected_valid_mask_4d"].astype(bool)

    npt.assert_array_equal(config["scale_resolution_factors"], np.ones((4, 3), dtype=np.int16))

    for scale_idx in range(expected_laplacian_4d.shape[3]):
        debug_outputs = native_hessian._compute_native_hessian_scale_debug(image, config, scale_idx)
        npt.assert_allclose(
            debug_outputs["laplacian"],
            expected_laplacian_4d[..., scale_idx],
            rtol=1e-6,
            atol=1e-7,
        )
        npt.assert_array_equal(
            debug_outputs["valid_voxels"],
            expected_valid_mask_4d[..., scale_idx],
        )
