from __future__ import annotations

import numpy as np
import numpy.testing as npt

from slavv_python.core.energy_internal import hessian_response as native_hessian
from slavv_python.core.energy_internal.energy_config import _prepare_energy_config


def test_matlab_projection_uses_per_voxel_minimum():
    energy_4d = np.array(
        [[[[-1.0, -3.0, -2.0]]]],
        dtype=np.float32,
    )

    energy, scale_indices = native_hessian.project_energy_stack(
        energy_4d,
        energy_sign=-1.0,
        projection_mode="matlab",
        spherical_to_annular_ratio=1.0,
    )

    npt.assert_allclose(energy, np.array([[[-3.0]]], dtype=np.float32))
    npt.assert_array_equal(scale_indices, np.array([[[1]]], dtype=np.int16))


def test_paper_projection_blends_annular_and_spherical_scale_estimates():
    energy_4d = np.array(
        [[[[-20.0, -10.0, -19.0, -19.0]]]],
        dtype=np.float32,
    )

    matlab_energy, matlab_scale = native_hessian.project_energy_stack(
        energy_4d,
        energy_sign=-1.0,
        projection_mode="matlab",
        spherical_to_annular_ratio=0.5,
    )
    paper_energy, paper_scale = native_hessian.project_energy_stack(
        energy_4d,
        energy_sign=-1.0,
        projection_mode="paper",
        spherical_to_annular_ratio=0.5,
    )

    npt.assert_allclose(matlab_energy, np.array([[[-20.0]]], dtype=np.float32))
    npt.assert_array_equal(matlab_scale, np.array([[[0]]], dtype=np.int16))
    npt.assert_allclose(paper_energy, np.array([[[-10.0]]], dtype=np.float32))
    npt.assert_array_equal(paper_scale, np.array([[[1]]], dtype=np.int16))


def test_native_energy_config_wires_projection_mode_and_octaves():
    image = np.zeros((9, 9, 9), dtype=np.float32)
    config = _prepare_energy_config(
        image,
        {
            "energy_projection_mode": "paper",
            "radius_of_smallest_vessel_in_microns": 1.0,
            "radius_of_largest_vessel_in_microns": 4.0,
            "scales_per_octave": 1.0,
        },
    )

    assert config["energy_projection_mode"] == "paper"
    assert len(config["octave_at_scales"]) == len(config["lumen_radius_microns"])
    assert config["scale_resolution_factors"].shape == (len(config["lumen_radius_microns"]), 3)
    assert np.all(config["scale_resolution_factors"] >= 1)


def test_native_hessian_energy_changes_when_psf_and_kernel_ratios_change():
    image = np.zeros((9, 9, 9), dtype=np.float32)
    image[4, :, 4] = 1.0

    config_default = _prepare_energy_config(
        image,
        {
            "radius_of_smallest_vessel_in_microns": 1.0,
            "radius_of_largest_vessel_in_microns": 2.0,
            "scales_per_octave": 1.0,
            "approximating_PSF": True,
            "gaussian_to_ideal_ratio": 1.0,
            "spherical_to_annular_ratio": 1.0,
        },
    )
    config_modified = _prepare_energy_config(
        image,
        {
            "radius_of_smallest_vessel_in_microns": 1.0,
            "radius_of_largest_vessel_in_microns": 2.0,
            "scales_per_octave": 1.0,
            "approximating_PSF": False,
            "gaussian_to_ideal_ratio": 0.4,
            "spherical_to_annular_ratio": 0.2,
        },
    )

    energy_default = native_hessian.compute_native_hessian_energy(image, config_default, 0)
    energy_modified = native_hessian.compute_native_hessian_energy(image, config_modified, 0)

    assert not np.allclose(energy_default, energy_modified, equal_nan=True)
