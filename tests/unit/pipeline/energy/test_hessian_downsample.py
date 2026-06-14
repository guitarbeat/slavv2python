from __future__ import annotations

import numpy as np
import numpy.testing as npt

from slavv_python.pipeline.energy import hessian_response as native_hessian
from slavv_python.pipeline.energy.config import _prepare_energy_config
from slavv_python.pipeline.energy.exact_mesh import (
    _interp3_matlab_linear_inf,
    _matlab_zero_based_linspace,
    get_chunking_lattice_v190,
    get_starts_and_counts_v200,
)


def test_downsample_volume_uses_matlab_last_chunk_stride_phase() -> None:
    # MATLAB get_starts_and_counts_V200 aligns the last/single chunk so the strided
    # read lands on the final pixel: 0-based start phase is (size - 1) mod rf per axis.
    image = np.arange(64 * 64 * 64, dtype=np.float32).reshape(64, 64, 64)
    resolution_factor = np.array([4, 4, 4], dtype=np.int16)

    downsampled = native_hessian._downsample_volume(image, resolution_factor)

    # (64 - 1) % 4 == 3
    npt.assert_array_equal(downsampled, image[3::4, 3::4, 3::4])
    assert downsampled[0, 0, 0] == image[3, 3, 3]


def test_downsample_volume_anisotropic_phase_lands_on_last_pixel() -> None:
    # rf that does not divide (size - 1) evenly: phase differs from rf - 1.
    image = np.arange(64 * 256 * 256, dtype=np.float32).reshape(64, 256, 256)
    resolution_factor = np.array([9, 20, 20], dtype=np.int16)

    downsampled = native_hessian._downsample_volume(image, resolution_factor)

    # (64 - 1) % 9 == 0 ; (256 - 1) % 20 == 15
    npt.assert_array_equal(downsampled, image[0::9, 15::20, 15::20])
    assert downsampled.shape == (8, 13, 13)
    # last strided sample lands exactly on the last pixel along each axis
    assert downsampled[-1, -1, -1] == image[63, 255, 255]


def test_upsample_volume_linearly_interpolates_finite_neighbors_only() -> None:
    volume = np.full((2, 2, 2), np.inf, dtype=np.float32)
    volume[0, 0, 0] = -6.0
    volume[0, 0, 1] = -8.0
    resolution_factor = np.array([2, 2, 2], dtype=np.int16)

    upsampled = native_hessian._upsample_volume(volume, (4, 4, 4), resolution_factor)

    assert np.isfinite(upsampled[0, 0, 0])
    npt.assert_allclose(upsampled[0, 0, 0], -6.0, rtol=0.0, atol=1e-6)


def test_octave_resolution_factors_non_unity_for_large_scales() -> None:
    lumen_radius_microns = np.geomspace(1.5, 48.0, 92).astype(np.float64)
    microns_per_voxel = np.array([0.5, 0.5, 2.0], dtype=np.float64)
    scales_per_octave = 6.0

    _, scale_resolution_factors = native_hessian.matlab_octave_resolution_factors(
        lumen_radius_microns,
        microns_per_voxel,
        scales_per_octave,
    )

    assert scale_resolution_factors.shape == (92, 3)
    assert np.all(scale_resolution_factors[-1] > 1)
    assert np.all(scale_resolution_factors[0] == 1)


def test_prepare_energy_config_exposes_non_unity_scale_resolution_factors() -> None:
    image = np.zeros((64, 256, 256), dtype=np.float32)
    params = {
        "radius_of_smallest_vessel_in_microns": 1.5,
        "radius_of_largest_vessel_in_microns": 48.0,
        "scales_per_octave": 6.0,
        "approximating_PSF": True,
        "numerical_aperture": 0.95,
        "excitation_wavelength_in_microns": 1.3,
        "sample_index_of_refraction": 1.33,
        "gaussian_to_ideal_ratio": 0.5,
        "spherical_to_annular_ratio": 0.5,
        "energy_projection_mode": "matlab",
        "microns_per_voxel": [0.5, 0.5, 2.0],
    }

    config = _prepare_energy_config(image, params)

    npt.assert_array_equal(
        config["scale_resolution_factors"][0], np.array([1, 1, 1], dtype=np.int16)
    )
    assert int(np.max(config["scale_resolution_factors"])) > 1


def test_energy_axis_permutation_reorders_microns_and_psf() -> None:
    image = np.zeros((64, 256, 256), dtype=np.float32)
    base_params = {
        "radius_of_smallest_vessel_in_microns": 1.5,
        "radius_of_largest_vessel_in_microns": 60.0,
        "scales_per_octave": 6.0,
        "approximating_PSF": True,
        "numerical_aperture": 0.95,
        "excitation_wavelength_in_microns": 0.95,
        "sample_index_of_refraction": 1.33,
        "gaussian_to_ideal_ratio": 0.5,
        "spherical_to_annular_ratio": 0.5,
        "energy_projection_mode": "matlab",
        "microns_per_voxel": [0.916, 0.916, 1.99688],
    }

    permuted = _prepare_energy_config(image, {**base_params, "energy_axis_permutation": [2, 0, 1]})

    # microns reorder so the axial (large-micron) axis maps to working axis 0
    npt.assert_allclose(permuted["microns_per_voxel"], [1.99688, 0.916, 0.916])
    # the high-octave resolution factor for the 64-length axis becomes the small factor
    assert int(permuted["scale_resolution_factors"][-1][0]) < int(
        permuted["scale_resolution_factors"][-1][1]
    )


def test_get_chunking_lattice_uses_matlab_uint16_rounding() -> None:
    lattice, count = get_chunking_lattice_v190(
        np.array([1.0, 1.0, 1.0], dtype=np.float64),
        1_000_000,
        np.array([150.0, 250.0, 349.0], dtype=np.float64),
    )

    # MATLAB uint16 rounds positive doubles half-up: [1.5, 2.5, 3.49] -> [2, 3, 3].
    npt.assert_array_equal(lattice, np.array([2, 3, 3], dtype=np.uint16))
    assert count == 18


def test_get_starts_and_counts_uses_matlab_uint16_border_rounding() -> None:
    starts_counts = get_starts_and_counts_v200(
        np.array([2, 3, 1], dtype=np.uint16),
        np.array([0, 0, 0], dtype=np.uint16),
        np.array([3, 5, 4], dtype=np.uint16),
        np.array([1, 1, 1], dtype=np.uint16),
    )

    y_writing_starts = starts_counts[6]
    x_writing_starts = starts_counts[7]
    y_writing_counts = starts_counts[9]
    x_writing_counts = starts_counts[10]

    # linspace(0, 3, 3) -> [0, 1.5, 3], and MATLAB uint16(1.5) == 2.
    npt.assert_array_equal(y_writing_starts, np.array([1.0, 3.0]))
    npt.assert_array_equal(y_writing_counts, np.array([2.0, 1.0]))
    # linspace(0, 5, 4) -> [0, 1.666..., 3.333..., 5].
    npt.assert_array_equal(x_writing_starts, np.array([1.0, 3.0, 4.0]))
    npt.assert_array_equal(x_writing_counts, np.array([2.0, 1.0, 2.0]))


def test_exact_interp3_propagates_inf_only_for_positive_weight_neighbors() -> None:
    volume = np.full((2, 2, 2), np.inf, dtype=np.float64)
    volume[0, 0, 0] = -6.0

    exact_corner = np.array([[[[0.0]]], [[[0.0]]], [[[0.0]]]], dtype=np.float64)
    halfway = np.array([[[[0.5]]], [[[0.0]]], [[[0.0]]]], dtype=np.float64)

    npt.assert_allclose(_interp3_matlab_linear_inf(volume, exact_corner), [[[-6.0]]])
    assert np.isposinf(_interp3_matlab_linear_inf(volume, halfway)[0, 0, 0])


def test_exact_mesh_uses_matlab_linspace_roundoff() -> None:
    mesh = _matlab_zero_based_linspace(offset=0, stride=3, count=128, local_start=0)
    np_mesh = np.linspace(0.0, 127.0 / 3.0, 128, dtype=np.float64)

    assert mesh[27] == 9.0
    assert np_mesh[27] > 9.0
    npt.assert_allclose(mesh[-1], 127.0 / 3.0, rtol=0.0, atol=0.0)


def test_exact_mesh_preserves_matlab_endpoint_arithmetic_at_chunk_boundary() -> None:
    # Crop energy parity voxel (z=0, y=6, x=96) depends on this X mesh value.
    # np.linspace(0, 85, 256)[45] is 15.000000000000002, which gives the
    # neighboring Inf a positive interpolation weight and suppresses MATLAB's
    # scale-36 winner. MATLAB's 1-based formula lands exactly on 15.0 here.
    mesh = _matlab_zero_based_linspace(offset=51, stride=3, count=51, local_start=17)
    np_mesh = np.linspace(0.0, 50.0 / 3.0, 51, dtype=np.float64)

    assert mesh[45] == 15.0
    assert np_mesh[45] > 15.0

    stride9_mesh = _matlab_zero_based_linspace(offset=0, stride=9, count=64, local_start=0)
    stride20_mesh = _matlab_zero_based_linspace(offset=15, stride=20, count=64, local_start=0)

    assert stride9_mesh[1] > np.linspace(0.0, 7.0, 64, dtype=np.float64)[1]
    assert stride20_mesh[-1] > np.linspace(0.75, 3.9, 64, dtype=np.float64)[-1]
