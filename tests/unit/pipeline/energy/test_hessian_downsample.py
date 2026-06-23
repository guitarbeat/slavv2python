from __future__ import annotations

import inspect

import numpy as np
import numpy.testing as npt
import pytest

from slavv_python.pipeline.energy import hessian_response as native_hessian
from slavv_python.pipeline.energy.config import _prepare_energy_config
from slavv_python.pipeline.energy.exact_mesh import (
    _interp3_matlab_linear_inf,
    _matlab_coarse_local_slices,
    _matlab_zero_based_linspace,
    compute_exact_parity_energy_chunked,
    get_chunking_lattice_v190,
    get_starts_and_counts_v200,
)
from slavv_python.pipeline.energy.math import compute_principal_energy
from slavv_python.pipeline.energy.policy import EnergyPolicy

EXACT_ENERGY_POLICY = EnergyPolicy(
    precision=np.dtype(np.float64),
    intensity_scaling=False,
    downsample_alignment="matlab",
    mesh_strategy="linspace",
    interpolation_mode="matlab_inf_prop",
    exact_sign_clipping=True,
)


def test_downsample_volume_uses_matlab_last_chunk_stride_phase() -> None:
    # MATLAB get_starts_and_counts_V200 aligns the last/single chunk so the strided
    # read lands on the final pixel: 0-based start phase is (size - 1) mod rf per axis.
    image = np.arange(64 * 64 * 64, dtype=np.float32).reshape(64, 64, 64)
    resolution_factor = np.array([4, 4, 4], dtype=np.int16)

    downsampled = native_hessian._downsample_volume(
        image, resolution_factor, policy=EXACT_ENERGY_POLICY
    )

    # (64 - 1) % 4 == 3
    npt.assert_array_equal(downsampled, image[3::4, 3::4, 3::4])
    assert downsampled[0, 0, 0] == image[3, 3, 3]


def test_downsample_volume_anisotropic_phase_lands_on_last_pixel() -> None:
    # rf that does not divide (size - 1) evenly: phase differs from rf - 1.
    image = np.arange(64 * 256 * 256, dtype=np.float32).reshape(64, 256, 256)
    resolution_factor = np.array([9, 20, 20], dtype=np.int16)

    downsampled = native_hessian._downsample_volume(
        image, resolution_factor, policy=EXACT_ENERGY_POLICY
    )

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


def test_resolution_factors_use_matlab_rounding_at_large_scales() -> None:
    image = np.zeros((64, 256, 256), dtype=np.float64)
    params = {
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
        "energy_axis_permutation": [2, 0, 1],
        "microns_per_voxel": [0.916, 0.916, 1.99688],
    }

    config = _prepare_energy_config(image, params)

    npt.assert_array_equal(config["scale_resolution_factors"][72], [5, 10, 10])


def test_exact_mesh_uses_shared_matlab_principal_energy_kernel() -> None:
    source = inspect.getsource(compute_exact_parity_energy_chunked)

    assert "compute_principal_energy(" in source
    assert "np.linalg.eigh" not in source


def test_exact_mesh_maps_resolution_factors_to_zyx_input() -> None:
    source = inspect.getsource(compute_exact_parity_energy_chunked)

    # ``rf`` is in the raw [Z, Y, X] input frame. Chunk slicing must retain
    # that mapping before each chunk is transposed into MATLAB [Y, X, Z].
    assert "stride_z = int(rf[0])" in source
    assert "stride_y = int(rf[1])" in source
    assert "stride_x = int(rf[2])" in source


def test_principal_energy_preserves_matlab_eig_component_order() -> None:
    gradients = np.zeros((1, 3), dtype=np.float64)
    # MATLAB energy_filter_V200 uses eig's returned component order, then clips
    # the third component. For this diagonal Hessian that is [-10, -1, 2].
    curvatures = np.array([[-1.0, 2.0, -10.0, 0.0, 0.0, 0.0]], dtype=np.float64)

    energy = compute_principal_energy(gradients, curvatures, energy_sign=-1.0)

    npt.assert_array_equal(energy, np.array([-11.0]))


CROP_EXACT_PARAMS = {
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
    "max_voxels_per_node_energy": 6_000,
    "energy_axis_permutation": [2, 0, 1],
    "microns_per_voxel": [0.916, 0.916, 1.99688],
}


def _crop_exact_energy_config(image_shape: tuple[int, int, int]) -> dict:
    return _prepare_energy_config(np.zeros(image_shape, dtype=np.float64), CROP_EXACT_PARAMS)


def _matlab_coarse_slice_geometry(
    image_shape: tuple[int, int, int],
    *,
    current_octave: int,
    chunk_idx: int = 0,
) -> dict[str, object]:
    config = _crop_exact_energy_config(image_shape)
    matlab_image_shape = np.array([image_shape[1], image_shape[2], image_shape[0]], dtype=float)
    microns_per_voxel = np.asarray(config["microns_per_voxel"], dtype=float)
    pixels_per_sigma_psf = np.asarray(config["pixels_per_sigma_PSF"], dtype=float)
    octave_at_scales = config["octave_at_scales"]
    lumen_radius_microns = np.asarray(config["lumen_radius_microns"], dtype=float)

    scale_indices = np.where(octave_at_scales == current_octave)[0]
    rf_zyx = np.asarray(config["scale_resolution_factors"][scale_indices[0]], dtype=float)
    rf_matlab = np.array([rf_zyx[1], rf_zyx[2], rf_zyx[0]], dtype=float)
    largest_radius = lumen_radius_microns[scale_indices[-1]]
    largest_pixels_per_radius = largest_radius / microns_per_voxel
    approx_size = np.round(matlab_image_shape / rf_matlab)
    microns_per_pixel = microns_per_voxel * rf_zyx
    microns_per_pixel_matlab = np.array(
        [microns_per_pixel[1], microns_per_pixel[2], microns_per_pixel[0]],
        dtype=float,
    )
    lattice_dims, _ = get_chunking_lattice_v190(
        1.0 / microns_per_pixel_matlab,
        float(config["max_voxels"]),
        approx_size,
    )
    chunk_overlap = np.ceil(
        6.0 * np.sqrt(pixels_per_sigma_psf**2 + largest_pixels_per_radius**2)
    ).astype(np.int32)[[1, 2, 0]]
    starts_counts = get_starts_and_counts_v200(
        lattice_dims,
        chunk_overlap,
        matlab_image_shape,
        rf_matlab,
    )

    stride_y, stride_x, stride_z = int(rf_zyx[1]), int(rf_zyx[2]), int(rf_zyx[0])
    y_idx, x_idx, z_idx = np.unravel_index(chunk_idx, lattice_dims, order="F")
    read_counts_yxz = (
        int(starts_counts[3][y_idx]),
        int(starts_counts[4][x_idx]),
        int(starts_counts[5][z_idx]),
    )
    strided_read_shape = (
        len(range(0, read_counts_yxz[0], stride_y)),
        len(range(0, read_counts_yxz[1], stride_x)),
        len(range(0, read_counts_yxz[2], stride_z)),
    )
    padded_shape = native_hessian._fourier_transform_input(
        np.zeros(strided_read_shape, dtype=np.float64)
    ).shape
    write_counts = (
        int(starts_counts[9][y_idx]),
        int(starts_counts[10][x_idx]),
        int(starts_counts[11][z_idx]),
    )
    offsets = (
        int(starts_counts[12][y_idx]),
        int(starts_counts[13][x_idx]),
        int(starts_counts[14][z_idx]),
    )
    y_local, x_local, z_local = _matlab_coarse_local_slices(
        offsets=offsets,
        write_counts=write_counts,
        strides=(stride_y, stride_x, stride_z),
        padded_shape=padded_shape,
    )
    retained_shape = (
        y_local.stop - y_local.start,
        x_local.stop - x_local.start,
        z_local.stop - z_local.start,
    )
    requested_shape = tuple(
        1
        + int(np.ceil((offset + write_count - 1) / stride))
        - max(0, int(np.floor(offset / stride)))
        for offset, write_count, stride in zip(
            offsets, write_counts, (stride_y, stride_x, stride_z)
        )
    )
    return {
        "octave": int(current_octave),
        "chunk_idx": int(chunk_idx),
        "rf_zyx": rf_zyx,
        "rf_matlab_yxz": rf_matlab,
        "strided_read_shape": strided_read_shape,
        "padded_shape": padded_shape,
        "requested_shape": requested_shape,
        "retained_shape": retained_shape,
        "offsets": offsets,
        "write_counts": write_counts,
        "local_slices": (y_local, x_local, z_local),
    }


def _assert_exact_crop_chunk_slices_stay_inside_padded_fft_grid(
    image_shape: tuple[int, int, int],
) -> None:
    config = _crop_exact_energy_config(image_shape)
    octave_at_scales = config["octave_at_scales"]

    for current_octave in np.unique(octave_at_scales):
        geometry = _matlab_coarse_slice_geometry(
            image_shape,
            current_octave=int(current_octave),
            chunk_idx=0,
        )
        rf_matlab = geometry["rf_matlab_yxz"]  # type: ignore[misc]
        matlab_image_shape = np.array([image_shape[1], image_shape[2], image_shape[0]], dtype=float)
        microns_per_voxel = np.asarray(config["microns_per_voxel"], dtype=float)
        microns_per_pixel = microns_per_voxel * geometry["rf_zyx"]  # type: ignore[operator]
        microns_per_pixel_matlab = np.array(
            [microns_per_pixel[1], microns_per_pixel[2], microns_per_pixel[0]],
            dtype=float,
        )
        _, number_of_chunks = get_chunking_lattice_v190(
            1.0 / microns_per_pixel_matlab,
            float(config["max_voxels"]),
            np.round(matlab_image_shape / rf_matlab),
        )
        for chunk_idx in range(int(number_of_chunks)):
            geometry = _matlab_coarse_slice_geometry(
                image_shape,
                current_octave=int(current_octave),
                chunk_idx=chunk_idx,
            )
            y_local, x_local, z_local = geometry["local_slices"]  # type: ignore[misc]
            padded_shape = geometry["padded_shape"]  # type: ignore[misc]
            local_stops = (y_local.stop, x_local.stop, z_local.stop)
            local_starts = (y_local.start, x_local.start, z_local.start)

            assert all(stop <= padded for stop, padded in zip(local_stops, padded_shape))
            assert all(stop >= start for start, stop in zip(local_starts, local_stops))


def test_exact_crop_chunk_slices_stay_inside_padded_fft_grid() -> None:
    _assert_exact_crop_chunk_slices_stay_inside_padded_fft_grid((256, 64, 256))


def test_exact_crop_unpermuted_resume_shape_avoids_coarse_slice_overrun() -> None:
    _assert_exact_crop_chunk_slices_stay_inside_padded_fft_grid((64, 256, 256))


def test_exact_crop_coarse_slice_retains_padded_fft_support_not_strided_read() -> None:
    geometry = _matlab_coarse_slice_geometry((64, 256, 256), current_octave=5, chunk_idx=0)

    npt.assert_array_equal(geometry["rf_zyx"], np.array([9.0, 20.0, 20.0]))
    npt.assert_array_equal(geometry["strided_read_shape"], (13, 13, 8))
    npt.assert_array_equal(geometry["padded_shape"], (14, 14, 10))
    npt.assert_array_equal(geometry["requested_shape"], (14, 14, 8))
    npt.assert_array_equal(geometry["retained_shape"], (14, 14, 8))
    assert geometry["retained_shape"][0] > geometry["strided_read_shape"][0]
    assert geometry["retained_shape"][1] > geometry["strided_read_shape"][1]


def test_exact_crop_coarse_slice_octave4_chunk0_matches_matlab_local_ranges() -> None:
    geometry = _matlab_coarse_slice_geometry((64, 256, 256), current_octave=4, chunk_idx=0)

    npt.assert_array_equal(geometry["rf_zyx"], np.array([5.0, 10.0, 10.0]))
    npt.assert_array_equal(geometry["strided_read_shape"], (26, 26, 13))
    npt.assert_array_equal(geometry["padded_shape"], (28, 28, 14))
    npt.assert_array_equal(geometry["requested_shape"], (27, 27, 14))
    npt.assert_array_equal(geometry["retained_shape"], (27, 27, 14))


def test_matlab_coarse_slice_rejects_support_outside_padded_fft_grid() -> None:
    """MATLAB local ranges must fit the padded FFT result, never be silently shortened."""
    with pytest.raises(ValueError, match="requested coarse support exceeds padded FFT grid"):
        _matlab_coarse_local_slices(
            offsets=(0, 0, 0),
            write_counts=(271, 1, 1),
            strides=(10, 1, 1),
            padded_shape=(27, 1, 1),
        )


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


def test_exact_interp3_returns_extrapval_outside_domain() -> None:
    volume = np.arange(8, dtype=np.float64).reshape(2, 2, 2)
    outside = np.array([[[[-0.25]]], [[[0.5]]], [[[0.5]]]], dtype=np.float64)
    npt.assert_allclose(_interp3_matlab_linear_inf(volume, outside, cval=0.0), [[[0.0]]])


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


def test_exact_mesh_preserves_matlab_linspace_positive_lead_for_crop_scale54() -> None:
    mesh = _matlab_zero_based_linspace(offset=0, stride=3, count=51, local_start=0)

    assert mesh[15] == np.nextafter(5.0, np.inf)
    assert mesh[30] == np.nextafter(10.0, np.inf)
    assert mesh[33] == 11.0
    assert mesh[42] == 14.0


def test_exact_mesh_preserves_matlab_endpoint_arithmetic_at_chunk_boundary() -> None:
    # Crop energy parity voxel (z=0, y=6, x=96) depends on this X mesh value.
    # np.linspace(0, 85, 256)[45] is 15.000000000000002, which gives the
    # neighboring Inf a positive interpolation weight and suppresses MATLAB's
    # scale-36 winner. MATLAB's 1-based formula lands exactly on 15.0 here.
    mesh = _matlab_zero_based_linspace(offset=51, stride=3, count=51, local_start=17)
    np_mesh = np.linspace(0.0, 50.0 / 3.0, 51, dtype=np.float64)

    assert mesh[45] == 15.0
    assert np_mesh[45] > 15.0


def test_exact_mesh_preserves_matlab_x_mesh_at_crop_voxel_0_43_104() -> None:
    # First post-rerun scale mismatch: Python drifted to mesh_x≈0.999 instead of
    # MATLAB's exact coarse X=2 (1-based), selecting the Inf corner at x=0.
    mesh = _matlab_zero_based_linspace(offset=73, stride=3, count=52, local_start=24)

    assert mesh[2] == 1.0
    assert np.floor(mesh[2]) == 1.0

    stride9_mesh = _matlab_zero_based_linspace(offset=0, stride=9, count=64, local_start=0)
    stride20_mesh = _matlab_zero_based_linspace(offset=15, stride=20, count=64, local_start=0)

    assert stride9_mesh[1] > np.linspace(0.0, 7.0, 64, dtype=np.float64)[1]
    assert stride20_mesh[-1] > np.linspace(0.75, 3.9, 64, dtype=np.float64)[-1]
