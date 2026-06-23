"""One-voxel exact-route Energy probes against the crop MATLAB oracle."""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest
from tests.support.parity_harness import (
    assert_bit_parity_scale,
    assert_oracle_energy_parity,
    crop_harness_available,
    load_crop_image_and_config,
    load_crop_oracle_energy,
    probe_voxel,
)

from slavv_python.pipeline.energy.exact_mesh import get_chunking_lattice_v190
from slavv_python.pipeline.energy.voxel_probe import resolve_write_chunk_idx_for_voxel

TARGET_VOXEL_ZYX = (12, 0, 0)
TARGET_RF_ZYX = (2, 5, 5)
MATLAB_WINNER_SCALE = 54
MATLAB_WINNER_ENERGY = -13.52067537392248

FIRST_MISMATCH_VOXEL_ZYX = (0, 43, 104)
FIRST_MISMATCH_RF_ZYX = (1, 3, 3)
FIRST_MISMATCH_SCALE = 47
FIRST_MISMATCH_ENERGY = -14.561063976954049


@pytest.mark.unit
@pytest.mark.parity
def test_crop_exact_chunk_lattice_matches_matlab_rf_255() -> None:
    if not crop_harness_available():
        pytest.skip("crop harness dataset/oracle not available locally")

    _, config = load_crop_image_and_config()
    octave_at_scales = config["octave_at_scales"]
    scale_indices = np.where(octave_at_scales == 3)[0]
    rf = np.asarray(config["scale_resolution_factors"][scale_indices[0]], dtype=float)
    matlab_shape = np.array([64, 256, 256], dtype=float)[[1, 2, 0]]
    rf_matlab = np.array([rf[1], rf[2], rf[0]], dtype=float)
    microns_per_voxel = np.asarray(config["microns_per_voxel"], dtype=float)
    approx_size = np.round(matlab_shape / rf_matlab)
    microns_per_pixel_matlab = (microns_per_voxel * rf)[[1, 2, 0]]
    lattice_dims, number_of_chunks = get_chunking_lattice_v190(
        1.0 / microns_per_pixel_matlab,
        float(config["max_voxels"]),
        approx_size,
    )
    npt.assert_array_equal(lattice_dims, np.array([3, 3, 2], dtype=np.uint16))
    assert number_of_chunks == 18


@pytest.mark.unit
@pytest.mark.parity
def test_resolve_write_chunk_idx_for_crop_voxel_12_0_0() -> None:
    if not crop_harness_available():
        pytest.skip("crop harness dataset/oracle not available locally")

    _, config = load_crop_image_and_config()
    chunk_idx = resolve_write_chunk_idx_for_voxel(
        config,
        voxel_zyx=TARGET_VOXEL_ZYX,
        target_rf_zyx=TARGET_RF_ZYX,
    )
    assert chunk_idx == 0


@pytest.mark.unit
@pytest.mark.parity
def test_voxel_probe_matches_matlab_oracle_at_12_0_0() -> None:
    if not crop_harness_available():
        pytest.skip("crop harness dataset/oracle not available locally")

    image, config = load_crop_image_and_config()
    probe = probe_voxel(
        image,
        config,
        voxel_zyx=TARGET_VOXEL_ZYX,
        oracle_scale=MATLAB_WINNER_SCALE,
    )

    oracle_energy, oracle_scales = load_crop_oracle_energy()
    matlab_energy = float(oracle_energy[TARGET_VOXEL_ZYX])
    matlab_scale = int(oracle_scales[TARGET_VOXEL_ZYX])

    assert_oracle_energy_parity(probe["octave_winner"]["upsampled_energy"], matlab_energy)
    assert_bit_parity_scale(probe["octave_winner"]["global_scale"], matlab_scale)
    assert_bit_parity_scale(matlab_scale, MATLAB_WINNER_SCALE)
    assert_oracle_energy_parity(matlab_energy, MATLAB_WINNER_ENERGY)


@pytest.mark.unit
@pytest.mark.parity
def test_voxel_probe_matches_matlab_oracle_at_0_43_104() -> None:
    if not crop_harness_available():
        pytest.skip("crop harness dataset/oracle not available locally")

    image, config = load_crop_image_and_config()
    probe = probe_voxel(
        image,
        config,
        voxel_zyx=FIRST_MISMATCH_VOXEL_ZYX,
        oracle_scale=FIRST_MISMATCH_SCALE,
    )

    oracle_energy, oracle_scales = load_crop_oracle_energy()
    matlab_energy = float(oracle_energy[FIRST_MISMATCH_VOXEL_ZYX])
    matlab_scale = int(oracle_scales[FIRST_MISMATCH_VOXEL_ZYX])

    assert_oracle_energy_parity(probe["octave_winner"]["upsampled_energy"], matlab_energy)
    assert_bit_parity_scale(probe["octave_winner"]["global_scale"], matlab_scale)
    assert_bit_parity_scale(matlab_scale, FIRST_MISMATCH_SCALE)
    assert_oracle_energy_parity(matlab_energy, FIRST_MISMATCH_ENERGY)
    assert probe["mesh_at_voxel"]["mesh_x"] == 1.0
