"""Parametrized crop Energy voxel probes for MATLAB bit-parity regression."""

from __future__ import annotations

import pytest
from tests.support.parity_harness import (
    assert_bit_parity_scale,
    assert_oracle_energy_parity,
    compare_voxel_probe_to_oracle,
    crop_harness_available,
    load_crop_image_and_config,
    load_crop_oracle_energy,
    load_voxel_regression_cases,
    probe_voxel,
)

pytestmark = [pytest.mark.unit, pytest.mark.parity, pytest.mark.regression]


def _regression_cases() -> list[dict]:
    return load_voxel_regression_cases()


@pytest.fixture(scope="module")
def crop_context():
    if not crop_harness_available():
        pytest.skip("crop harness dataset/oracle not available locally")
    image, config = load_crop_image_and_config()
    oracle_energy, oracle_scales = load_crop_oracle_energy()
    return image, config, oracle_energy, oracle_scales


@pytest.mark.parametrize(
    "case", _regression_cases(), ids=lambda case: str(case.get("id", case["voxel_zyx"]))
)
def test_crop_energy_voxel_matches_matlab_oracle(case: dict, crop_context) -> None:
    image, config, oracle_energy, oracle_scales = crop_context
    voxel_zyx = tuple(int(v) for v in case["voxel_zyx"])
    oracle_scale = int(case["oracle_scale"])

    probe = probe_voxel(image, config, voxel_zyx=voxel_zyx, oracle_scale=oracle_scale)
    report = compare_voxel_probe_to_oracle(
        probe,
        voxel_zyx=voxel_zyx,
        oracle_energy=oracle_energy,
        oracle_scales=oracle_scales,
    )

    assert_bit_parity_scale(report["actual_scale"], report["expected_scale"])
    assert_oracle_energy_parity(report["actual_energy"], report["expected_energy"])
    if "oracle_energy" in case:
        assert_oracle_energy_parity(report["actual_energy"], float(case["oracle_energy"]))

    assert report["passed"] is True
