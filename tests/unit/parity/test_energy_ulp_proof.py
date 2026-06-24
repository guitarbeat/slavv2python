from __future__ import annotations

import numpy as np

from slavv_python.analytics.parity.energy_ulp_proof import build_energy_ulp_proof_report


def test_energy_ulp_proof_passes_when_scales_match_and_energy_within_max_ulps() -> None:
    matlab_energy = np.array([-20.37433178324523], dtype=np.float64)
    python_energy = np.array([-20.374331783245218], dtype=np.float64)
    scales = np.array([90], dtype=np.int16)

    report = build_energy_ulp_proof_report(
        matlab_energy,
        python_energy,
        scales,
        scales,
        max_ulps=8,
    )

    assert report["passed"] is True
    assert report["scale_mismatch_count"] == 0
    assert report["scale_agree_energy_ulp_over_max_count"] == 0


def test_energy_ulp_proof_fails_on_scale_mismatch() -> None:
    matlab_energy = np.zeros((1, 1, 1), dtype=np.float64)
    python_energy = np.zeros((1, 1, 1), dtype=np.float64)
    matlab_scales = np.array([1], dtype=np.int16)
    python_scales = np.array([2], dtype=np.int16)

    report = build_energy_ulp_proof_report(
        matlab_energy,
        python_energy,
        matlab_scales,
        python_scales,
        max_ulps=8,
    )

    assert report["passed"] is False
    assert report["scale_mismatch_count"] == 1


def test_energy_ulp_proof_fails_when_energy_exceeds_max_ulps() -> None:
    matlab_energy = np.array([-7.305582611971218], dtype=np.float64)
    python_energy = np.array([-7.305582611971176], dtype=np.float64)
    scales = np.array([12], dtype=np.int16)

    report = build_energy_ulp_proof_report(
        matlab_energy,
        python_energy,
        scales,
        scales,
        max_ulps=8,
    )

    assert report["passed"] is False
    assert report["scale_agree_energy_ulp_over_max_count"] == 1
