from __future__ import annotations

import numpy as np

from slavv_python.analytics.parity.energy_ulp_proof import (
    EnergyFloatGateOptions,
    build_energy_ulp_proof_report,
    evaluate_energy_float_gate,
)


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


def test_energy_ulp_proof_denorm_escape_allows_high_ulp_when_delta_is_tiny() -> None:
    matlab_energy = np.array([1e-20], dtype=np.float64)
    python_energy = np.array([2e-20], dtype=np.float64)
    scales = np.array([1], dtype=np.int16)

    report = build_energy_ulp_proof_report(
        matlab_energy,
        python_energy,
        scales,
        scales,
        max_ulps=8,
    )

    assert report["passed"] is True
    assert report["scale_agree_denorm_escape_count"] == 1


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


def test_certification_gate_passes_near_zero_high_ulp_within_allclose() -> None:
    # A near-zero energy where ULP distance is huge but |Delta| is trivial.
    matlab_energy = np.array([1e-3], dtype=np.float64)
    python_energy = np.array([1e-3 + 2e-11], dtype=np.float64)
    scales = np.array([5], dtype=np.int16)

    gate = evaluate_energy_float_gate(
        matlab_energy, python_energy, scales, scales, options=EnergyFloatGateOptions()
    )

    assert gate["use_allclose"] is True
    assert gate["passed"] is True
    assert gate["scale_agree_tol_over_max_count"] == 0
    # ULP is still reported as a diagnostic and is large here.
    assert gate["scale_agree_energy_ulp_over_max_count"] >= 1


def test_certification_gate_fails_when_delta_exceeds_allclose_tol() -> None:
    matlab_energy = np.array([1.0], dtype=np.float64)
    python_energy = np.array([1.0 + 1e-6], dtype=np.float64)
    scales = np.array([5], dtype=np.int16)

    gate = evaluate_energy_float_gate(
        matlab_energy, python_energy, scales, scales, options=EnergyFloatGateOptions()
    )

    assert gate["passed"] is False
    assert gate["scale_agree_tol_over_max_count"] == 1


def test_certification_gate_strict_scales() -> None:
    matlab_energy = np.zeros((2,), dtype=np.float64)
    python_energy = np.zeros((2,), dtype=np.float64)
    matlab_scales = np.array([1, 2], dtype=np.int16)
    python_scales = np.array([1, 3], dtype=np.int16)

    gate = evaluate_energy_float_gate(
        matlab_energy, python_energy, matlab_scales, python_scales,
        options=EnergyFloatGateOptions(),
    )

    assert gate["passed"] is False
    assert gate["scale_mismatch_count"] == 1
