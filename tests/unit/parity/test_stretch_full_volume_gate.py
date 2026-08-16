"""Tests for full-volume stretch unlock gate (U6)."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003

import numpy as np

from slavv_python.analytics.parity.proof.artifact_comparator import compare_exact_artifacts
from slavv_python.analytics.parity.proof.energy_ulp_proof import EnergyFloatGateOptions
from slavv_python.analytics.parity.proof.stretch import (
    PHASE1_CLAIM_RUN_NAME,
    StretchFieldSet,
    StretchStatus,
    classify_stretch_energy_orientation,
    gate_full_stretch_entry,
    write_stretch_status,
    write_stretch_unlock,
)


def test_full_entry_without_unlock_refused(tmp_path: Path) -> None:
    decision = gate_full_stretch_entry(
        unlock_path=tmp_path / "missing.json",
        requested_field_set=StretchFieldSet.ENERGY,
        dest_run_root=tmp_path / "full_stretch_v1",
        oracle_root=tmp_path / "oracle",
    )
    assert decision.allowed is False
    assert decision.status == StretchStatus.FULL_REFUSED


def test_energy_only_unlock_cannot_claim_full_discrete(tmp_path: Path) -> None:
    oracle = tmp_path / "oracle"
    unlock = tmp_path / "unlock.json"
    write_stretch_unlock(
        unlock,
        field_set=StretchFieldSet.ENERGY,
        dest_run_root=tmp_path / "crop",
        oracle_root=oracle,
        proof_path=tmp_path / "proof.json",
    )
    decision = gate_full_stretch_entry(
        unlock_path=unlock,
        requested_field_set=StretchFieldSet.ENERGY_AND_DISCRETE,
        dest_run_root=tmp_path / "full_stretch_v1",
        oracle_root=oracle,
    )
    assert decision.allowed is False


def test_completing_stretch_status_does_not_flip_one_truth(tmp_path: Path) -> None:
    findings = tmp_path / "EXACT_PROOF_FINDINGS.md"
    closed = "Certification is CLOSED on canonical_full_v18\n"
    findings.write_text(closed, encoding="utf-8")
    write_stretch_status(
        tmp_path / "stretch_status.json",
        status=StretchStatus.STRETCH_COMPLETE,
        findings_path=findings,
        note="full green",
    )
    assert findings.read_text(encoding="utf-8") == closed


def test_refuse_overwrite_phase1_claim_root(tmp_path: Path) -> None:
    decision = gate_full_stretch_entry(
        unlock_path=tmp_path / "unlock.json",
        requested_field_set=StretchFieldSet.ENERGY,
        dest_run_root=tmp_path / PHASE1_CLAIM_RUN_NAME,
        oracle_root=tmp_path / "oracle",
    )
    assert decision.allowed is False
    assert PHASE1_CLAIM_RUN_NAME in decision.reason


def test_orientation_512_64_512_is_infra_not_float_bar() -> None:
    decision = classify_stretch_energy_orientation(
        energy_shape=(512, 64, 512),
        oracle_shape=(64, 512, 512),
    )
    assert decision is not None
    assert decision.allowed is False
    assert decision.status == StretchStatus.INCOMPLETE_INFRA
    assert "orientation" in decision.reason


def test_matching_energy_shape_is_not_orientation_refuse() -> None:
    assert (
        classify_stretch_energy_orientation(
            energy_shape=(64, 256, 256),
            oracle_shape=(64, 256, 256),
        )
        is None
    )


def test_energy_compare_refuses_orientation_before_ulp() -> None:
    oracle = np.zeros((2, 4, 4), dtype=np.float64)
    swapped = np.zeros((4, 2, 4), dtype=np.float64)
    payload_oracle = {
        "energy": oracle,
        "scale_indices": np.ones((2, 4, 4), dtype=np.int64),
        "energy_4d": np.empty((0, 0, 0, 0), dtype=np.float64),
        "lumen_radius_microns": np.array([1.0], dtype=np.float64),
    }
    payload_swapped = {
        **payload_oracle,
        "energy": swapped,
        "scale_indices": np.ones((4, 2, 4), dtype=np.int64),
    }
    report = compare_exact_artifacts(
        {"energy": payload_oracle},
        {"energy": payload_swapped},
        ("energy",),
        energy_float_options=EnergyFloatGateOptions(strict_floats=True),
    )
    assert report["passed"] is False
    gate = report["energy_float_gate"]
    assert gate["incomplete_infra"] is True
    assert gate["orientation_refuse"] is True
    assert "ulp_stats_on_mismatches" not in gate
    assert report["first_failure"]["mismatch_type"] == "incomplete_infra orientation"
