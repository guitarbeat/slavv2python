"""Characterization and gate tests for stretch unlock / status (U1)."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003

import pytest

from slavv_python.analytics.parity.proof.stretch import (
    StretchFailureClass,
    StretchFieldSet,
    StretchStatus,
    classify_stretch_failure,
    gate_full_stretch_entry,
    load_stretch_unlock,
    write_stretch_status,
    write_stretch_unlock,
)

ONE_TRUTH_CLOSED_SNIPPET = (
    "## ONE TRUTH — Phase 1 parity (validated from disk)\n\n"
    "> **Answer:** Phase 1 exact-route **Certification is CLOSED** on full `180709_E`.\n"
)


def test_full_stretch_entry_without_unlock_refuses(tmp_path: Path) -> None:
    decision = gate_full_stretch_entry(
        unlock_path=tmp_path / "missing_unlock.json",
        requested_field_set=StretchFieldSet.ENERGY,
        dest_run_root=tmp_path / "full_stretch",
        oracle_root=tmp_path / "oracle",
    )
    assert decision.allowed is False
    assert decision.status == StretchStatus.FULL_REFUSED


def test_energy_only_unlock_does_not_authorize_discrete_full(tmp_path: Path) -> None:
    unlock_path = tmp_path / "unlock.json"
    write_stretch_unlock(
        unlock_path,
        field_set=StretchFieldSet.ENERGY,
        dest_run_root=tmp_path / "crop_dest",
        oracle_root=tmp_path / "oracle",
        proof_path=tmp_path / "exact_proof_energy.json",
    )
    decision = gate_full_stretch_entry(
        unlock_path=unlock_path,
        requested_field_set=StretchFieldSet.ENERGY_AND_DISCRETE,
        dest_run_root=tmp_path / "full_stretch",
        oracle_root=tmp_path / "oracle",
    )
    assert decision.allowed is False
    assert decision.status == StretchStatus.FULL_REFUSED
    assert "discrete" in decision.reason.lower()


def test_status_writer_does_not_mutate_one_truth_closed(tmp_path: Path) -> None:
    findings = tmp_path / "EXACT_PROOF_FINDINGS.md"
    findings.write_text(ONE_TRUTH_CLOSED_SNIPPET, encoding="utf-8")
    status_path = tmp_path / "stretch_status.json"
    write_stretch_status(
        status_path,
        status=StretchStatus.CROP_ENERGY_PASSED,
        findings_path=findings,
        note="crop energy unlocked",
    )
    after = findings.read_text(encoding="utf-8")
    assert "Certification is CLOSED" in after
    assert after == ONE_TRUTH_CLOSED_SNIPPET
    payload = status_path.read_text(encoding="utf-8")
    assert StretchStatus.CROP_ENERGY_PASSED.value in payload


def test_infra_failure_is_not_blocked_float_path() -> None:
    classified = classify_stretch_failure(
        StretchFailureClass.INFRA,
        detail="MATLAB engine license unavailable",
    )
    assert classified.status == StretchStatus.INCOMPLETE_INFRA
    assert classified.status != StretchStatus.BLOCKED_FLOAT_PATH


def test_energy_unlock_authorizes_matching_energy_full(tmp_path: Path) -> None:
    oracle = tmp_path / "oracle"
    unlock_path = tmp_path / "unlock.json"
    write_stretch_unlock(
        unlock_path,
        field_set=StretchFieldSet.ENERGY,
        dest_run_root=tmp_path / "crop_dest",
        oracle_root=oracle,
        proof_path=tmp_path / "exact_proof_energy.json",
    )
    decision = gate_full_stretch_entry(
        unlock_path=unlock_path,
        requested_field_set=StretchFieldSet.ENERGY,
        dest_run_root=tmp_path / "full_stretch",
        oracle_root=oracle,
    )
    assert decision.allowed is True
    assert load_stretch_unlock(unlock_path).field_set == StretchFieldSet.ENERGY


@pytest.mark.parametrize(
    ("failure", "expected"),
    [
        (StretchFailureClass.FLOAT_PATH, StretchStatus.BLOCKED_FLOAT_PATH),
        (StretchFailureClass.DISCRETE, StretchStatus.INCOMPLETE_DISCRETE),
        (StretchFailureClass.AT_FULL, StretchStatus.INCOMPLETE_AT_FULL),
    ],
)
def test_failure_class_taxonomy(failure: StretchFailureClass, expected: StretchStatus) -> None:
    assert classify_stretch_failure(failure, detail="x").status == expected
