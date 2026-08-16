"""Tests for full-volume stretch unlock gate (U6)."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003

from slavv_python.analytics.parity.proof.stretch import (
    PHASE1_CLAIM_RUN_NAME,
    StretchFieldSet,
    StretchStatus,
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
