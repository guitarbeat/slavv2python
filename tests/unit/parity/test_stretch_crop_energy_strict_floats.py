"""Tests for stretch crop Energy --strict-floats unlock emission (U4)."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003

from slavv_python.analytics.parity.proof.stretch import (
    StretchFieldSet,
    emit_stretch_energy_unlock_if_eligible,
    load_stretch_unlock,
)


def test_ulp_drift_style_fail_does_not_emit_unlock(tmp_path: Path) -> None:
    path = emit_stretch_energy_unlock_if_eligible(
        dest_run_root=tmp_path / "dest",
        oracle_root=tmp_path / "oracle",
        proof_path=tmp_path / "proof.json",
        report={"passed": False, "energy_float_gate": {"passed": False, "strict_floats": True}},
        strict_floats=True,
    )
    assert path is None


def test_allclose_green_without_strict_does_not_emit_unlock(tmp_path: Path) -> None:
    path = emit_stretch_energy_unlock_if_eligible(
        dest_run_root=tmp_path / "dest",
        oracle_root=tmp_path / "oracle",
        proof_path=tmp_path / "proof.json",
        report={"passed": True, "stage_summaries": {"energy": {"passed": True}}},
        strict_floats=False,
    )
    assert path is None


def test_strict_green_emits_energy_unlock(tmp_path: Path) -> None:
    dest = tmp_path / "dest"
    oracle = tmp_path / "oracle"
    proof = tmp_path / "exact_proof_energy.json"
    dest.mkdir()
    path = emit_stretch_energy_unlock_if_eligible(
        dest_run_root=dest,
        oracle_root=oracle,
        proof_path=proof,
        report={
            "passed": True,
            "stage_summaries": {"energy": {"passed": True}},
            "energy_float_gate": {"passed": True, "strict_floats": True},
        },
        strict_floats=True,
    )
    assert path is not None
    assert path.is_file()
    token = load_stretch_unlock(path)
    assert token.field_set == StretchFieldSet.ENERGY
    assert token.strict_floats is True
