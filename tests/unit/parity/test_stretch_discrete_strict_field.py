"""Tests for stretch discrete strict-field classification (U5)."""

from __future__ import annotations

from pathlib import Path  # noqa: TC003

from slavv_python.analytics.parity.proof.stretch import (
    ClassifiedStretchFailure,
    StretchFieldSet,
    StretchStatus,
    evaluate_stretch_discrete_connections,
    expand_unlock_with_discrete,
    load_stretch_unlock,
    write_stretch_unlock,
)


def test_connections_mismatch_is_incomplete_discrete(tmp_path: Path) -> None:
    write_stretch_unlock(
        tmp_path / "unlock.json",
        field_set=StretchFieldSet.ENERGY,
        dest_run_root=tmp_path / "crop",
        oracle_root=tmp_path / "oracle",
        proof_path=tmp_path / "proof.json",
    )
    result = evaluate_stretch_discrete_connections(
        matlab_connections=[[1, 2], [3, 4]],
        python_connections=[[1, 2], [4, 3]],
        energy_unlock_present=True,
    )
    assert isinstance(result, ClassifiedStretchFailure)
    assert result.status == StretchStatus.INCOMPLETE_DISCRETE


def test_exact_connections_expand_unlock(tmp_path: Path) -> None:
    unlock = tmp_path / "unlock.json"
    write_stretch_unlock(
        unlock,
        field_set=StretchFieldSet.ENERGY,
        dest_run_root=tmp_path / "crop",
        oracle_root=tmp_path / "oracle",
        proof_path=tmp_path / "proof.json",
    )
    result = evaluate_stretch_discrete_connections(
        matlab_connections=[[1, 2], [3, 4]],
        python_connections=[[1, 2], [3, 4]],
        energy_unlock_present=True,
    )
    assert result == StretchStatus.STRETCH_COMPLETE
    expand_unlock_with_discrete(
        unlock,
        dest_run_root=tmp_path / "crop",
        oracle_root=tmp_path / "oracle",
        proof_path=tmp_path / "discrete_proof.json",
    )
    assert load_stretch_unlock(unlock).field_set == StretchFieldSet.ENERGY_AND_DISCRETE


def test_adr0012_ownership_alone_does_not_expand_without_exact_connections(
    tmp_path: Path,
) -> None:
    """Ownership-map green is not discrete stretch complete — need exact connections."""
    result = evaluate_stretch_discrete_connections(
        matlab_connections=[[1, 2]],
        python_connections=[[2, 1]],
        energy_unlock_present=True,
    )
    assert isinstance(result, ClassifiedStretchFailure)
    assert result.status == StretchStatus.INCOMPLETE_DISCRETE
