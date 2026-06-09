"""Tests for normalized Oracle Artifact readiness helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.io import savemat

from slavv_python.analytics.parity.oracle_artifacts import ensure_oracle_artifacts

if TYPE_CHECKING:
    from pathlib import Path


def _write_energy_batch(oracle_root: Path) -> Path:
    batch_dir = oracle_root / "01_Input" / "matlab_results" / "batch_test"
    data_dir = batch_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    savemat(
        data_dir / "energy_test.mat",
        {
            "energy": np.array([[[-1.0], [-2.0]]], dtype=np.float64),
            "scale_indices": np.array([[[1], [2]]], dtype=np.int16),
            "energy_4d": np.array([[[[-1.0, -2.0]]]], dtype=np.float64),
            "lumen_radius_microns": np.array([1.0, 2.0], dtype=np.float64),
        },
    )
    return batch_dir


def test_ensure_oracle_artifacts_repairs_missing_stage(tmp_path: Path) -> None:
    oracle_root = tmp_path / "oracle"
    _write_energy_batch(oracle_root)

    statuses = ensure_oracle_artifacts(oracle_root, stages=("energy",))

    status = statuses["energy"]
    assert status.ready is True
    assert status.repaired is True
    assert status.path.is_file()
    assert status.path.with_name("energy.pkl.sha256").is_file()
    assert status.summary["energy"] == {
        "kind": "ndarray",
        "shape": [1, 2, 1],
        "dtype": "float64",
    }
    assert status.summary["scale_indices"]["dtype"] == "int64"


def test_ensure_oracle_artifacts_can_verify_without_repair(tmp_path: Path) -> None:
    oracle_root = tmp_path / "oracle"
    _write_energy_batch(oracle_root)
    ensure_oracle_artifacts(oracle_root, stages=("energy",))

    statuses = ensure_oracle_artifacts(oracle_root, stages=("energy",), repair=False)

    assert statuses["energy"].ready is True
    assert statuses["energy"].repaired is False


def test_ensure_oracle_artifacts_reports_missing_without_repair(tmp_path: Path) -> None:
    oracle_root = tmp_path / "oracle"
    _write_energy_batch(oracle_root)

    statuses = ensure_oracle_artifacts(oracle_root, stages=("energy",), repair=False)

    assert statuses["energy"].ready is False
    assert statuses["energy"].error == "missing normalized Oracle Artifact"
