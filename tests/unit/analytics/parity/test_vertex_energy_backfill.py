"""Curated MATLAB vertices carry a rank-ramp energy; the loader must recover the
true physical energies from the raw vertices*.mat sibling (matched by position)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy.io import savemat

from slavv_python.analytics.parity.matlab_vector_loader import load_normalized_matlab_stage

if TYPE_CHECKING:
    from pathlib import Path


def _write_vertices(
    path: Path, positions: np.ndarray, scales: np.ndarray, energies: np.ndarray
) -> None:
    savemat(
        str(path),
        {
            "vertex_space_subscripts": positions.astype(np.uint16),
            "vertex_scale_subscripts": scales.astype(np.uint8).reshape(-1, 1),
            "vertex_energies": energies.astype(np.float64).reshape(-1, 1),
        },
    )


@pytest.mark.unit
@pytest.mark.parity
def test_curated_energies_recovered_from_raw(tmp_path: Path) -> None:
    raw_positions = np.array([[10, 20, 30], [40, 50, 60], [70, 80, 90]])
    raw_energies = np.array([-5.5, -3.3, -1.1])
    _write_vertices(tmp_path / "vertices_t.mat", raw_positions, np.array([2, 3, 4]), raw_energies)

    # Curated: subset (reordered) with a corrupted rank-ramp energy.
    curated_positions = np.array([[40, 50, 60], [10, 20, 30]])
    _write_vertices(
        tmp_path / "curated_vertices_t.mat",
        curated_positions,
        np.array([3, 2]),
        np.array([-65535.0, -32767.0]),
    )

    result = load_normalized_matlab_stage(tmp_path / "curated_vertices_t.mat", "vertices")

    # Energies must be the raw physical values for those positions, NOT the ramp.
    assert np.allclose(np.asarray(result["energies"]).ravel(), [-3.3, -5.5])


@pytest.mark.unit
@pytest.mark.parity
def test_curated_energies_fallback_without_raw_sibling(tmp_path: Path) -> None:
    # No raw vertices*.mat present -> fall back to the curated energies as-is.
    curated_positions = np.array([[40, 50, 60], [10, 20, 30]])
    ramp = np.array([-65535.0, -32767.0])
    _write_vertices(tmp_path / "curated_vertices_t.mat", curated_positions, np.array([3, 2]), ramp)

    result = load_normalized_matlab_stage(tmp_path / "curated_vertices_t.mat", "vertices")
    assert np.allclose(np.asarray(result["energies"]).ravel(), ramp)
