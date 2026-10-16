from __future__ import annotations

import numpy as np
import pytest

from slavv_python.pipeline.energy import matlab_principal_energy
from slavv_python.pipeline.energy.matlab_principal_energy import (
    _NATIVE_LIB,
    compute_principal_energy,
)


def test_native_library_available():
    """Verify that the C++20 accelerated shared library is compiled and loaded."""
    assert _NATIVE_LIB is not None, "Native C++20 principal energy library should be compiled"


@pytest.mark.parametrize("energy_sign", [-1.0, 1.0])
def test_native_vs_numpy_parity(energy_sign: float):
    """Test exact parity between C++20 analytical kernel and NumPy implementation."""
    np.random.seed(1337)
    num_voxels = 5000
    gradients = np.random.randn(num_voxels, 3)
    curvatures = np.random.randn(num_voxels, 6)

    # 1. Run with C++ enabled
    res_cpp = compute_principal_energy(gradients, curvatures, energy_sign=energy_sign)

    # 2. Run with C++ disabled (force pure NumPy fallback)
    try:
        matlab_principal_energy._NATIVE_LIB = None
        res_numpy = compute_principal_energy(gradients, curvatures, energy_sign=energy_sign)
    finally:
        matlab_principal_energy._NATIVE_LIB = _NATIVE_LIB

    # Verify infinite/finite match
    assert np.array_equal(np.isfinite(res_cpp), np.isfinite(res_numpy))

    finite_mask = np.isfinite(res_cpp)
    rel_diff = np.abs(res_cpp[finite_mask] - res_numpy[finite_mask]) / (
        np.abs(res_numpy[finite_mask]) + 1e-12
    )
    assert np.max(rel_diff) < 1e-10


def test_empty_input():
    """Verify handling of 0 voxels."""
    empty_g = np.empty((0, 3), dtype=np.float64)
    empty_c = np.empty((0, 6), dtype=np.float64)
    res = compute_principal_energy(empty_g, empty_c, energy_sign=-1.0)
    assert res.shape == (0,)
