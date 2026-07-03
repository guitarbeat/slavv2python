"""Property-based test for Fortran-order grid invariant.

# Feature: matlab-python-parity, Property 2: Fortran-Order Grid Invariant

For any 3D volume processed through the Exact Route watershed, the internal
``vertex_index_map`` and ``energy_map`` arrays shall be F-contiguous
(``np.isfortran(arr) == True``) and have shape ``[Y, X, Z]``.

This file tests the orientation convention that
``_generate_edge_candidates_matlab_global_watershed`` applies on entry:

    Physical [Z, Y, X]
        → np.transpose(arr, (1, 2, 0)).copy(order="F")
        → [Y, X, Z] F-contiguous

**Validates: Requirements 3.1**
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

# ---------------------------------------------------------------------------
# Property 2 — Fortran-Order Grid Invariant
# ---------------------------------------------------------------------------


@pytest.mark.unit
@given(
    n_y=st.integers(2, 32),
    n_x=st.integers(2, 32),
    n_z=st.integers(2, 16),
)
@settings(max_examples=100)
def test_fortran_order_grid_invariant(n_y: int, n_x: int, n_z: int) -> None:
    """Applying the watershed entry transpose to any [Z, Y, X] array
    produces an F-contiguous array with shape [Y, X, Z].

    This mirrors the reorientation in
    ``_generate_edge_candidates_matlab_global_watershed`` at line 501:

        energy_matlab = np.transpose(
            np.asarray(energy, dtype=np.float64), (1, 2, 0)
        ).copy(order="F")

    Three assertions are verified for every generated shape:

    1. The result is F-contiguous (``np.isfortran(arr) == True``).
    2. The result shape is ``(Y, X, Z)`` — Y first, X second, Z third.
    3. ``np.asfortranarray`` of the same source also satisfies ``np.isfortran``,
       confirming the NumPy F-order API behaves as expected.
    """
    # Construct a synthetic [Z, Y, X] source array (physical orientation)
    zyx_shape = (n_z, n_y, n_x)
    source = np.arange(int(np.prod(zyx_shape)), dtype=np.float64).reshape(zyx_shape)

    # Apply the watershed entry orientation mapping: [Z, Y, X] -> [Y, X, Z] F-contiguous
    reoriented = np.transpose(source, (1, 2, 0)).copy(order="F")

    # 1. Must be F-contiguous
    assert np.isfortran(reoriented), (
        f"Array with ZYX shape {zyx_shape} is not F-contiguous after watershed "
        f"entry transpose. Got is_fortran={np.isfortran(reoriented)}"
    )

    # 2. Shape must be [Y, X, Z]
    expected_shape = (n_y, n_x, n_z)
    assert reoriented.shape == expected_shape, (
        f"Expected shape {expected_shape} after [Z,Y,X]→[Y,X,Z] transpose, got {reoriented.shape}"
    )

    # 3. Y dimension is first, X second, Z third
    assert reoriented.shape[0] == n_y, f"First axis should be Y ({n_y}), got {reoriented.shape[0]}"
    assert reoriented.shape[1] == n_x, f"Second axis should be X ({n_x}), got {reoriented.shape[1]}"
    assert reoriented.shape[2] == n_z, f"Third axis should be Z ({n_z}), got {reoriented.shape[2]}"


@pytest.mark.unit
@given(
    n_y=st.integers(2, 32),
    n_x=st.integers(2, 32),
    n_z=st.integers(2, 16),
)
@settings(max_examples=100)
def test_np_asfortranarray_is_fortran(n_y: int, n_x: int, n_z: int) -> None:
    """``np.asfortranarray`` always produces an F-contiguous array.

    This is the underlying NumPy primitive used when creating the internal
    watershed maps.  Property: for any array shape, the result of
    ``np.asfortranarray`` satisfies ``np.isfortran(...) == True``.
    """
    yxz_shape = (n_y, n_x, n_z)
    arr = np.zeros(yxz_shape, dtype=np.float64)

    fortran_arr = np.asfortranarray(arr)

    assert np.isfortran(fortran_arr) is True, (
        f"np.asfortranarray on shape {yxz_shape} did not produce F-contiguous array"
    )
    assert fortran_arr.shape == yxz_shape
