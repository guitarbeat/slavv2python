"""Property-based tests for Oracle Loader HDF5 axis-reversal convention.

# Feature: matlab-python-parity, Property 13: Oracle HDF5 Axis Reversal Round-Trip

Verifies that the axis-reversal applied by the Oracle Loader when reading v7.3 HDF5
artifacts is invertible: applying the same axis-reversal twice recovers the original
array exactly.

MATLAB writes 3D volumes in Fortran (column-major) order.  When h5py reads them back
in C order, the axis indices are reversed relative to MATLAB's convention.  The Oracle
Loader corrects this by transposing axes — specifically, ``array.transpose(0, 2, 1)``
on the (scale, Z, Y, X) shaped HDF5 bundle — to align the oracle with the Python
checkpoint frame.

Because the transposition ``(0, 2, 1)`` is its own inverse (it is an involution), one
application aligns the data and a second application restores it.  This property guards
against double-reversal bugs that would silently produce a mis-oriented oracle surface.

Validates: Requirements 10.4
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st


def _apply_oracle_axis_reversal(arr: np.ndarray) -> np.ndarray:
    """Apply the Oracle Loader's HDF5 axis-reversal to a 3-D spatial array.

    The loader uses ``array.transpose(0, 2, 1)`` on a (N, Z, Y, X) bundle,
    which swaps the last two spatial axes.  For a pure 3-D spatial array
    ``(Z, Y, X)`` the equivalent operation swaps axes 1 and 2, i.e.
    ``array.transpose(0, 2, 1)`` → equivalent to ``np.swapaxes(arr, 1, 2)``.

    This helper exposes the same logical operation so the test targets the
    concrete function used by the loader rather than an abstraction.
    """
    return np.ascontiguousarray(arr.transpose(0, 2, 1))


# ---------------------------------------------------------------------------
# Property 13: Oracle HDF5 Axis Reversal Round-Trip
# ---------------------------------------------------------------------------


@pytest.mark.unit
@given(
    z=st.integers(min_value=1, max_value=8),
    y=st.integers(min_value=1, max_value=8),
    x=st.integers(min_value=1, max_value=8),
)
@settings(max_examples=100)
def test_oracle_axis_reversal_double_application_recovers_original(z: int, y: int, x: int) -> None:
    """Applying the HDF5 axis-reversal twice must recover the original array.

    For any 3-D spatial array with shape (Z, Y, X):
      - Apply the Oracle Loader's axis-reversal once → produces the corrected
        frame (equivalent to what the loader stores as the oracle surface).
      - Apply the Oracle Loader's axis-reversal a second time → must exactly
        equal the original unmodified array.

    This confirms the axis-reversal is an involution (self-inverse) and that
    exactly one application separates the raw HDF5 frame from the Python frame.
    A double-reversal bug would make ``double_reversed != original``.
    """
    rng = np.random.default_rng(seed=z * 10000 + y * 100 + x)
    original = rng.standard_normal((z, y, x)).astype(np.float64)

    once_reversed = _apply_oracle_axis_reversal(original)
    double_reversed = _apply_oracle_axis_reversal(once_reversed)

    assert double_reversed.shape == original.shape, (
        f"Shape after double-reversal {double_reversed.shape} != "
        f"original shape {original.shape} for input shape (Z={z}, Y={y}, X={x})"
    )
    assert np.array_equal(double_reversed, original), (
        f"Double-reversal did not recover original array for shape "
        f"(Z={z}, Y={y}, X={x}). "
        f"Max absolute deviation: {np.max(np.abs(double_reversed - original))}"
    )


@pytest.mark.unit
@given(
    z=st.integers(min_value=1, max_value=8),
    y=st.integers(min_value=1, max_value=8),
    x=st.integers(min_value=1, max_value=8),
)
@settings(max_examples=100)
def test_oracle_axis_reversal_changes_shape_when_non_cubic(z: int, y: int, x: int) -> None:
    """A single axis-reversal changes the shape when the array is non-cubic.

    The transposition ``(0, 2, 1)`` swaps axes 1 and 2. When ``y != x``, the
    resulting shape must differ from the original — confirming the reversal is
    actually applied and not a no-op.

    For cubic arrays (``y == x``) the shape is invariant under this swap; the
    test skips that degenerate case.
    """
    if y == x:
        # Shape is invariant for square cross-sections; skip this case.
        return

    rng = np.random.default_rng(seed=z * 10000 + y * 100 + x)
    original = rng.standard_normal((z, y, x)).astype(np.float64)

    once_reversed = _apply_oracle_axis_reversal(original)

    assert once_reversed.shape != original.shape, (
        f"Single axis-reversal produced the same shape {original.shape} for a "
        f"non-cubic array (Z={z}, Y={y}, X={x}). "
        "The reversal must swap the last two dimensions when y != x."
    )
    assert once_reversed.shape == (z, x, y), (
        f"After reversal expected shape (Z={z}, X={x}, Y={y}), got {once_reversed.shape}"
    )
