"""Property-based tests for orientation persistence round-trip.

# Feature: matlab-python-parity, Property 4: Orientation Persistence Round-Trip

Verifies that persisting an internal [Y, X, Z] array to physical [Z, Y, X] storage
and then loading it back (transposing to [Y, X, Z]) is the identity operation.

The two transpose operations in the codebase are:
  - Persist  [Y, X, Z] → [Z, Y, X]: np.transpose(arr, (2, 0, 1))
  - Load     [Z, Y, X] → [Y, X, Z]: np.transpose(arr, (1, 2, 0))

Their composition must equal the identity for any float64 array.

Validates: Requirements 3.3
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def persist_yxz_to_zyx(arr: np.ndarray) -> np.ndarray:
    """Map internal [Y, X, Z] orientation to physical [Z, Y, X] storage."""
    return np.transpose(arr, (2, 0, 1))


def load_zyx_to_yxz(arr: np.ndarray) -> np.ndarray:
    """Map physical [Z, Y, X] storage back to internal [Y, X, Z] orientation."""
    return np.transpose(arr, (1, 2, 0))


# ---------------------------------------------------------------------------
# Property 4: Orientation Persistence Round-Trip
# ---------------------------------------------------------------------------


@pytest.mark.unit
@given(
    arr=arrays(
        dtype=np.float64,
        shape=st.tuples(
            st.integers(min_value=1, max_value=8),  # Y
            st.integers(min_value=1, max_value=8),  # X
            st.integers(min_value=1, max_value=8),  # Z
        ),
    )
)
@settings(max_examples=100)
def test_orientation_round_trip(arr: np.ndarray) -> None:
    """Persisting [Y,X,Z]→[Z,Y,X] then loading back [Z,Y,X]→[Y,X,Z] is identity.

    The round-trip must recover the exact original array (floating-point identity,
    not just approximate equality), because transposition is a lossless view
    operation that never modifies element values.
    """
    persisted = persist_yxz_to_zyx(arr)
    recovered = load_zyx_to_yxz(persisted)

    # equal_nan=True: transposition is a lossless view — NaN in position i must
    # remain NaN in the same position after the round-trip, so NaN == NaN here.
    assert np.array_equal(recovered, arr, equal_nan=True), (
        f"Round-trip [Y,X,Z]→[Z,Y,X]→[Y,X,Z] did not recover original array. "
        f"Input shape: {arr.shape}, persisted shape: {persisted.shape}, "
        f"recovered shape: {recovered.shape}"
    )


@pytest.mark.unit
@given(
    arr=arrays(
        dtype=np.float64,
        shape=st.tuples(
            st.integers(min_value=1, max_value=8),  # Y
            st.integers(min_value=1, max_value=8),  # X
            st.integers(min_value=1, max_value=8),  # Z
        ),
    )
)
@settings(max_examples=100)
def test_orientation_round_trip_shapes(arr: np.ndarray) -> None:
    """Intermediate [Z,Y,X] shape and final recovered shape are consistent.

    After persist: shape becomes (Z, Y, X).
    After load back: shape returns to (Y, X, Z).
    """
    Y, X, Z = arr.shape

    persisted = persist_yxz_to_zyx(arr)
    assert persisted.shape == (Z, Y, X), (
        f"Persisted shape {persisted.shape} != expected (Z={Z}, Y={Y}, X={X})"
    )

    recovered = load_zyx_to_yxz(persisted)
    assert recovered.shape == (Y, X, Z), (
        f"Recovered shape {recovered.shape} != original (Y={Y}, X={X}, Z={Z})"
    )
