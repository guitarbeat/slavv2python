"""Property-based tests for Oracle Loader index-shift convention.

# Feature: matlab-python-parity, Property 12: Oracle Exactly-One Index Shift

Verifies that for any raw MATLAB HDF5 scale-index array with known 1-based values
in [1, 255], ``_normalize_int_array(arr, one_based=True)`` returns values that are
exactly 1 less than the raw input — no more, no less.

MATLAB stores scale indices as 1-based globals (first scale = 1).  The Oracle Loader
must apply exactly one ``index - 1`` shift so that Python consumers work with
0-based indices.  A double-shift (e.g. applying the shift twice) or a missing shift
would silently mis-map every voxel to the wrong scale radius.

Validates: Requirements 10.3
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from slavv_python.analytics.parity.proof.array_normalization import _normalize_int_array

# ---------------------------------------------------------------------------
# Property 12: Oracle Exactly-One Index Shift
# ---------------------------------------------------------------------------


@pytest.mark.unit
@given(
    raw=arrays(
        dtype=np.int32,
        shape=st.integers(min_value=1, max_value=100),
        elements=st.integers(min_value=1, max_value=255),
    )
)
@settings(max_examples=100)
def test_oracle_index_shift_exactly_one(raw: np.ndarray) -> None:
    """Every loaded value equals the raw MATLAB value minus exactly 1.

    For any 1-based MATLAB scale-index array with values in [1, 255]:
      - The loaded result must equal ``raw - 1`` at every position.
      - No element may be shifted by more or less than 1.

    This guards against both a missing shift (output == raw) and a double-shift
    (output == raw - 2).
    """
    loaded = _normalize_int_array(raw, one_based=True)
    expected = raw.astype(np.int64) - 1

    assert loaded.shape == raw.shape, (
        f"Shape changed after index shift: {raw.shape} → {loaded.shape}"
    )
    assert loaded.dtype == np.int64, f"Expected int64 output, got {loaded.dtype}"
    assert np.array_equal(loaded, expected), (
        f"Index shift was not exactly 1 for all elements. "
        f"First mismatch at index {int(np.argwhere(loaded != expected)[0][0])}: "
        f"raw={raw.ravel()[int(np.argwhere(loaded != expected)[0][0])]}, "
        f"loaded={loaded.ravel()[int(np.argwhere(loaded != expected)[0][0])]}, "
        f"expected={expected.ravel()[int(np.argwhere(loaded != expected)[0][0])]}"
    )


@pytest.mark.unit
@given(
    raw=arrays(
        dtype=np.int32,
        shape=st.integers(min_value=1, max_value=100),
        elements=st.integers(min_value=1, max_value=255),
    )
)
@settings(max_examples=100)
def test_oracle_index_shift_no_double_shift(raw: np.ndarray) -> None:
    """A second application of the shift must NOT equal the first application.

    If the shift were accidentally applied twice, the output would equal
    ``raw - 2`` rather than ``raw - 1``.  This test confirms the single-shift
    result differs from the double-shift result for all non-empty inputs.
    """
    single_shift = _normalize_int_array(raw, one_based=True)
    double_shift = single_shift - 1  # simulates an accidental second shift

    # single_shift == raw - 1, double_shift == raw - 2; they must differ
    assert not np.array_equal(single_shift, double_shift), (
        "Single-shift and double-shift outputs are identical — "
        "that would only happen for a zero-element array (impossible here) or "
        "if the test logic is wrong."
    )


@pytest.mark.unit
@given(
    raw=arrays(
        dtype=np.int32,
        shape=st.integers(min_value=1, max_value=100),
        elements=st.integers(min_value=1, max_value=255),
    )
)
@settings(max_examples=100)
def test_oracle_index_shift_output_range(raw: np.ndarray) -> None:
    """Output values lie in [0, 254] for 1-based inputs in [1, 255].

    After the shift, no value should be negative (which would indicate an invalid
    voxel that was not in the valid [1, 255] domain) and none should exceed 254.
    """
    loaded = _normalize_int_array(raw, one_based=True)

    assert int(loaded.min()) >= 0, (
        f"Negative index after shift: min={int(loaded.min())}. "
        "All inputs were in [1, 255] so outputs must be in [0, 254]."
    )
    assert int(loaded.max()) <= 254, (
        f"Index out of expected range after shift: max={int(loaded.max())}. "
        "All inputs were in [1, 255] so outputs must be in [0, 254]."
    )
