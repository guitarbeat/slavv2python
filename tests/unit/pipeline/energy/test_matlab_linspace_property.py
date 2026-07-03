"""Hypothesis-driven property tests for the MATLAB-equivalent linspace mesh.

# Feature: matlab-python-parity, Property 6: MATLAB Linspace Mesh Correctness

Validates: Requirements 4.4

Property 6: For any valid (count, stride) parameters, the MATLAB-equivalent linspace
implementation shall agree with MATLAB's expected mesh output to within 1e-14 absolute
error at every grid point, including coarse-cell boundaries where the integer-mod
formula avoids the ~1 ULP drift that np.linspace accumulates.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from slavv_python.pipeline.energy.matlab_get_energy_v202_chunked import (
    _matlab_zero_based_linspace_raw,
)

# ── tolerance ──────────────────────────────────────────────────────────────────
_ABS_TOL = 1e-14  # per Property 6 spec


# ── helpers ────────────────────────────────────────────────────────────────────

def _reference_mesh_matlab_linspace(count: int, stride: int) -> np.ndarray:
    """Reference mesh matching MATLAB's ``linspace(1, count/stride, count) - 1``.

    This is the independent re-implementation of the MATLAB linspace formula
    used as a cross-check.  MATLAB's linspace is:
        d1 + (0:n-1) .* (d2-d1) / (n-1)   with endpoints forced.

    For offset=0 the function-under-test uses d1=1.0 and d2=1 + (count-1)/stride,
    then subtracts 1.  So the reference is:
        np.linspace(1.0, 1.0 + (count-1)/stride, count) - 1.0
    which matches the MATLAB call ``linspace(1, count/stride, count) - 1``.
    """
    d1 = 1.0
    d2 = 1.0 + float(count - 1) / float(stride)
    ref = np.linspace(d1, d2, count, dtype=np.float64) - 1.0
    # Force endpoints to match what the raw formula does
    ref[0] = 0.0
    ref[-1] = float(count - 1) / float(stride)
    return ref


def _integer_landing_indices(count: int, stride: int) -> np.ndarray:
    """Return the indices i where i is an exact multiple of stride (0 <= i < count)."""
    return np.arange(0, count, stride, dtype=np.intp)


# ── strategies ─────────────────────────────────────────────────────────────────

# Strides drawn from realistic pipeline values (1–20, matching chunk strides).
_stride_st = st.integers(min_value=1, max_value=20)
# count: ≥2 for a non-trivial linspace; ≤200 to keep tests fast.
_count_st = st.integers(min_value=2, max_value=200)


# ── property tests ─────────────────────────────────────────────────────────────


@pytest.mark.unit
@pytest.mark.parity
@given(count=_count_st, stride=_stride_st)
@settings(max_examples=100)
def test_matlab_linspace_agrees_with_reference_at_every_point(
    count: int, stride: int
) -> None:
    """Property 6: max absolute error < 1e-14 versus the independent MATLAB formula re-implementation.

    The reference is np.linspace(1, 1+(count-1)/stride, count) - 1, which is the
    same algorithm MATLAB uses.  The function-under-test IS that formula; this test
    cross-checks the raw implementation against an independent call to np.linspace
    using matching d1/d2 parameters, confirming both agree to within 1e-14.

    Note: non-boundary points may differ from arange/stride by ~1-2 ULP due to
    floating-point accumulation, which is expected and acceptable (the test here
    is against MATLAB's formula, not against the mathematical value).

    Validates: Requirements 4.4
    """
    mesh = _matlab_zero_based_linspace_raw(
        offset=0, stride=stride, count=count, local_start=0
    )
    ref = _reference_mesh_matlab_linspace(count, stride)

    assert mesh.shape == (count,), (
        f"Unexpected mesh length {mesh.shape} for count={count}"
    )
    assert mesh.dtype == np.float64, "Mesh must be float64"

    max_delta = float(np.max(np.abs(mesh - ref)))
    assert max_delta < _ABS_TOL, (
        f"Max absolute error {max_delta:.2e} >= 1e-14 for count={count}, stride={stride}. "
        f"Worst index: {int(np.argmax(np.abs(mesh - ref)))}"
    )


@pytest.mark.unit
@pytest.mark.parity
@given(count=_count_st, stride=_stride_st)
@settings(max_examples=100)
def test_matlab_linspace_integer_landings_are_exact(
    count: int, stride: int
) -> None:
    """Property 6 (coarse-cell boundary): mesh[k*stride] == float(k) to within 1e-14.

    At every index i that is an exact multiple of stride, the mesh value should
    land on the integer k = i // stride.  np.linspace can drift by ~1 ULP here;
    the MATLAB integer-mod formula avoids that drift.

    Validates: Requirements 4.4
    """
    mesh = _matlab_zero_based_linspace_raw(
        offset=0, stride=stride, count=count, local_start=0
    )

    idx = _integer_landing_indices(count, stride)
    expected_integers = (idx // stride).astype(np.float64)
    actual_at_boundaries = mesh[idx]

    max_delta = float(np.max(np.abs(actual_at_boundaries - expected_integers)))
    assert max_delta < _ABS_TOL, (
        f"Boundary delta {max_delta:.2e} >= 1e-14 at integer landings "
        f"for count={count}, stride={stride}; "
        f"worst index={idx[np.argmax(np.abs(actual_at_boundaries - expected_integers))]}"
    )


@pytest.mark.unit
@pytest.mark.parity
@given(
    count=_count_st,
    stride=_stride_st,
    offset=st.integers(min_value=0, max_value=200),
)
@settings(max_examples=100)
def test_matlab_linspace_nonzero_offset_agrees_with_reference(
    count: int, stride: int, offset: int
) -> None:
    """Property 6 (non-zero offset): the phase shift from offset is applied correctly.

    With offset > 0, the mesh starts at (offset % stride) / stride instead of 0,
    corresponding to a sub-cell fractional phase.  The reference for a given offset
    is (offset + np.arange(count)) / stride minus the local_start floor.

    Validates: Requirements 4.4
    """
    local_start = offset // stride
    mesh = _matlab_zero_based_linspace_raw(
        offset=offset, stride=stride, count=count, local_start=local_start
    )

    # Reference: the i-th fine-grid point maps to fine-global index (offset + i),
    # which in coarse coordinates is (offset + i) / stride, re-based to local_start.
    ref = (float(offset) + np.arange(count, dtype=np.float64)) / float(stride) - float(local_start)

    assert mesh.shape == (count,), (
        f"Unexpected mesh length {mesh.shape}"
    )
    assert mesh.dtype == np.float64

    max_delta = float(np.max(np.abs(mesh - ref)))
    assert max_delta < _ABS_TOL, (
        f"Max delta {max_delta:.2e} >= 1e-14 for "
        f"offset={offset}, stride={stride}, count={count}, local_start={local_start}"
    )


@pytest.mark.unit
@pytest.mark.parity
@given(count=_count_st, stride=_stride_st)
@settings(max_examples=100)
def test_matlab_linspace_endpoints_are_forced(
    count: int, stride: int
) -> None:
    """Property 6 (endpoint forcing): first and last points match expected values exactly.

    MATLAB's linspace forces endpoints; the port must too.  This guards against
    the accumulated-arithmetic form which can have a rounding error at the final point.

    Validates: Requirements 4.4
    """
    mesh = _matlab_zero_based_linspace_raw(
        offset=0, stride=stride, count=count, local_start=0
    )

    expected_start = 0.0
    expected_end = float(count - 1) / float(stride)

    # Endpoints must be within 1e-14 (forced endpoint means they should be exact,
    # but we use the tolerance to guard against any future formula adjustment).
    assert abs(mesh[0] - expected_start) < _ABS_TOL, (
        f"Start mismatch: {mesh[0]} vs {expected_start} for count={count}, stride={stride}"
    )
    assert abs(mesh[-1] - expected_end) < _ABS_TOL, (
        f"End mismatch: {mesh[-1]} vs {expected_end} for count={count}, stride={stride}"
    )
