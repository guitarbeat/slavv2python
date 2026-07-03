"""Property-based tests for round-half-away-from-zero rounding.

# Feature: matlab-python-parity, Property 5: Round-Half-Away-from-Zero

Verifies that slavv_round() matches MATLAB's round() at every .5-boundary:
- Positive x with x - floor(x) == 0.5: result == floor(x) + 1  (rounds up)
- Negative x with x - floor(x) == 0.5: result == floor(x)       (rounds more negative, i.e. away from zero)

Key examples:
    slavv_round(2.5)  == 3   (floor(2.5) + 1 == 2 + 1 == 3)
    slavv_round(-2.5) == -3  (floor(-2.5)    == -3)

MATLAB round(-2.5) == -3 because MATLAB rounds half away from zero, meaning
-2.5 rounds to -3 (more negative), not -2.  The formula floor(x) applies for
negative .5-boundary values: floor(-2.5) == -3 == ceil(-2.5 - 0.5).

Validates: Requirements 3.4
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from slavv_python.utils.math_utils import slavv_round

# ---------------------------------------------------------------------------
# Strategy: floats at positive .5 boundaries
# ---------------------------------------------------------------------------

# Generate n such that x = n + 0.5 where n >= 0 (positive .5-boundary values).
# We restrict the integer part to a range where float representation is exact.
_positive_half = st.integers(min_value=0, max_value=2**52 - 1).map(lambda n: float(n) + 0.5)

# Generate n such that x = -(n + 0.5) where n >= 0 (negative .5-boundary values).
_negative_half = st.integers(min_value=0, max_value=2**52 - 1).map(lambda n: -(float(n) + 0.5))


# ---------------------------------------------------------------------------
# Property 5a: Positive .5-boundary — round half up (away from zero == toward +inf)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@given(x=_positive_half)
@settings(max_examples=100)
def test_round_half_away_from_zero_positive(x: float) -> None:
    """For x >= 0 with x - floor(x) == 0.5, slavv_round(x) == floor(x) + 1.

    MATLAB round(2.5) == 3, round(0.5) == 1, round(100.5) == 101.
    Round-half-away-from-zero for positive values means rounding up.

    **Validates: Requirements 3.4**
    """
    assert x - math.floor(x) == 0.5, f"Generator invariant failed: {x}"
    expected = math.floor(x) + 1
    result = slavv_round(x)
    assert result == expected, f"slavv_round({x}) == {result}, expected floor({x})+1 == {expected}"


# ---------------------------------------------------------------------------
# Property 5b: Negative .5-boundary — round half down (away from zero == toward -inf)
# ---------------------------------------------------------------------------


@pytest.mark.unit
@given(x=_negative_half)
@settings(max_examples=100)
def test_round_half_away_from_zero_negative(x: float) -> None:
    """For x < 0 with x - floor(x) == 0.5, slavv_round(x) == floor(x).

    MATLAB round(-2.5) == -3, round(-0.5) == -1, round(-100.5) == -101.
    Round-half-away-from-zero for negative values means rounding more negative.

    floor(-2.5) == -3, which is away from zero for negative inputs.
    Equivalently: ceil(x - 0.5) == floor(x) at .5-boundary negative values.

    **Validates: Requirements 3.4**
    """
    assert x - math.floor(x) == 0.5, f"Generator invariant failed: {x}"
    expected = math.floor(x)
    result = slavv_round(x)
    assert result == expected, f"slavv_round({x}) == {result}, expected floor({x}) == {expected}"


# ---------------------------------------------------------------------------
# Spot-check: canonical MATLAB examples from the docstring
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_round_half_canonical_examples() -> None:
    """Canonical MATLAB round() examples that define correct half-away-from-zero."""
    assert slavv_round(2.5) == 3
    assert slavv_round(-2.5) == -3
    assert slavv_round(0.5) == 1
    assert slavv_round(-0.5) == -1
    assert slavv_round(1.5) == 2
    assert slavv_round(-1.5) == -2
    assert slavv_round(100.5) == 101
    assert slavv_round(-100.5) == -101
