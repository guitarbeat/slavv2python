"""
MATLAB-compatible mathematical helper utilities.

This module provides rounding and numeric helpers that match MATLAB's semantics
rather than Python's built-in behaviours.
"""

from __future__ import annotations

import math

import numpy as np


def slavv_round(x: float | np.floating) -> int:
    """Round a scalar to the nearest integer using half-away-from-zero.

    MATLAB's ``round()`` rounds ``0.5`` away from zero:
    ``round(2.5) == 3``, ``round(-2.5) == -3``.

    Python's built-in ``round()`` uses banker's rounding (round-to-even):
    ``round(2.5) == 2``, ``round(3.5) == 4``.

    This function replicates MATLAB's behaviour via ``floor(x + 0.5)`` for
    non-negative values and ``ceil(x - 0.5)`` for negative values.  The
    canonical formula ``int(math.floor(x + 0.5))`` handles both cases
    correctly because ``floor(-2.5 + 0.5) = floor(-2.0) = -2`` would be
    wrong for ``-2.5`` — but the sign-aware form below matches MATLAB exactly.

    Args:
        x: A scalar float value to round.

    Returns:
        Nearest integer with half-away-from-zero tie-breaking.

    Examples:
        >>> slavv_round(2.5)
        3
        >>> slavv_round(-2.5)
        -3
        >>> slavv_round(2.4)
        2
        >>> slavv_round(-2.4)
        -2

    Requirements: 3.4
    """
    val = float(x)
    if val >= 0.0:
        return math.floor(val + 0.5)
    return math.ceil(val - 0.5)


def slavv_round_array(x: np.ndarray) -> np.ndarray:
    """Vectorised MATLAB-compatible round-half-away-from-zero.

    Equivalent to applying :func:`slavv_round` element-wise but implemented
    with NumPy operations for efficiency: ``floor(x + 0.5)`` for x >= 0 and
    ``ceil(x - 0.5)`` for x < 0.

    For the common parity-sensitive use case of rounding vertex coordinates
    this is the correct replacement for ``np.round``/``np.rint`` (which use
    banker's rounding) and for ``round()`` (same issue).

    Args:
        x: Array of float values to round.

    Returns:
        Integer-valued float array with half-away-from-zero tie-breaking,
        same shape as *x*, dtype ``float64``.

    Requirements: 3.4
    """
    arr = np.asarray(x, dtype=np.float64)
    result = np.where(arr >= 0.0, np.floor(arr + 0.5), np.ceil(arr - 0.5))
    return result


__all__ = [
    "slavv_round",
    "slavv_round_array",
]
