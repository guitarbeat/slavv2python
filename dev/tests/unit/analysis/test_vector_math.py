from __future__ import annotations

import numpy as np
import pytest

from source.analysis._geometry import angle_degrees, safe_normalize_rows, scaled_positions


def test_safe_normalize_rows_handles_zero_rows_without_nans() -> None:
    vectors = np.array([[3.0, 0.0, 4.0], [0.0, 0.0, 0.0]], dtype=float)

    normalized = safe_normalize_rows(vectors)

    np.testing.assert_allclose(normalized[0], np.array([0.6, 0.0, 0.8], dtype=float))
    np.testing.assert_allclose(normalized[1], np.zeros(3, dtype=float))


def test_safe_normalize_rows_with_eps_clamps_tiny_norms() -> None:
    vectors = np.array([[1e-12, 0.0, 0.0]], dtype=float)

    normalized = safe_normalize_rows(vectors, eps=1e-6)

    np.testing.assert_allclose(normalized, np.array([[1e-6, 0.0, 0.0]], dtype=float))


def test_angle_degrees_basic_cases() -> None:
    assert np.isclose(angle_degrees(np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])), 90.0)
    assert np.isclose(angle_degrees(np.array([1.0, 0.0, 0.0]), np.array([-1.0, 0.0, 0.0])), 180.0)
    assert np.isclose(angle_degrees(np.array([0.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0])), 0.0)


def test_scaled_positions_applies_per_axis_scale() -> None:
    positions = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=float)
    scale = [0.5, 2.0, 3.0]

    scaled = scaled_positions(positions, scale)

    np.testing.assert_allclose(
        scaled,
        np.array([[0.5, 4.0, 9.0], [2.0, 10.0, 18.0]], dtype=float),
    )


def test_scaled_positions_rejects_mismatched_scale_dimension() -> None:
    with pytest.raises(ValueError, match="scale length must match"):
        scaled_positions(np.array([[1.0, 2.0, 3.0]], dtype=float), [1.0, 2.0])
