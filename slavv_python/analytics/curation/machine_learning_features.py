from __future__ import annotations

from typing import Any, cast

import numpy as np


def compute_local_gradient(energy_field: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Calculate the energy gradient at a specific position using central difference."""
    pos_int = np.round(pos).astype(int)
    y, x, z = pos_int
    shape = energy_field.shape

    # Central difference in each dimension
    def _diff(axis, val):
        coords_plus = list(pos_int)
        coords_minus = list(pos_int)
        coords_plus[axis] = min(val + 1, shape[axis] - 1)
        coords_minus[axis] = max(val - 1, 0)
        return (energy_field[tuple(coords_plus)] - energy_field[tuple(coords_minus)]) / 2.0

    return cast("np.ndarray", np.array([_diff(0, y), _diff(1, x), _diff(2, z)]))


def feature_importance(model: Any) -> np.ndarray | None:
    """Extract feature importances from a supported classifier model."""
    if hasattr(model, "feature_importances_"):
        return cast("np.ndarray", model.feature_importances_)
    return None


def in_bounds(pos: np.ndarray, shape: tuple[int, ...]) -> bool:
    """Check if a coordinate is within the boundaries of a given shape."""
    p = np.asarray(pos)
    # Ensure it returns a plain bool for 'is True' checks in tests
    result = all(0 <= p[i] < shape[i] for i in range(len(shape)))
    return bool(result)
