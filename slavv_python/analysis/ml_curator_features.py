"""Shared feature helpers for ML curator workflows."""

from __future__ import annotations

from typing import Any

import numpy as np


def compute_local_gradient(energy_field: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Compute a central-difference gradient at the given position."""
    pos_int = np.round(pos).astype(int)
    gradient = np.zeros(3)

    for i in range(3):
        if 0 < pos_int[i] < energy_field.shape[i] - 1:
            pos_plus = pos_int.copy()
            pos_minus = pos_int.copy()
            pos_plus[i] += 1
            pos_minus[i] -= 1
            gradient[i] = (energy_field[tuple(pos_plus)] - energy_field[tuple(pos_minus)]) / 2

    return gradient


def in_bounds(pos: np.ndarray, shape: tuple[int, ...]) -> bool:
    """Check if a position is within array bounds."""
    return all(0 <= p < s for p, s in zip(pos, shape))


def feature_importance(classifier: Any) -> np.ndarray | None:
    """Get feature importance from a classifier if available."""
    if hasattr(classifier, "feature_importances_"):
        return classifier.feature_importances_
    return None
