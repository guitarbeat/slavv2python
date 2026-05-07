from __future__ import annotations

import numpy as np
from slavv_python.analysis.ml_curator_features import (
    compute_local_gradient,
    feature_importance,
    in_bounds,
)


def test_compute_local_gradient_uses_central_difference():
    energy = np.zeros((5, 5, 5), dtype=float)
    energy[3, 2, 2] = 3.0
    energy[1, 2, 2] = 1.0
    energy[2, 3, 2] = 8.0
    energy[2, 1, 2] = 2.0
    energy[2, 2, 3] = 5.0
    energy[2, 2, 1] = 1.0

    gradient = compute_local_gradient(energy, np.array([2.0, 2.0, 2.0]))

    assert gradient.tolist() == [1.0, 3.0, 2.0]


def test_in_bounds_checks_all_dimensions():
    assert in_bounds(np.array([1, 2, 3]), (2, 3, 4)) is True
    assert in_bounds(np.array([2, 2, 3]), (2, 3, 4)) is False


def test_feature_importance_returns_none_when_missing():
    class _NoImportance:
        pass

    class _WithImportance:
        feature_importances_ = np.array([0.1, 0.9])

    assert feature_importance(_NoImportance()) is None
    assert feature_importance(_WithImportance()).tolist() == [0.1, 0.9]
