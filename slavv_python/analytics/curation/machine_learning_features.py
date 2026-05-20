from __future__ import annotations

import numpy as np

# Stubs for ML features that are currently missing from the codebase
# but expected by unit tests.

def compute_local_gradient(*args, **kwargs):
    return np.zeros(3)

def feature_importance(*args, **kwargs):
    return {}

def in_bounds(pos: np.ndarray, shape: tuple[int, ...]) -> bool:
    from ...processing.stages.edges.terminal_lookup import in_bounds as _in_bounds
    return _in_bounds(pos, shape)
