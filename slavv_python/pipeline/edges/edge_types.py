"""Shared numpy type aliases for edge-stage modules."""

from __future__ import annotations

from typing_extensions import TypeAlias

Int16Array: TypeAlias = "np.ndarray"
Int32Array: TypeAlias = "np.ndarray"
Int64Array: TypeAlias = "np.ndarray"
Float32Array: TypeAlias = "np.ndarray"  # DEPRECATED: use Float64Array for exact route
Float64Array: TypeAlias = "np.ndarray"
BoolArray: TypeAlias = "np.ndarray"
