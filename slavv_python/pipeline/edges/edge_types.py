"""Shared numpy type aliases for edge-stage modules."""

from __future__ import annotations

from typing import Any

import numpy as np
from typing_extensions import TypeAlias

Int16Array: TypeAlias = np.ndarray[Any, Any]
Int32Array: TypeAlias = np.ndarray[Any, Any]
Int64Array: TypeAlias = np.ndarray[Any, Any]
Float32Array: TypeAlias = np.ndarray[Any, Any]  # DEPRECATED: use Float64Array for exact route
Float64Array: TypeAlias = np.ndarray[Any, Any]
BoolArray: TypeAlias = np.ndarray[Any, Any]
