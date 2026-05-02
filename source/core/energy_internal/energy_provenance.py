"""Preferred internal name for energy provenance helpers."""

from __future__ import annotations

from .._energy.provenance import (
    energy_origin_for_method,
    is_exact_compatible_energy_origin,
)

__all__ = [
    "energy_origin_for_method",
    "is_exact_compatible_energy_origin",
]
