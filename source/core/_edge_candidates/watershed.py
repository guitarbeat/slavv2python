"""Compatibility surface for watershed-backed edge-candidate helpers."""

from __future__ import annotations

from .watershed_contacts import _augment_matlab_frontier_candidates_with_watershed_contacts
from .watershed_joins import _supplement_matlab_frontier_candidates_with_watershed_joins

__all__ = [
    "_augment_matlab_frontier_candidates_with_watershed_contacts",
    "_supplement_matlab_frontier_candidates_with_watershed_joins",
]
