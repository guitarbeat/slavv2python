"""
Comparison metrics for SLAVV validation.

This module preserves the public parity metrics surface while delegating the
implementations to focused helpers under ``slavv.parity._metrics``.
"""

from __future__ import annotations

from ._metrics.edges import compare_edges
from ._metrics.network import compare_networks
from ._metrics.results import compare_results
from ._metrics.shared_neighborhood import build_shared_neighborhood_audit
from ._metrics.vertices import compare_vertices, match_vertices

__all__ = [
    "build_shared_neighborhood_audit",
    "compare_edges",
    "compare_networks",
    "compare_results",
    "compare_vertices",
    "match_vertices",
]
