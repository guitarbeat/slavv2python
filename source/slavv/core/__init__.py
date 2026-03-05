"""
Core processing pipeline for SLAVV.

This subpackage contains the main processing modules:
- energy: Multi-scale energy field calculation
- tracing: Vertex extraction and edge tracing
- graph: Network construction from traces
- pipeline: Orchestration of the complete workflow
"""

from __future__ import annotations

from .energy import (
    calculate_energy_field,
    compute_gradient_fast,
    compute_gradient_impl,
    spherical_structuring_element,
)
from .graph import (
    construct_network,
    sort_and_validate_strands_sparse,
    trace_strand_sparse,
)
from .pipeline import SLAVVProcessor
from .tracing import (
    compute_gradient,
    estimate_vessel_directions,
    extract_edges,
    extract_edges_watershed,
    extract_vertices,
    find_terminal_vertex,
    generate_edge_directions,
    in_bounds,
    near_vertex,
    trace_edge,
)

__all__ = [
    "SLAVVProcessor",
    "calculate_energy_field",
    "compute_gradient",
    "compute_gradient_fast",
    "compute_gradient_impl",
    "construct_network",
    "estimate_vessel_directions",
    "extract_edges",
    "extract_edges_watershed",
    "extract_vertices",
    "find_terminal_vertex",
    "generate_edge_directions",
    "in_bounds",
    "near_vertex",
    "sort_and_validate_strands_sparse",
    "spherical_structuring_element",
    "trace_edge",
    "trace_strand_sparse",
]
