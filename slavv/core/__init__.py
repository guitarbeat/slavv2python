"""
Core processing pipeline for SLAVV.

This subpackage contains the main processing modules:
- energy: Multi-scale energy field calculation
- tracing: Vertex extraction and edge tracing
- graph: Network construction from traces
- pipeline: Orchestration of the complete workflow
"""
from .pipeline import SLAVVProcessor
from .energy import (
    calculate_energy_field,
    spherical_structuring_element,
    compute_gradient_impl,
    compute_gradient_fast,
)
from .tracing import (
    extract_vertices,
    extract_edges,
    extract_edges_watershed,
    trace_edge,
    near_vertex,
    find_terminal_vertex,
    generate_edge_directions,
    estimate_vessel_directions,
    compute_gradient,
    in_bounds,
)
from .graph import (
    construct_network,
    trace_strand_sparse,
    sort_and_validate_strands_sparse,
)

__all__ = [
    "SLAVVProcessor",
    "calculate_energy_field",
    "spherical_structuring_element",
    "compute_gradient_impl",
    "compute_gradient_fast",
    "extract_vertices",
    "extract_edges",
    "extract_edges_watershed",
    "trace_edge",
    "near_vertex",
    "find_terminal_vertex",
    "generate_edge_directions",
    "estimate_vessel_directions",
    "compute_gradient",
    "in_bounds",
    "construct_network",
    "trace_strand_sparse",
    "sort_and_validate_strands_sparse",
]

