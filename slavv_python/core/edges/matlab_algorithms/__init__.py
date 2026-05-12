"""MATLAB-shaped orchestration surface for parity audits and proof routing."""

from __future__ import annotations

from .stages import (
    add_vertices_to_edges,
    choose_edges_v200,
    get_edge_metric,
    get_edges_v300,
    get_energy_v202,
    get_network_v190,
    get_strand_objects,
    get_vertices_v200,
    sort_network_v180,
)

from .vectorize_v200 import vectorize_v200

__all__ = [
    "add_vertices_to_edges",
    "choose_edges_v200",
    "get_edge_metric",
    "get_edges_v300",
    "get_energy_v202",
    "get_network_v190",
    "get_strand_objects",
    "get_vertices_v200",
    "sort_network_v180",
    "vectorize_v200",
]
