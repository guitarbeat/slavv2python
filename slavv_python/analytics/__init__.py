from __future__ import annotations

from .curation.automated import AutomaticCurator
from .curation.machine_learning import MLCurator, DrewsCurator
from .metrics.topology import calculate_network_statistics, calculate_surface_area, calculate_vessel_volume
from .math import calculate_path_length
from .registration import evaluate_registration, register_vector_sets
from .cropping import crop_edges, crop_vertices, crop_vertices_by_mask
from .trace_ops import get_edge_metric, get_edges_for_vertex

def choose_edges(edges, vertices, parameters):
    return AutomaticCurator().curate_edges_automatic(edges, vertices, parameters)

def choose_vertices(vertices, energy_data, parameters):
    return AutomaticCurator().curate_vertices_automatic(vertices, energy_data, parameters)

__all__ = [
    "AutomaticCurator",
    "MLCurator",
    "DrewsCurator",
    "calculate_network_statistics",
    "calculate_surface_area",
    "calculate_vessel_volume",
    "calculate_path_length",
    "evaluate_registration",
    "register_vector_sets",
    "crop_edges",
    "crop_vertices",
    "crop_vertices_by_mask",
    "get_edge_metric",
    "get_edges_for_vertex",
    "choose_edges",
    "choose_vertices",
]
