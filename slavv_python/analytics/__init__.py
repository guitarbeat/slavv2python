from __future__ import annotations

import numpy as np

from .curation.automated import AutomaticCurator
from .curation.machine_learning import MLCurator, DrewsCurator
from .metrics.topology import calculate_network_statistics, calculate_surface_area, calculate_vessel_volume
from .math import calculate_path_length, resample_vectors, smooth_edge_traces
from .registration import evaluate_registration, register_vector_sets, transform_vector_set
from .cropping import crop_edges, crop_vertices, crop_vertices_by_mask
from .trace_ops import get_edge_metric, get_edges_for_vertex

def choose_edges(edges, vertices=None, parameters=None, **kwargs):
    if parameters is None: parameters = {}
    p = {**parameters, **kwargs}
    
    # Handle thresholds from kwargs for legacy compatibility
    if "min_energy" in p: p["edge_energy_threshold"] = -p.pop("min_energy")
    if "min_length" in p: p["min_edge_length"] = p.pop("min_length")
    
    if isinstance(edges, np.ndarray):
        edges = {"connections": edges}
    
    if isinstance(edges, dict):
        # Ensure all required fields exist for AutomaticCurator as proper arrays
        n_edges = len(edges.get("connections", edges.get("traces", [])))
        edges.setdefault("connections", np.zeros((n_edges, 2), dtype=int))
        edges.setdefault("traces", [np.zeros((0, 3))] * n_edges)
        edges.setdefault("energies", np.zeros((n_edges,), dtype=float))
        edges.setdefault("vertex_positions", np.zeros((0, 3)))
        
    if vertices is None:
        vertices = {"positions": np.zeros((0, 3)), "original_indices": np.array([], dtype=int)}
    elif isinstance(vertices, np.ndarray):
        vertices = {"positions": vertices}
        
    if isinstance(vertices, dict):
        n_verts = len(vertices.get("positions", vertices.get("energies", [])))
        vertices.setdefault("positions", np.zeros((n_verts, 3)))
        vertices.setdefault("energies", np.zeros((n_verts,), dtype=float))
        vertices.setdefault("scales", np.zeros((n_verts,), dtype=int))
        vertices.setdefault("radii_microns", np.zeros((n_verts,), dtype=float))
        vertices.setdefault("original_indices", np.arange(n_verts))

    return AutomaticCurator().curate_edges_automatic(edges, vertices, p)["original_indices"]

def choose_vertices(vertices, energy_data=None, parameters=None, **kwargs):
    if parameters is None: parameters = {}
    p = {**parameters, **kwargs}
    
    # Handle thresholds from kwargs for legacy compatibility
    if "min_energy" in p: p["vertex_energy_threshold"] = -p.pop("min_energy")
    if "min_radius" in p: p["min_vertex_radius"] = p.pop("min_radius")
    
    if isinstance(vertices, np.ndarray):
        vertices = {"positions": vertices}
    
    if isinstance(vertices, dict):
        n_verts = len(vertices.get("positions", vertices.get("energies", [])))
        vertices.setdefault("positions", np.zeros((n_verts, 3)))
        vertices.setdefault("energies", np.zeros((n_verts,), dtype=float))
        vertices.setdefault("scales", np.zeros((n_verts,), dtype=int))
        vertices.setdefault("radii_pixels", np.zeros((n_verts,), dtype=float))
        
    if energy_data is None:
        energy_data = {"energy": np.zeros((1, 1, 1)), "image_shape": (1, 1, 1)}
    elif isinstance(energy_data, np.ndarray):
        energy_data = {"energy": energy_data, "image_shape": energy_data.shape}

    return AutomaticCurator().curate_vertices_automatic(vertices, energy_data, p)["original_indices"]

__all__ = [
    "AutomaticCurator",
    "MLCurator",
    "DrewsCurator",
    "calculate_network_statistics",
    "calculate_surface_area",
    "calculate_vessel_volume",
    "calculate_path_length",
    "resample_vectors",
    "smooth_edge_traces",
    "evaluate_registration",
    "register_vector_sets",
    "transform_vector_set",
    "crop_edges",
    "crop_vertices",
    "crop_vertices_by_mask",
    "get_edge_metric",
    "get_edges_for_vertex",
    "choose_edges",
    "choose_vertices",
]
