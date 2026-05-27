from __future__ import annotations

from typing import Any

import numpy as np

from .cropping import crop_edges, crop_vertices, crop_vertices_by_mask
from .curation.automated import AutomaticCurator
from .curation.machine_learning import DrewsCurator, MLCurator, extract_uncurated_info
from .math import calculate_path_length, resample_vectors, smooth_edge_traces
from .metrics.topology import (
    calculate_network_statistics,
    calculate_surface_area,
    calculate_vessel_volume,
)
from .registration import evaluate_registration, register_vector_sets, transform_vector_set
from .trace_ops import get_edge_metric, get_edges_for_vertex


def _ensure_dict(obj, default_keys, primary_key):
    # 1. Start with a clean slate of defaults
    d: dict[str, Any] = {}
    length = 0

    # 2. Extract and normalize the primary source
    if isinstance(obj, np.ndarray):
        arr = np.atleast_2d(obj)
        if primary_key == "connections" and arr.shape[1] > 2:
            arr = arr.reshape(-1, 2)
        elif primary_key == "positions" and arr.shape[1] != 3 and arr.size > 0:
            arr = arr.reshape(-1, 3)
        d[primary_key] = arr
        length = len(arr)
    elif isinstance(obj, dict):
        d = dict(obj)
        # Find length from any array or list in the input dictionary
        lengths = [
            len(v) for v in d.values() if hasattr(v, "__len__") and not isinstance(v, (str, bytes))
        ]
        length = max(lengths) if lengths else 0

        # Normalize 2D arrays if they exist in input
        if "positions" in d:
            d["positions"] = np.atleast_2d(d["positions"])
            if d["positions"].shape[1] != 3 and d["positions"].size > 0:
                d["positions"] = d["positions"].reshape(-1, 3)
        if "connections" in d:
            d["connections"] = np.atleast_2d(d["connections"])
            if d["connections"].shape[1] != 2 and d["connections"].size > 0:
                d["connections"] = d["connections"].reshape(-1, 2)
        if "vertex_positions" in d:
            d["vertex_positions"] = np.atleast_2d(d["vertex_positions"])
            if d["vertex_positions"].shape[1] != 3 and d["vertex_positions"].size > 0:
                d["vertex_positions"] = d["vertex_positions"].reshape(-1, 3)
    elif obj is None:
        length = 0
    else:
        raise TypeError(f"Unsupported input type for shimming: {type(obj)}")

    # 3. Fill in ALL missing keys from defaults with correct length & dimensionality
    for k, v in default_keys.items():
        if k not in d:
            if k == "traces":
                d[k] = [np.zeros((0, 3))] * length
            elif isinstance(v, np.ndarray):
                shape = (length, *v.shape[1:])
                d[k] = np.zeros(shape, dtype=v.dtype)
            else:
                d[k] = np.zeros(length, dtype=getattr(v, "dtype", type(v)))
        else:
            # Ensure existing parallel arrays are also 1D and match length
            if k in ["energies", "scales", "radii_microns", "radii_pixels"]:
                d[k] = np.asarray(d[k]).reshape(length)

    return d


def choose_edges(edges, vertices=None, parameters=None, **kwargs):
    if parameters is None:
        parameters = {}
    p = {**parameters, **kwargs}

    if "min_energy" in p:
        sign = p.get("energy_sign", -1.0)
        p["edge_energy_threshold"] = sign * p.pop("min_energy")
    if "min_length" in p:
        p["min_edge_length"] = p.pop("min_length")

    p.setdefault("boundary_margin", 0)

    edges_dict = _ensure_dict(
        edges,
        {
            "connections": np.zeros((0, 2), dtype=int),
            "traces": [],
            "energies": np.zeros(0, dtype=float),
            "vertex_positions": np.zeros((0, 3), dtype=float),
        },
        "connections",
    )

    vertices_dict = _ensure_dict(
        vertices,
        {
            "positions": np.zeros((0, 3), dtype=float),
            "original_indices": np.array([], dtype=int),
        },
        "positions",
    )

    return AutomaticCurator().curate_edges_automatic(edges_dict, vertices_dict, p)[
        "original_indices"
    ]


def choose_vertices(vertices, energy_data=None, parameters=None, **kwargs):
    if parameters is None:
        parameters = {}
    p = {**parameters, **kwargs}

    if "min_energy" in p:
        sign = p.get("energy_sign", -1.0)
        p["vertex_energy_threshold"] = sign * p.pop("min_energy")
    if "min_radius" in p:
        p["min_vertex_radius"] = p.pop("min_radius")

    p.setdefault("boundary_margin", 0)
    p.setdefault("contrast_threshold", -np.inf)

    vertices_dict = _ensure_dict(
        vertices,
        {
            "positions": np.zeros((0, 3), dtype=float),
            "energies": np.zeros(0, dtype=float),
            "scales": np.zeros(0, dtype=int),
        },
        "positions",
    )

    if energy_data is None:
        energy_data = {"energy": np.zeros((1, 1, 1)), "image_shape": (1, 1, 1)}
    elif isinstance(energy_data, np.ndarray):
        energy_data = {"energy": energy_data, "image_shape": energy_data.shape}

    return AutomaticCurator().curate_vertices_automatic(vertices_dict, energy_data, p)[
        "original_indices"
    ]


__all__ = [
    "AutomaticCurator",
    "DrewsCurator",
    "MLCurator",
    "calculate_network_statistics",
    "calculate_path_length",
    "calculate_surface_area",
    "calculate_vessel_volume",
    "choose_edges",
    "choose_vertices",
    "crop_edges",
    "crop_vertices",
    "crop_vertices_by_mask",
    "evaluate_registration",
    "extract_uncurated_info",
    "get_edge_metric",
    "get_edges_for_vertex",
    "register_vector_sets",
    "resample_vectors",
    "smooth_edge_traces",
    "transform_vector_set",
]
