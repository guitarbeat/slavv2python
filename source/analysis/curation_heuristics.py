from __future__ import annotations

from typing import Any

import numpy as np

try:
    from ..utils import calculate_path_length
except ImportError:  # pragma: no cover - fallback for direct execution
    from source.utils import calculate_path_length


def choose_vertices(
    vertices: dict[str, Any],
    min_energy: float = 0.0,
    min_radius: float = 0.0,
    energy_sign: float = -1.0,
) -> np.ndarray:
    energies = vertices["energies"] * energy_sign
    radii = vertices.get("radii_microns", vertices.get("radii_pixels"))
    radii = np.asarray(radii, dtype=float)
    mask = (energies >= min_energy) & (radii >= min_radius)
    return np.flatnonzero(mask)


def choose_edges(
    edges: dict[str, Any],
    min_energy: float = 0.0,
    min_length: float = 0.0,
    energy_sign: float = -1.0,
) -> np.ndarray:
    energies = edges["energies"] * energy_sign
    lengths = np.array([calculate_path_length(trace) for trace in edges["traces"]])
    mask = (energies >= min_energy) & (lengths >= min_length)
    return np.flatnonzero(mask)


def extract_uncurated_info(
    vertices: dict[str, Any],
    edges: dict[str, Any],
    energy_data: dict[str, Any],
    image_shape: tuple[int, ...],
) -> dict[str, np.ndarray]:
    try:
        from .ml_curator import MLCurator
    except ImportError:  # pragma: no cover - fallback for direct execution
        from source.analysis.ml_curator import MLCurator

    curator = MLCurator()
    vertex_features = curator.extract_vertex_features(vertices, energy_data, image_shape)
    edge_features = curator.extract_edge_features(edges, vertices, energy_data)
    return {"vertex_features": vertex_features, "edge_features": edge_features}
