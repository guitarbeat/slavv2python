"""Watershed-based edge extraction paths."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import scipy.ndimage as ndi
from skimage.segmentation import watershed

logger = logging.getLogger(__name__)


def extract_edges_watershed(
    energy_data: dict[str, Any], vertices: dict[str, Any], params: dict[str, Any]
) -> dict[str, Any]:
    """Extract edges using watershed segmentation seeded at vertices."""
    logger.info("Extracting edges via watershed")

    energy = energy_data["energy"]
    energy_sign = float(energy_data.get("energy_sign", -1.0))
    vertex_positions = vertices["positions"]

    markers = np.zeros_like(energy, dtype=np.int32)
    idxs = np.floor(vertex_positions).astype(int)
    idxs = np.clip(idxs, 0, np.array(energy.shape) - 1)
    markers[idxs[:, 0], idxs[:, 1], idxs[:, 2]] = np.arange(1, len(vertex_positions) + 1)

    logger.info("Running watershed on volume (this may take several minutes)...")
    labels = watershed(-energy_sign * energy, markers)
    logger.info("Watershed complete, extracting edges between regions...")
    structure = ndi.generate_binary_structure(3, 1)

    edges: list[np.ndarray] = []
    connections: list[list[int]] = []
    edge_energies: list[float] = []
    seen = set()
    n_vertices = len(vertex_positions)
    log_interval = max(1, n_vertices // 20)

    for label in range(1, n_vertices + 1):
        if label % log_interval == 0 or label == n_vertices:
            logger.info(
                "Watershed progress: vertex %d / %d, edges so far: %d",
                label,
                n_vertices,
                len(edges),
            )
        region = labels == label
        dilated = ndi.binary_dilation(region, structure)
        neighbors = np.unique(labels[dilated & (labels != label)])
        for neighbor in neighbors:
            if neighbor <= label or neighbor == 0:
                continue
            pair = (label - 1, neighbor - 1)
            if pair in seen:
                continue
            boundary = (ndi.binary_dilation(labels == neighbor, structure) & region) | (
                ndi.binary_dilation(region, structure) & (labels == neighbor)
            )
            coords = np.argwhere(boundary)
            if coords.size == 0:
                continue
            coords = coords.astype(np.float32)
            edges.append(coords)
            idx = np.floor(coords).astype(int)
            energies = energy[idx[:, 0], idx[:, 1], idx[:, 2]]
            edge_energies.append(float(np.mean(energies)))
            connections.append([label - 1, neighbor - 1])
            seen.add(pair)

    logger.info("Extracted %d watershed edges", len(edges))

    return {
        "traces": edges,
        "connections": np.asarray(connections, dtype=np.int32).reshape(-1, 2),
        "energies": np.asarray(edge_energies, dtype=np.float32),
        "vertex_positions": vertex_positions.astype(np.float32),
    }
