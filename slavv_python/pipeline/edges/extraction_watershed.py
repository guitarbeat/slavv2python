"""Watershed-based edge extraction paths."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import scipy.ndimage as ndi

from slavv_python.pipeline.edges.naive_watershed import (
    collect_naive_watershed_label_unit,
    paint_vertex_watershed_markers,
    run_skimage_watershed_labels,
)
from slavv_python.schema.results import EdgeSet, EnergyResult, VertexSet

logger = logging.getLogger(__name__)


def extract_edges_watershed(
    energy_data: EnergyResult, vertices: VertexSet, params: dict[str, Any]
) -> EdgeSet:
    """Extract edges using watershed segmentation seeded at vertices."""
    del params
    logger.info("Extracting edges via watershed")

    energy = energy_data.energy
    energy_sign = float(energy_data.extra.get("energy_sign", -1.0))
    vertex_positions = vertices.positions

    markers = paint_vertex_watershed_markers(vertex_positions, energy.shape)
    logger.info("Running watershed on volume (this may take several minutes)...")
    labels = run_skimage_watershed_labels(energy, markers, energy_sign=energy_sign)
    logger.info("Watershed complete, extracting edges between regions...")
    structure = ndi.generate_binary_structure(3, 1)

    edges: list[np.ndarray] = []
    connections: list[list[int]] = []
    edge_energies: list[float] = []
    seen_pairs: set[tuple[int, int]] = set()
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
        unit = collect_naive_watershed_label_unit(
            label,
            labels,
            energy,
            structure,
            seen_pairs,
            coord_dtype=np.float32,
        )
        edges.extend(unit.traces)
        connections.extend(unit.connections)
        edge_energies.extend(unit.metrics)

    logger.info("Extracted %d watershed edges", len(edges))

    return EdgeSet.create(
        traces=edges,
        connections=np.asarray(connections, dtype=np.int32).reshape(-1, 2),
        energies=np.asarray(edge_energies, dtype=np.float32),
        vertex_positions=vertex_positions.astype(np.float32),
    )
