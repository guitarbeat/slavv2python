"""Resumable watershed edge extraction."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.ndimage as ndi
from skimage.segmentation import watershed

from slavv_python.pipeline.edges.candidate_manifest import _append_candidate_unit
from slavv_python.pipeline.edges.payloads import _empty_edge_diagnostics
from slavv_python.pipeline.edges.units import _load_edge_units
from slavv_python.schema.results import EdgeSet, EnergyResult, VertexSet

if TYPE_CHECKING:
    from slavv_python.engine.state import StageController

logger = logging.getLogger(__name__)


def extract_edges_watershed_resumable(
    energy_data: EnergyResult,
    vertices: VertexSet,
    params: dict[str, Any],
    stage_controller: StageController,
) -> EdgeSet:
    """Extract watershed edges with per-label persisted units."""
    from slavv_python.engine.state.tracker import atomic_joblib_dump

    del params
    energy = energy_data.energy
    energy_sign = float(energy_data.extra.get("energy_sign", -1.0))
    vertex_positions = vertices.positions
    markers = np.zeros_like(energy, dtype=np.int32)
    idxs = np.floor(vertex_positions).astype(int)
    idxs = np.clip(idxs, 0, np.array(energy.shape) - 1)
    markers[idxs[:, 0], idxs[:, 1], idxs[:, 2]] = np.arange(1, len(vertex_positions) + 1)

    logger.info("Running watershed on volume (this may take several minutes)...")
    labels = watershed(-energy_sign * energy, markers)
    logger.info("Watershed complete, extracting edges between regions...")
    structure = ndi.generate_binary_structure(3, 1)

    units_dir = stage_controller.artifact_path("units")
    units_dir.mkdir(parents=True, exist_ok=True)
    existing_payload, completed = _load_edge_units(
        units_dir, _append_candidate_unit, _empty_edge_diagnostics
    )
    edges = existing_payload["traces"]
    connections = (
        existing_payload["connections"].tolist() if existing_payload["connections"].size else []
    )
    edge_energies = existing_payload["metrics"].tolist()
    seen_pairs = {
        tuple(sorted((int(start), int(end))))
        for start, end in np.asarray(existing_payload["connections"], dtype=np.int32).reshape(-1, 2)
        if int(start) >= 0 and int(end) >= 0
    }
    stage_controller.begin(
        detail="Tracing watershed label adjacencies",
        units_total=len(vertex_positions),
        units_completed=len(completed),
        substage="watershed_labels",
        resumed=bool(completed),
    )

    for label in range(1, len(vertex_positions) + 1):
        origin_index = label - 1
        if origin_index in completed:
            continue
        region = labels == label
        dilated = ndi.binary_dilation(region, structure)
        neighbors = np.unique(labels[dilated & (labels != label)])
        unit_traces: list[np.ndarray] = []
        unit_connections: list[list[int]] = []
        unit_energies: list[float] = []
        for neighbor in neighbors:
            if neighbor <= label or neighbor == 0:
                continue
            pair = (label - 1, neighbor - 1)
            if pair in seen_pairs:
                continue
            boundary = (ndi.binary_dilation(labels == neighbor, structure) & region) | (
                ndi.binary_dilation(region, structure) & (labels == neighbor)
            )
            coords = np.argwhere(boundary)
            if coords.size == 0:
                continue
            coords = coords.astype(np.float32)
            idx = np.floor(coords).astype(int)
            energies = energy[idx[:, 0], idx[:, 1], idx[:, 2]]
            unit_traces.append(coords)
            unit_connections.append([label - 1, neighbor - 1])
            unit_energies.append(float(np.mean(energies)))
            seen_pairs.add(pair)

        payload = {
            "origin_index": origin_index,
            "candidate_source": "fallback",
            "traces": unit_traces,
            "connections": unit_connections,
            "metrics": unit_energies,
            "energy_traces": [
                np.asarray([energy_value], dtype=np.float32) for energy_value in unit_energies
            ],
            "scale_traces": [np.zeros((len(trace),), dtype=np.int16) for trace in unit_traces],
            "origin_indices": [origin_index] * len(unit_traces),
            "connection_sources": ["fallback"] * len(unit_traces),
        }
        atomic_joblib_dump(payload, units_dir / f"label_{origin_index:06d}.pkl")
        edges.extend(unit_traces)
        connections.extend(unit_connections)
        edge_energies.extend(unit_energies)
        completed.add(origin_index)
        stage_controller.save_state({"last_completed_label": origin_index})
        stage_controller.update(
            units_total=len(vertex_positions),
            units_completed=len(completed),
            substage="watershed_labels",
            detail=f"Watershed label {label}/{len(vertex_positions)}",
            resumed=bool(completed - {origin_index}),
        )

    return EdgeSet.create(
        traces=edges,
        connections=np.asarray(connections, dtype=np.int32).reshape(-1, 2),
        energies=np.asarray(edge_energies, dtype=np.float32),
        vertex_positions=vertex_positions.astype(np.float32),
    )


__all__ = ["extract_edges_watershed_resumable"]
