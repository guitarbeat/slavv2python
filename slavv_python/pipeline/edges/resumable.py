"""Resumable watershed edge extraction."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import scipy.ndimage as ndi

from slavv_python.pipeline.edges.candidate_manifest import _append_candidate_unit
from slavv_python.pipeline.edges.naive_watershed import (
    collect_naive_watershed_label_unit,
    paint_vertex_watershed_markers,
    run_skimage_watershed_labels,
)
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
    markers = paint_vertex_watershed_markers(vertex_positions, energy.shape)

    logger.info("Running watershed on volume (this may take several minutes)...")
    labels = run_skimage_watershed_labels(energy, markers, energy_sign=energy_sign)
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
        unit = collect_naive_watershed_label_unit(
            label,
            labels,
            energy,
            structure,
            seen_pairs,
            coord_dtype=np.float64,
        )
        payload = unit.to_unit_payload()
        atomic_joblib_dump(payload, units_dir / f"label_{origin_index:06d}.pkl")
        edges.extend(unit.traces)
        connections.extend(unit.connections)
        edge_energies.extend(unit.metrics)
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
        energies=np.asarray(edge_energies, dtype=np.float64),
        vertex_positions=vertex_positions.astype(np.float64),
    )


__all__ = ["extract_edges_watershed_resumable"]
