"""Resumable edge extraction paths."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, cast

import numpy as np
import scipy.ndimage as ndi
from scipy.spatial import cKDTree
from skimage.segmentation import watershed

from .edge_units import _load_edge_units

if TYPE_CHECKING:
    from source.runtime import StageController

logger = logging.getLogger(__name__)


def extract_edges_resumable(
    energy_data: dict[str, Any],
    vertices: dict[str, Any],
    params: dict[str, Any],
    stage_controller: StageController,
    *,
    atomic_joblib_dump: Callable[..., None],
    empty_edges_result: Callable[[np.ndarray], dict[str, Any]],
    build_edge_candidate_audit: Callable[..., dict[str, Any]],
    build_frontier_candidate_lifecycle: Callable[..., dict[str, Any]],
    finalize_matlab_parity_candidates: Callable[..., dict[str, Any]],
    normalize_candidate_origin_counts: Callable[..., dict[int, int]],
    generate_edge_candidates_matlab_frontier: Callable[..., dict[str, Any]],
    generate_edge_candidates: Callable[..., dict[str, Any]],
    choose_edges_for_workflow: Callable[..., dict[str, Any]],
    add_vertices_to_edges_matlab_style: Callable[..., dict[str, Any]],
    finalize_edges_matlab_style: Callable[..., dict[str, Any]],
    paint_vertex_center_image: Callable[[np.ndarray, tuple[int, ...]], np.ndarray],
    paint_vertex_image: Callable[[np.ndarray, np.ndarray, np.ndarray, tuple[int, ...]], np.ndarray],
    use_matlab_frontier_tracer: Callable[[dict[str, Any], dict[str, Any]], bool],
) -> dict[str, Any]:
    """Generate edge candidates through the maintained or MATLAB-parity workflow."""
    from source.io.matlab_fail_fast import build_candidate_snapshot_payload
    from source.runtime.run_state import atomic_write_json

    energy = energy_data["energy"]
    vertex_positions = vertices["positions"]
    vertex_scales = vertices["scales"]
    lumen_radius_microns = energy_data["lumen_radius_microns"]
    scale_indices = energy_data.get("scale_indices")
    energy_sign = energy_data.get("energy_sign", -1.0)
    microns_per_voxel = np.array(params.get("microns_per_voxel", [1.0, 1.0, 1.0]), dtype=float)

    if len(vertex_positions) == 0:
        return cast("dict[str, Any]", empty_edges_result(vertex_positions))

    candidate_manifest_path = stage_controller.artifact_path("candidates.pkl")
    candidate_audit_path = stage_controller.artifact_path("candidate_audit.json")
    chosen_manifest_path = stage_controller.artifact_path("chosen_edges.pkl")
    lumen_radius_pixels_axes = np.asarray(
        energy_data.get(
            "lumen_radius_pixels_axes",
            np.repeat(
                np.asarray(energy_data["lumen_radius_pixels"], dtype=np.float32).reshape(-1, 1),
                3,
                axis=1,
            ),
        ),
        dtype=np.float32,
    )
    logger.info("Creating vertex center lookup image...")
    vertex_center_image = paint_vertex_center_image(vertex_positions, energy.shape)
    logger.info("Vertex center lookup image created")
    use_frontier = use_matlab_frontier_tracer(energy_data, params)
    vertex_image: np.ndarray | None = None
    if not use_frontier:
        logger.info("Creating painted vertex occupancy image...")
        vertex_image = paint_vertex_image(
            vertex_positions,
            vertex_scales,
            lumen_radius_pixels_axes,
            energy.shape,
        )
        logger.info("Painted vertex occupancy image created")

    stage_controller.begin(
        detail=(
            "Generating edge candidates through MATLAB-style frontier workflow"
            if use_frontier
            else "Generating edge candidates through maintained frontier workflow"
        ),
        units_total=3,
        units_completed=0,
        substage="generate_candidates",
        resumed=False,
    )
    if use_frontier:

        def _heartbeat(iteration_count: int, candidate_count: int) -> None:
            stage_controller.update(
                units_total=3,
                units_completed=0,
                substage="generate_candidates",
                detail=(
                    "Generating edge candidates through MATLAB-style frontier workflow "
                    f"(iterations={iteration_count}, candidates={candidate_count})"
                ),
                resumed=False,
            )

        candidates = generate_edge_candidates_matlab_frontier(
            energy,
            scale_indices,
            vertex_positions,
            vertex_scales,
            lumen_radius_microns,
            microns_per_voxel,
            vertex_center_image,
            params,
            heartbeat=_heartbeat,
        )
        candidates = finalize_matlab_parity_candidates(
            candidates,
            energy,
            scale_indices,
            vertex_positions,
            energy_sign,
            params,
            microns_per_voxel,
        )
        frontier_origin_counts_raw = normalize_candidate_origin_counts(
            candidates.get("diagnostics", {}).get("frontier_per_origin_candidate_counts")
        )
        frontier_origin_counts = {
            int(origin_index): int(count)
            for origin_index, count in frontier_origin_counts_raw.items()
        }
    else:
        vertex_positions_microns = vertex_positions * microns_per_voxel
        tree = cKDTree(vertex_positions_microns)
        max_vertex_radius = np.max(lumen_radius_microns) if len(lumen_radius_microns) > 0 else 0.0
        max_search_radius = max_vertex_radius * 5.0
        candidates = generate_edge_candidates(
            energy=energy,
            scale_indices=scale_indices,
            vertex_positions=vertex_positions,
            vertex_scales=vertex_scales,
            lumen_radius_pixels=energy_data["lumen_radius_pixels"],
            lumen_radius_microns=lumen_radius_microns,
            microns_per_voxel=microns_per_voxel,
            vertex_center_image=vertex_center_image,
            vertex_image=vertex_image,
            tree=tree,
            max_search_radius=max_search_radius,
            params=params,
            energy_sign=energy_sign,
        )

        connection_sources = [str(value) for value in candidates.get("connection_sources", [])]
        origin_indices = np.asarray(
            candidates.get("origin_indices", np.zeros((0,), dtype=np.int32)),
            dtype=np.int32,
        ).reshape(-1)
        frontier_origin_counts = {}
        for index, origin_index in enumerate(origin_indices):
            if index >= len(connection_sources) or connection_sources[index] != "frontier":
                continue
            origin_key = int(origin_index)
            frontier_origin_counts[origin_key] = frontier_origin_counts.get(origin_key, 0) + 1

    supplement_origin_counts = normalize_candidate_origin_counts(
        candidates.get("diagnostics", {}).get("watershed_per_origin_candidate_counts")
    )

    candidate_audit = build_edge_candidate_audit(
        candidates,
        len(vertex_positions),
        use_frontier_tracer=use_frontier,
        frontier_origin_counts=frontier_origin_counts,
        supplement_origin_counts={
            int(origin_index): int(count)
            for origin_index, count in (supplement_origin_counts or {}).items()
        },
    )
    atomic_write_json(candidate_audit_path, candidate_audit)

    atomic_joblib_dump(candidates, candidate_manifest_path)
    if use_frontier:
        candidate_checkpoint_path = (
            stage_controller.run_context.checkpoints_dir / "checkpoint_edge_candidates.pkl"
        )
        candidate_checkpoint_payload = build_candidate_snapshot_payload(candidates)
        atomic_joblib_dump(candidate_checkpoint_payload, candidate_checkpoint_path)
    stage_controller.update(
        units_total=3,
        units_completed=1,
        substage="generate_candidates",
        detail="Generated edge candidates",
        resumed=False,
    )
    chosen = choose_edges_for_workflow(
        candidates,
        vertex_positions,
        vertex_scales,
        lumen_radius_microns,
        lumen_radius_pixels_axes,
        energy.shape,
        params,
    )
    if use_frontier:
        chosen = add_vertices_to_edges_matlab_style(
            chosen,
            vertices,
            energy=energy,
            scale_indices=scale_indices,
            microns_per_voxel=microns_per_voxel,
            lumen_radius_microns=lumen_radius_microns,
            lumen_radius_pixels_axes=lumen_radius_pixels_axes,
            size_of_image=energy.shape,
            params=params,
        )
    chosen = finalize_edges_matlab_style(
        chosen,
        lumen_radius_microns=lumen_radius_microns,
        microns_per_voxel=microns_per_voxel,
        size_of_image=energy.shape,
    )
    chosen["lumen_radius_microns"] = np.asarray(lumen_radius_microns, dtype=np.float32).copy()
    if use_frontier and candidates.get("frontier_lifecycle_events"):
        candidate_lifecycle_path = stage_controller.artifact_path("candidate_lifecycle.json")
        candidate_lifecycle = build_frontier_candidate_lifecycle(
            candidates,
            chosen.get("chosen_candidate_indices"),
        )
        atomic_write_json(candidate_lifecycle_path, candidate_lifecycle)
    atomic_joblib_dump(chosen, chosen_manifest_path)
    stage_controller.update(
        units_total=3,
        units_completed=3,
        substage="choose_edges",
        detail=("Selected MATLAB-style terminal edges" if use_frontier else "Selected final edges"),
        resumed=False,
    )
    return chosen


def extract_edges_watershed_resumable(
    energy_data: dict[str, Any],
    vertices: dict[str, Any],
    params: dict[str, Any],
    stage_controller: StageController,
    *,
    atomic_joblib_dump: Callable[..., None],
    append_candidate_unit: Callable[..., None],
    empty_edge_diagnostics: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    """Extract watershed edges with per-label persisted units."""
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

    units_dir = stage_controller.artifact_path("units")
    units_dir.mkdir(parents=True, exist_ok=True)
    existing_payload, completed = _load_edge_units(
        units_dir, append_candidate_unit, empty_edge_diagnostics
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

    return {
        "traces": edges,
        "connections": np.asarray(connections, dtype=np.int32).reshape(-1, 2),
        "energies": np.asarray(edge_energies, dtype=np.float32),
        "vertex_positions": vertex_positions.astype(np.float32),
    }


__all__ = [
    "extract_edges_resumable",
    "extract_edges_watershed_resumable",
]
