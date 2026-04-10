"""Edge extraction orchestration for SLAVV."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import joblib
import numpy as np
import scipy.ndimage as ndi
from scipy.spatial import cKDTree
from skimage.segmentation import watershed

from .edge_candidates import (
    _append_candidate_unit,
    _build_edge_candidate_audit,
    _finalize_matlab_parity_candidates,
    _generate_edge_candidates,
    _generate_edge_candidates_matlab_frontier,
    _normalize_candidate_connection_sources,
    _normalize_candidate_origin_counts,
    _trace_origin_edges_matlab_frontier,
    _use_matlab_frontier_tracer,
)
from .edge_primitives import (
    _edge_metric_from_energy_trace,
    _record_trace_diagnostics,
    _scalar_radius,
    _trace_energy_series,
    _trace_scale_series,
    estimate_vessel_directions,
    generate_edge_directions,
    trace_edge,
)
from .edge_selection import (
    _choose_edges_matlab_style,
    _empty_edge_diagnostics,
    _empty_edges_result,
    _merge_edge_diagnostics,
)
from .vertices import paint_vertex_center_image

if TYPE_CHECKING:
    from pathlib import Path

    from slavv.runtime import StageController

logger = logging.getLogger(__name__)


def extract_edges(
    energy_data: dict[str, Any], vertices: dict[str, Any], params: dict[str, Any]
) -> dict[str, Any]:
    """
    Extract edges by tracing from vertices through energy field.
    MATLAB Equivalent: `get_edges_V300.m`
    """
    logger.info("Extracting edges")

    energy = energy_data["energy"]
    vertex_positions = vertices["positions"]
    vertex_scales = vertices["scales"]
    lumen_radius_pixels = energy_data["lumen_radius_pixels"]
    lumen_radius_microns = energy_data["lumen_radius_microns"]
    energy_sign = energy_data.get("energy_sign", -1.0)

    microns_per_voxel = np.array(params.get("microns_per_voxel", [1.0, 1.0, 1.0]), dtype=float)
    scale_indices = energy_data.get("scale_indices")

    if len(vertex_positions) == 0:
        logger.info("Extracted 0 edges")
        return _empty_edges_result(vertex_positions)

    lumen_radius_pixels_axes = energy_data["lumen_radius_pixels_axes"]
    logger.info("Creating vertex center lookup image...")
    vertex_center_image = paint_vertex_center_image(vertex_positions, energy.shape)
    logger.info("Vertex center lookup image created")

    vertex_positions_microns = vertex_positions * microns_per_voxel
    tree = cKDTree(vertex_positions_microns)
    max_vertex_radius = np.max(lumen_radius_microns) if len(lumen_radius_microns) > 0 else 0.0
    max_search_radius = max_vertex_radius * 5.0
    if _use_matlab_frontier_tracer(energy_data, params):
        candidates = _generate_edge_candidates_matlab_frontier(
            energy,
            scale_indices,
            vertex_positions,
            vertex_scales,
            lumen_radius_microns,
            microns_per_voxel,
            vertex_center_image,
            params,
        )
        candidates = _finalize_matlab_parity_candidates(
            candidates,
            energy,
            scale_indices,
            vertex_positions,
            energy_sign,
            params,
            microns_per_voxel,
        )
    else:
        candidates = _generate_edge_candidates(
            energy,
            scale_indices,
            vertex_positions,
            vertex_scales,
            lumen_radius_pixels,
            lumen_radius_microns,
            microns_per_voxel,
            vertex_center_image,
            tree,
            max_search_radius,
            params,
            energy_sign,
        )
    chosen = _choose_edges_matlab_style(
        candidates,
        vertex_positions.astype(np.float32),
        vertex_scales,
        lumen_radius_pixels_axes,
        energy.shape,
        params,
    )
    logger.info(
        "Extracted %d chosen edges from %d traced candidates",
        len(chosen["traces"]),
        chosen["diagnostics"]["candidate_traced_edge_count"],
    )
    return chosen


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
    log_interval = max(1, n_vertices // 20)  # Log ~20 times over the loop

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
        "connections": np.asarray(connections, dtype=np.int32),
        "energies": np.asarray(edge_energies, dtype=np.float32),
        "vertex_positions": vertex_positions.astype(np.float32),
    }


def _load_edge_units(
    units_dir: Path,
    n_vertices: int,
) -> tuple[dict[str, Any], set[int]]:
    payload = {
        "traces": [],
        "connections": np.zeros((0, 2), dtype=np.int32),
        "metrics": np.zeros((0,), dtype=np.float32),
        "energy_traces": [],
        "scale_traces": [],
        "origin_indices": np.zeros((0,), dtype=np.int32),
        "connection_sources": [],
        "diagnostics": _empty_edge_diagnostics(),
    }
    completed: set[int] = set()

    if not units_dir.exists():
        return payload, completed

    traces: list[np.ndarray] = []
    connections: list[list[int]] = []
    metrics: list[float] = []
    energy_traces: list[np.ndarray] = []
    scale_traces: list[np.ndarray] = []
    origin_indices: list[int] = []
    connection_sources: list[str] = []
    for unit_file in sorted(units_dir.glob("*.pkl")):
        unit_payload = joblib.load(unit_file)
        origin_index = int(unit_payload["origin_index"])
        completed.add(origin_index)
        unit_traces = [np.asarray(trace, dtype=np.float32) for trace in unit_payload["traces"]]
        unit_connections = np.asarray(unit_payload["connections"], dtype=np.int32).reshape(-1, 2)
        traces.extend(unit_traces)
        connections.extend(
            [int(connection[0]), int(connection[1])] for connection in unit_connections
        )
        metrics.extend(float(metric) for metric in unit_payload["metrics"])
        energy_traces.extend(
            np.asarray(trace, dtype=np.float32) for trace in unit_payload["energy_traces"]
        )
        scale_traces.extend(
            np.asarray(trace, dtype=np.int16) for trace in unit_payload["scale_traces"]
        )
        origin_indices.extend(int(value) for value in unit_payload.get("origin_indices", []))
        connection_sources.extend(
            _normalize_candidate_connection_sources(
                unit_payload.get("connection_sources"),
                len(unit_connections),
                default_source=str(unit_payload.get("candidate_source", "unknown")),
            )
        )
        _merge_edge_diagnostics(
            cast("dict[str, Any]", payload["diagnostics"]),
            cast("dict[str, Any]", unit_payload.get("diagnostics", {})),
        )

    payload["traces"] = traces
    payload["connections"] = np.asarray(connections, dtype=np.int32).reshape(-1, 2)
    payload["metrics"] = np.asarray(metrics, dtype=np.float32)
    payload["energy_traces"] = energy_traces
    payload["scale_traces"] = scale_traces
    payload["origin_indices"] = np.asarray(origin_indices, dtype=np.int32)
    payload["connection_sources"] = connection_sources
    return payload, completed


def extract_edges_resumable(
    energy_data: dict[str, Any],
    vertices: dict[str, Any],
    params: dict[str, Any],
    stage_controller: StageController,
) -> dict[str, Any]:
    """Trace edges with per-origin persisted units."""
    from slavv.runtime.run_state import atomic_joblib_dump, atomic_write_json

    energy = energy_data["energy"]
    vertex_positions = vertices["positions"]
    vertex_scales = vertices["scales"]
    lumen_radius_pixels = energy_data["lumen_radius_pixels"]
    lumen_radius_microns = energy_data["lumen_radius_microns"]
    scale_indices = energy_data.get("scale_indices")
    energy_sign = energy_data.get("energy_sign", -1.0)
    max_edges_per_vertex = params.get("number_of_edges_per_vertex", 4)
    step_size_ratio = params.get("step_size_per_origin_radius", 1.0)
    max_edge_energy = params.get("max_edge_energy", 0.0)
    max_length_ratio = params.get("max_edge_length_per_origin_radius", 60.0)
    microns_per_voxel = np.array(params.get("microns_per_voxel", [1.0, 1.0, 1.0]), dtype=float)
    discrete_tracing = params.get("discrete_tracing", False)
    direction_method = params.get("direction_method", "hessian")

    if len(vertex_positions) == 0:
        return _empty_edges_result(vertex_positions)

    units_dir = stage_controller.artifact_path("units")
    units_dir.mkdir(parents=True, exist_ok=True)
    candidate_manifest_path = stage_controller.artifact_path("candidates.pkl")
    candidate_audit_path = stage_controller.artifact_path("candidate_audit.json")
    chosen_manifest_path = stage_controller.artifact_path("chosen_edges.pkl")
    candidates, completed = _load_edge_units(units_dir, len(vertex_positions))

    lumen_radius_pixels_axes = energy_data["lumen_radius_pixels_axes"]
    logger.info("Creating vertex center lookup image...")
    vertex_center_image = paint_vertex_center_image(vertex_positions, energy.shape)
    logger.info("Vertex center lookup image created")
    vertex_positions_microns = vertex_positions * microns_per_voxel
    tree = cKDTree(vertex_positions_microns)
    max_vertex_radius = np.max(lumen_radius_microns) if len(lumen_radius_microns) > 0 else 0.0
    max_search_radius = max_vertex_radius * 5.0
    energy_prepared = np.ascontiguousarray(energy, dtype=np.float64)
    mpv_prepared = np.asarray(microns_per_voxel, dtype=np.float64)
    use_frontier_tracer = _use_matlab_frontier_tracer(energy_data, params)

    stage_controller.begin(
        detail="Tracing edges with resumable origin units",
        units_total=len(vertex_positions) + 2,
        units_completed=len(completed),
        substage="trace_origins",
        resumed=bool(completed),
    )

    frontier_origin_counts: dict[int, int] = {}
    if use_frontier_tracer:
        for origin_index, count in (
            candidates.get("diagnostics", {})
            .get("frontier_per_origin_candidate_counts", {})
            .items()
        ):
            try:
                frontier_origin_counts[int(origin_index)] = int(count)
            except (TypeError, ValueError):
                continue

    # Results for the current vertex
    unit_traces: list[np.ndarray] = []
    unit_connections: list[list[int]] = []
    unit_metrics: list[float] = []
    unit_energy_traces: list[np.ndarray] = []
    unit_scale_traces: list[np.ndarray] = []
    unit_trace_metadata_v: list[dict[str, Any]] = []

    for vertex_idx, (start_pos, start_scale) in enumerate(zip(vertex_positions, vertex_scales)):
        if vertex_idx in completed:
            continue

        if use_frontier_tracer:
            frontier_payload = _trace_origin_edges_matlab_frontier(
                energy,
                scale_indices,
                vertex_positions,
                vertex_scales,
                lumen_radius_microns,
                microns_per_voxel,
                vertex_center_image,
                vertex_idx,
                params,
            )
            unit_traces = cast("list[np.ndarray]", frontier_payload["traces"])
            unit_connections = cast("list[list[int]]", frontier_payload["connections"])
            unit_metrics = cast("list[float]", frontier_payload["metrics"])
            unit_energy_traces = cast("list[np.ndarray]", frontier_payload["energy_traces"])
            unit_scale_traces = cast("list[np.ndarray]", frontier_payload["scale_traces"])
            unit_diagnostics = cast("dict[str, Any]", frontier_payload["diagnostics"])
            unit_trace_metadata_v = []
            frontier_count = len(unit_connections)
            if frontier_count > 0:
                frontier_origin_counts[vertex_idx] = frontier_count
        else:
            unit_traces = []
            unit_connections = []
            unit_metrics = []
            unit_energy_traces = []
            unit_scale_traces = []
            unit_trace_metadata_v = []
            start_radius = _scalar_radius(lumen_radius_pixels[start_scale])
            step_size = start_radius * step_size_ratio
            max_length = start_radius * max_length_ratio
            max_steps = max(1, int(np.ceil(max_length / max(step_size, 1e-12))))
            if direction_method == "hessian":
                directions = estimate_vessel_directions(
                    energy,
                    start_pos,
                    start_radius,
                    microns_per_voxel,
                )
                if directions.shape[0] < max_edges_per_vertex:
                    extra = generate_edge_directions(
                        max_edges_per_vertex - directions.shape[0], seed=vertex_idx
                    )
                    directions = np.vstack([directions, extra])
                else:
                    directions = directions[:max_edges_per_vertex]
            else:
                directions = generate_edge_directions(max_edges_per_vertex, seed=vertex_idx)

            for direction in directions:
                res_te = trace_edge(
                    energy_prepared,
                    start_pos,
                    direction,
                    step_size,
                    max_edge_energy,
                    vertex_positions,
                    vertex_scales,
                    lumen_radius_pixels,
                    lumen_radius_microns,
                    max_steps,
                    mpv_prepared,
                    energy_sign,
                    discrete_steps=discrete_tracing,
                    vertex_center_image=vertex_center_image,
                    tree=tree,
                    max_search_radius=max_search_radius,
                    origin_vertex_idx=vertex_idx,
                    return_metadata=True,
                )
                edge_trace, trace_metadata = cast("tuple[list[np.ndarray], dict[str, Any]]", res_te)
                if len(edge_trace) <= 1:
                    continue
                edge_arr = np.asarray(edge_trace, dtype=np.float32)
                term_v: int = (
                    int(trace_metadata["terminal_vertex"])
                    if trace_metadata["terminal_vertex"] is not None
                    else -1
                )
                energy_trace = _trace_energy_series(edge_arr, energy)
                scale_trace = _trace_scale_series(edge_arr, scale_indices)
                unit_traces.append(edge_arr)
                unit_connections.append([vertex_idx, term_v])
                unit_metrics.append(_edge_metric_from_energy_trace(energy_trace))
                unit_energy_traces.append(energy_trace)
                unit_scale_traces.append(scale_trace)
                unit_trace_metadata_v.append(trace_metadata)

            unit_diagnostics = _empty_edge_diagnostics()
            for item in unit_trace_metadata_v:
                trace_metadata_item = cast("dict[str, Any]", item)
                _record_trace_diagnostics(unit_diagnostics, trace_metadata_item)

        if use_frontier_tracer:
            payload = {
                "origin_index": vertex_idx,
                "candidate_source": str(frontier_payload.get("candidate_source", "frontier")),
                "traces": unit_traces,
                "connections": unit_connections,
                "metrics": unit_metrics,
                "energy_traces": unit_energy_traces,
                "scale_traces": unit_scale_traces,
                "origin_indices": list(frontier_payload.get("origin_indices", [])),
                "connection_sources": list(frontier_payload.get("connection_sources", [])),
                "diagnostics": unit_diagnostics,
            }
        else:
            payload = {
                "origin_index": vertex_idx,
                "candidate_source": "fallback",
                "traces": unit_traces,
                "connections": unit_connections,
                "metrics": unit_metrics,
                "energy_traces": unit_energy_traces,
                "scale_traces": unit_scale_traces,
                "origin_indices": [vertex_idx] * len(unit_traces),
                "connection_sources": ["fallback"] * len(unit_traces),
                "diagnostics": unit_diagnostics,
            }
        atomic_joblib_dump(payload, units_dir / f"vertex_{vertex_idx:06d}.pkl")
        _append_candidate_unit(candidates, payload)
        completed.add(vertex_idx)
        stage_controller.save_state({"last_completed_origin": vertex_idx})
        stage_controller.update(
            units_total=len(vertex_positions) + 2,
            units_completed=len(completed),
            substage="trace_origins",
            detail=f"Tracing origin {vertex_idx + 1}/{len(vertex_positions)}",
            resumed=bool(completed - {vertex_idx}),
        )

    if use_frontier_tracer:
        candidates = _finalize_matlab_parity_candidates(
            candidates,
            energy,
            scale_indices,
            vertex_positions,
            energy_sign,
            params,
            microns_per_voxel,
        )
        supplement_origin_counts = _normalize_candidate_origin_counts(
            candidates.get("diagnostics", {}).get("watershed_per_origin_candidate_counts")
        )
    else:
        supplement_origin_counts = {}

    candidate_audit = _build_edge_candidate_audit(
        candidates,
        len(vertex_positions),
        use_frontier_tracer=use_frontier_tracer,
        frontier_origin_counts=frontier_origin_counts,
        supplement_origin_counts={
            int(origin_index): int(count)
            for origin_index, count in (supplement_origin_counts or {}).items()
        },
    )
    atomic_write_json(candidate_audit_path, candidate_audit)

    atomic_joblib_dump(candidates, candidate_manifest_path)
    stage_controller.update(
        units_total=len(vertex_positions) + 2,
        units_completed=len(completed) + 1,
        substage="consolidate_candidates",
        detail="Consolidated edge candidates",
        resumed=bool(completed),
    )
    chosen = _choose_edges_matlab_style(
        candidates,
        vertex_positions.astype(np.float32),
        vertex_scales,
        lumen_radius_pixels_axes,
        energy.shape,
        params,
    )
    atomic_joblib_dump(chosen, chosen_manifest_path)
    stage_controller.update(
        units_total=len(vertex_positions) + 2,
        units_completed=len(vertex_positions) + 2,
        substage="choose_edges",
        detail="Selected MATLAB-style terminal edges",
        resumed=bool(completed),
    )
    return chosen


def extract_edges_watershed_resumable(
    energy_data: dict[str, Any],
    vertices: dict[str, Any],
    params: dict[str, Any],
    stage_controller: StageController,
) -> dict[str, Any]:
    """Extract watershed edges with per-label persisted units."""
    from slavv.runtime.run_state import atomic_joblib_dump

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
    existing_payload, completed = _load_edge_units(units_dir, len(vertex_positions))
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
            "energy_traces": [np.asarray([energy], dtype=np.float32) for energy in unit_energies],
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
