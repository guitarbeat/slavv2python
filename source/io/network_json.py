"""Authoritative JSON export helpers for SLAVV vascular networks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import networkx as nx
import numpy as np

from source.models import normalize_pipeline_result

if TYPE_CHECKING:
    from collections.abc import Mapping

    from source.runtime import RunSnapshot

NETWORK_JSON_SCHEMA_NAME = "slavv-network"
NETWORK_JSON_SCHEMA_VERSION = 2

_VERTEX_EXPORT_FIELDS = (
    "positions",
    "radii_microns",
    "radii_pixels",
    "energies",
    "scales",
)
_EDGE_EXPORT_FIELDS = (
    "connections",
    "traces",
    "energies",
    "scale_traces",
    "energy_traces",
    "lumen_radius_microns",
    "bridge_vertex_positions",
)
_NETWORK_EXPORT_FIELDS = (
    "strands",
    "bifurcations",
    "orphans",
    "cycles",
    "mismatched_strands",
    "vertex_degrees",
    "edge_indices_in_strands",
    "edge_backwards_in_strands",
    "end_vertices_in_strands",
    "strand_subscripts",
    "strand_traces",
    "strand_space_traces",
    "strand_scale_traces",
    "strand_energy_traces",
    "mean_strand_energies",
    "vessel_directions",
)


def _normalize_vertices_array(vertices: Any) -> np.ndarray:
    """Return vertex data in stable ``(N, 3)`` float form."""
    array = np.asarray(vertices, dtype=float)
    if array.size == 0:
        return np.empty((0, 3), dtype=float)
    return np.atleast_2d(array)


def _normalize_edges_array(edges: Any) -> np.ndarray:
    """Return edge connections in stable ``(N, 2)`` integer form."""
    array = np.asarray(edges, dtype=int)
    return np.empty((0, 2), dtype=int) if array.size == 0 else np.atleast_2d(array)[:, :2]


def _normalize_numeric_vector(values: Any, *, dtype: Any) -> np.ndarray:
    """Normalize a flat numeric vector."""
    array = np.asarray(values, dtype=dtype).reshape(-1)
    return cast("np.ndarray", array)


def _normalize_trace_list(values: Any) -> list[np.ndarray]:
    """Normalize a list of spatial trace arrays."""
    return [np.asarray(value, dtype=float) for value in values or []]


def _normalize_vector_list(values: Any, *, dtype: Any) -> list[np.ndarray]:
    """Normalize a list of flat numeric vectors."""
    return [np.asarray(value, dtype=dtype).reshape(-1) for value in values or []]


def _normalize_optional_list(values: Any) -> list[Any]:
    """Normalize JSON-like lists while tolerating ``None``."""
    return list(values or [])


def _convert_edges_to_strands(edges: np.ndarray, *, vertex_count: int) -> list[list[int]]:
    """Build simple strand-like paths from edge connectivity."""
    if vertex_count == 0 or edges.size == 0:
        return []

    graph = nx.Graph()
    graph.add_nodes_from(range(vertex_count))
    for origin_idx, destination_idx in edges.tolist():
        origin = int(origin_idx)
        destination = int(destination_idx)
        if origin == destination:
            continue
        if not (0 <= origin < vertex_count and 0 <= destination < vertex_count):
            continue
        graph.add_edge(origin, destination)

    strands: list[list[int]] = []
    visited_edges: set[tuple[int, int]] = set()

    for origin, destination in graph.edges():
        undirected_edge = (min(origin, destination), max(origin, destination))
        if undirected_edge in visited_edges:
            continue

        strand = [origin, destination]
        visited_edges.add(undirected_edge)

        current = destination
        while graph.degree(current) == 2:
            neighbors = list(graph.neighbors(current))
            next_node = int(neighbors[0] if neighbors[1] == strand[-2] else neighbors[1])
            next_edge = (min(current, next_node), max(current, next_node))
            if next_edge in visited_edges:
                break
            strand.append(next_node)
            visited_edges.add(next_edge)
            current = next_node

        current = origin
        while graph.degree(current) == 2:
            neighbors = list(graph.neighbors(current))
            next_node = int(neighbors[0] if neighbors[1] == strand[1] else neighbors[1])
            next_edge = (min(current, next_node), max(current, next_node))
            if next_edge in visited_edges:
                break
            strand.insert(0, next_node)
            visited_edges.add(next_edge)
            current = next_node

        strands.append(strand)

    return strands


def infer_image_shape_from_vertices(vertex_positions: Any) -> tuple[int, int, int]:
    """Infer a minimal positive image shape from exported vertex positions."""
    positions = _normalize_vertices_array(vertex_positions)
    if positions.size == 0:
        return (1, 1, 1)
    maxima = np.max(positions, axis=0)
    inferred_axes = [max(1, int(np.ceil(float(axis_max))) + 1) for axis_max in maxima[:3]]
    while len(inferred_axes) < 3:
        inferred_axes.append(1)
    return tuple(inferred_axes[:3])


def _json_safe(value: Any) -> Any:
    """Convert numpy-heavy payloads into JSON-safe builtin types."""
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, set):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def _select_fields(payload: Mapping[str, Any], allowed_fields: tuple[str, ...]) -> dict[str, Any]:
    """Copy only the selected fields from a mapping."""
    return {field: payload[field] for field in allowed_fields if field in payload}


def _build_authoritative_topology(
    *,
    vertices: Mapping[str, Any],
    edges: Mapping[str, Any],
    network: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the public topology block, computing missing basics when needed."""
    vertex_positions = _normalize_vertices_array(vertices.get("positions", []))
    edge_connections = _normalize_edges_array(edges.get("connections", []))
    vertex_count = len(vertex_positions)

    graph = nx.Graph()
    graph.add_nodes_from(range(vertex_count))
    if edge_connections.size:
        graph.add_edges_from(edge_connections.tolist())

    computed_degrees = np.array([graph.degree(node) for node in range(vertex_count)], dtype=np.int32)
    topology = _select_fields(network, _NETWORK_EXPORT_FIELDS)
    topology.setdefault("strands", _convert_edges_to_strands(edge_connections, vertex_count=vertex_count))
    topology.setdefault(
        "bifurcations",
        np.flatnonzero(computed_degrees > 2).astype(np.int32, copy=False),
    )
    topology.setdefault("orphans", np.flatnonzero(computed_degrees == 0).astype(np.int32, copy=False))
    topology.setdefault("vertex_degrees", computed_degrees)
    topology.setdefault("mismatched_strands", [])
    topology.setdefault("cycles", [])
    return topology


def _build_summary_block(
    *,
    vertices: Mapping[str, Any],
    edges: Mapping[str, Any],
    network: Mapping[str, Any],
    parameters: Mapping[str, Any],
    image_shape: tuple[int, int, int],
) -> dict[str, Any] | None:
    """Calculate analysis-ready summary statistics when the payload is complete enough."""
    if "positions" not in vertices:
        return None

    from source.analysis import calculate_network_statistics

    try:
        radii = np.asarray(vertices.get("radii_microns", []), dtype=float)
        stats = calculate_network_statistics(
            list(network.get("strands", [])),
            np.asarray(network.get("bifurcations", []), dtype=int),
            _normalize_vertices_array(vertices.get("positions", [])),
            radii,
            list(parameters.get("microns_per_voxel", [1.0, 1.0, 1.0])),
            image_shape,
            edge_energies=np.asarray(edges.get("energies", []), dtype=float),
        )
    except Exception:
        return None
    return cast("dict[str, Any]", _json_safe(stats))


def _build_metadata(
    *,
    parameters: Mapping[str, Any],
    image_shape: tuple[int, int, int],
    run_snapshot: RunSnapshot | None,
    run_dir: str | Path | None,
    metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build the public metadata block for authoritative exports."""
    stage_provenance: dict[str, Any] = {}
    if run_snapshot is not None:
        stage_provenance = {
            stage_name: {
                "status": stage_snapshot.status,
                "elapsed_seconds": float(stage_snapshot.elapsed_seconds),
                "peak_memory_bytes": int(stage_snapshot.peak_memory_bytes),
                "completed_at": stage_snapshot.completed_at,
            }
            for stage_name, stage_snapshot in run_snapshot.stages.items()
        }

    payload = {
        "pipeline_profile": parameters.get("pipeline_profile", "unprofiled"),
        "image_shape": list(image_shape),
        "microns_per_voxel": list(parameters.get("microns_per_voxel", [1.0, 1.0, 1.0])),
        "run_id": None if run_snapshot is None else run_snapshot.run_id,
        "run_dir": None if run_dir is None else str(run_dir),
        "run_status": None if run_snapshot is None else run_snapshot.status,
        "target_stage": None if run_snapshot is None else run_snapshot.target_stage,
        "stage_provenance": stage_provenance,
    }
    if metadata:
        payload.update(dict(metadata))
    return {key: value for key, value in payload.items() if value is not None}


def build_network_json_payload(
    processing_results: Mapping[str, Any],
    *,
    run_snapshot: RunSnapshot | None = None,
    run_dir: str | Path | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the authoritative versioned JSON payload for public network exports."""
    normalized_results = normalize_pipeline_result(processing_results).to_dict()
    parameters = cast("dict[str, Any]", dict(normalized_results.get("parameters", {})))
    vertices = cast(
        "dict[str, Any]",
        _select_fields(cast("Mapping[str, Any]", normalized_results.get("vertices", {})), _VERTEX_EXPORT_FIELDS),
    )
    edges = cast(
        "dict[str, Any]",
        _select_fields(cast("Mapping[str, Any]", normalized_results.get("edges", {})), _EDGE_EXPORT_FIELDS),
    )
    topology = _build_authoritative_topology(
        vertices=cast("Mapping[str, Any]", vertices),
        edges=cast("Mapping[str, Any]", edges),
        network=cast("Mapping[str, Any]", normalized_results.get("network", {})),
    )
    if "vertex_degrees" in topology:
        vertices["degrees"] = np.asarray(topology["vertex_degrees"], dtype=np.int32)

    energy_data = cast("Mapping[str, Any]", normalized_results.get("energy_data", {}))
    image_shape = tuple(
        int(axis)
        for axis in energy_data.get(
            "image_shape",
            infer_image_shape_from_vertices(vertices.get("positions", [])),
        )
    )
    summary = _build_summary_block(
        vertices=cast("Mapping[str, Any]", vertices),
        edges=cast("Mapping[str, Any]", edges),
        network=topology,
        parameters=parameters,
        image_shape=image_shape,
    )

    payload = {
        "schema": {
            "name": NETWORK_JSON_SCHEMA_NAME,
            "version": NETWORK_JSON_SCHEMA_VERSION,
        },
        "metadata": _build_metadata(
            parameters=parameters,
            image_shape=image_shape,
            run_snapshot=run_snapshot,
            run_dir=run_dir,
            metadata=metadata,
        ),
        "parameters": parameters,
        "vertices": vertices,
        "edges": edges,
        "network": topology,
    }
    if summary is not None:
        payload["summary"] = summary
    return cast("dict[str, Any]", _json_safe(payload))


def load_network_json_payload(path: str | Path) -> dict[str, Any]:
    """Load a network JSON export with authoritative-schema and legacy compatibility."""
    with open(Path(path), encoding="utf-8") as handle:
        payload = cast("dict[str, Any]", json.load(handle))

    schema = cast("dict[str, Any]", payload.get("schema", {}))
    if schema.get("name") == NETWORK_JSON_SCHEMA_NAME and schema.get("version") == NETWORK_JSON_SCHEMA_VERSION:
        metadata = cast("dict[str, Any]", payload.get("metadata", {}))
        vertices = cast("dict[str, Any]", dict(payload.get("vertices", {})))
        edges = cast("dict[str, Any]", dict(payload.get("edges", {})))
        network = cast("dict[str, Any]", dict(payload.get("network", {})))
        parameters = cast("dict[str, Any]", dict(payload.get("parameters", {})))
    else:
        metadata = {
            "pipeline_profile": payload.get("parameters", {}).get("pipeline_profile", "legacy"),
            "image_shape": list(
                payload.get(
                    "image_shape",
                    infer_image_shape_from_vertices(payload.get("vertices", {}).get("positions", [])),
                )
            ),
            "microns_per_voxel": list(
                payload.get("parameters", {}).get("microns_per_voxel", [1.0, 1.0, 1.0])
            ),
        }
        vertices = cast("dict[str, Any]", dict(payload.get("vertices", {})))
        edges = cast("dict[str, Any]", dict(payload.get("edges", {})))
        network = cast("dict[str, Any]", dict(payload.get("network", {})))
        parameters = cast("dict[str, Any]", dict(payload.get("parameters", {})))

    vertex_positions = _normalize_vertices_array(vertices.get("positions", []))
    edge_connections = _normalize_edges_array(edges.get("connections", []))
    topology = _build_authoritative_topology(
        vertices=vertices,
        edges=edges,
        network=network,
    )
    radii_microns = _normalize_numeric_vector(vertices.get("radii_microns", vertices.get("radii", [])), dtype=float)
    if radii_microns.size == 0 and len(vertex_positions) > 0:
        radii_microns = np.zeros((len(vertex_positions),), dtype=float)
    radii_pixels = _normalize_numeric_vector(vertices.get("radii_pixels", []), dtype=float)
    if radii_pixels.size == 0 and radii_microns.size == len(vertex_positions):
        radii_pixels = radii_microns.copy()

    normalized_payload = {
        "schema": {
            "name": NETWORK_JSON_SCHEMA_NAME,
            "version": NETWORK_JSON_SCHEMA_VERSION,
        },
        "metadata": metadata,
        "parameters": parameters,
        "image_shape": tuple(
            int(axis) for axis in metadata.get("image_shape", infer_image_shape_from_vertices(vertex_positions))
        ),
        "vertices": {
            "positions": vertex_positions,
            "radii_microns": radii_microns,
            "radii_pixels": radii_pixels,
            "energies": _normalize_numeric_vector(vertices.get("energies", []), dtype=float),
            "scales": _normalize_numeric_vector(vertices.get("scales", []), dtype=np.int16),
            "degrees": _normalize_numeric_vector(
                vertices.get("degrees", topology.get("vertex_degrees", [])),
                dtype=np.int32,
            ),
        },
        "edges": {
            "connections": edge_connections,
            "traces": _normalize_trace_list(edges.get("traces", [])),
            "energies": _normalize_numeric_vector(edges.get("energies", []), dtype=float),
            "scale_traces": _normalize_vector_list(edges.get("scale_traces", []), dtype=float),
            "energy_traces": _normalize_vector_list(edges.get("energy_traces", []), dtype=float),
            "lumen_radius_microns": _normalize_numeric_vector(
                edges.get("lumen_radius_microns", []),
                dtype=float,
            ),
            "bridge_vertex_positions": _normalize_vertices_array(
                edges.get("bridge_vertex_positions", [])
            ),
        },
        "network": {
            "strands": [list(strand) for strand in topology.get("strands", [])],
            "bifurcations": _normalize_numeric_vector(topology.get("bifurcations", []), dtype=np.int32),
            "orphans": _normalize_numeric_vector(topology.get("orphans", []), dtype=np.int32),
            "cycles": _normalize_optional_list(topology.get("cycles", [])),
            "mismatched_strands": _normalize_optional_list(topology.get("mismatched_strands", [])),
            "vertex_degrees": _normalize_numeric_vector(
                topology.get("vertex_degrees", []),
                dtype=np.int32,
            ),
            "edge_indices_in_strands": _normalize_vector_list(
                topology.get("edge_indices_in_strands", []),
                dtype=np.int32,
            ),
            "edge_backwards_in_strands": [
                np.asarray(value, dtype=bool).reshape(-1)
                for value in topology.get("edge_backwards_in_strands", [])
            ],
            "end_vertices_in_strands": _normalize_vector_list(
                topology.get("end_vertices_in_strands", []),
                dtype=np.int32,
            ),
            "strand_subscripts": _normalize_trace_list(topology.get("strand_subscripts", [])),
            "strand_traces": _normalize_trace_list(topology.get("strand_traces", [])),
            "strand_space_traces": _normalize_trace_list(topology.get("strand_space_traces", [])),
            "strand_scale_traces": _normalize_vector_list(
                topology.get("strand_scale_traces", []),
                dtype=float,
            ),
            "strand_energy_traces": _normalize_vector_list(
                topology.get("strand_energy_traces", []),
                dtype=float,
            ),
            "mean_strand_energies": _normalize_numeric_vector(
                topology.get("mean_strand_energies", []),
                dtype=float,
            ),
            "vessel_directions": _normalize_trace_list(topology.get("vessel_directions", [])),
        },
    }
    if "summary" in payload and isinstance(payload["summary"], dict):
        normalized_payload["summary"] = dict(payload["summary"])
    return normalized_payload


__all__ = [
    "NETWORK_JSON_SCHEMA_NAME",
    "NETWORK_JSON_SCHEMA_VERSION",
    "build_network_json_payload",
    "infer_image_shape_from_vertices",
    "load_network_json_payload",
]
