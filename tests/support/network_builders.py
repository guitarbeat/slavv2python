"""Reusable network-object and export payload builders for tests."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

import numpy as np

from slavv_python.io import Network
from slavv_python.io.network_json import build_network_json_payload

from .payload_builders import build_processing_results

if TYPE_CHECKING:
    from pathlib import Path


def build_network_object(
    *,
    vertices: Any | None = None,
    edges: Any | None = None,
    radii: Any | None = None,
) -> Network:
    """Build a basic ``Network`` object for I/O tests."""
    vertex_array = np.asarray(
        vertices if vertices is not None else [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        dtype=float,
    )
    edge_array = np.asarray(edges if edges is not None else [[0, 1]], dtype=int)
    if radii is None:
        radii_array = np.array([4.0, 7.0], dtype=float) if len(vertex_array) else None
    else:
        radii_array = np.asarray(radii, dtype=float)
    return Network(vertices=vertex_array, edges=edge_array, radii=radii_array)


def build_export_ready_processing_results(**kwargs: Any) -> dict[str, Any]:
    """Build processing results with the fields expected by visualizer exports."""
    payload = build_processing_results(**kwargs)
    payload["parameters"].setdefault("microns_per_voxel", [1.0, 1.0, 1.0])
    return payload


def build_minimal_network_json_payload(
    *,
    network: Network | None = None,
    include_radii: bool = True,
) -> dict[str, Any]:
    """Build a minimal JSON-style network payload for import tests."""
    network_obj = network if network is not None else build_network_object()
    payload = {
        "vertices": {"positions": network_obj.vertices.tolist()},
        "edges": {"connections": network_obj.edges.tolist()},
    }
    if include_radii and network_obj.radii is not None:
        payload["vertices"]["radii_microns"] = network_obj.radii.tolist()
    return payload


def build_authoritative_network_json_payload(
    *,
    network: Network | None = None,
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a versioned authoritative JSON-style network payload for tests."""
    network_obj = network if network is not None else build_network_object()
    connections = np.asarray(network_obj.edges, dtype=int)
    vertex_count = len(np.asarray(network_obj.vertices, dtype=float))
    degrees = np.zeros((vertex_count,), dtype=int)
    for start_vertex, end_vertex in connections.tolist():
        degrees[int(start_vertex)] += 1
        degrees[int(end_vertex)] += 1
    strands = (
        [connections[0].astype(int).tolist()]
        if len(connections) == 1
        else [sorted({int(vertex) for vertex in connections.reshape(-1).tolist()})]
        if len(connections) > 1
        else []
    )
    processing_results = build_processing_results(
        vertex_positions=network_obj.vertices.tolist(),
        edge_connections=network_obj.edges.tolist(),
        network={
            "strands": strands,
            "bifurcations": np.flatnonzero(degrees > 2).astype(int, copy=False),
            "vertex_degrees": degrees,
        },
        parameters=dict(parameters or {"microns_per_voxel": [1.0, 1.0, 1.0]}),
    )
    if network_obj.radii is not None:
        radii = np.asarray(network_obj.radii, dtype=float)
        processing_results["vertices"]["radii_microns"] = radii
        processing_results["vertices"]["radii_pixels"] = radii.copy()
    return build_network_json_payload(processing_results)


def write_network_json_fixture(path: Path, *, payload: dict[str, Any]) -> Path:
    """Write a network JSON fixture to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


__all__ = [
    "build_authoritative_network_json_payload",
    "build_export_ready_processing_results",
    "build_minimal_network_json_payload",
    "build_network_object",
    "write_network_json_fixture",
]
