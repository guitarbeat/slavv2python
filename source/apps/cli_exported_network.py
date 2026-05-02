"""Helpers for loading exported network JSON payloads."""

from __future__ import annotations

from typing import cast

import networkx as nx
import numpy as np

from source.io.network_json import (
    infer_image_shape_from_vertices as _infer_image_shape_from_vertices,
)
from source.io.network_json import (
    load_network_json_payload,
)

from .cli_shared import _require_existing_file


def _normalize_exported_edge_connections(raw_connections: object) -> np.ndarray:
    """Normalize exported edge connections into a 2-column integer array."""
    connections = np.asarray(raw_connections, dtype=int)
    if connections.size == 0:
        empty_connections: np.ndarray = np.empty((0, 2), dtype=int)
        return empty_connections
    connections = np.atleast_2d(connections)
    if connections.shape[1] < 2:
        insufficient_connections: np.ndarray = np.empty((0, 2), dtype=int)
        return insufficient_connections
    return cast("np.ndarray", np.asarray(connections[:, :2], dtype=int))


def _build_strands_from_edge_connections(
    edge_connections: np.ndarray, *, vertex_count: int
) -> list[list[int]]:
    """Reconstruct strand-like paths from exported edge connections."""
    if vertex_count == 0 or edge_connections.size == 0:
        return []

    graph = nx.Graph()
    graph.add_nodes_from(range(vertex_count))
    for origin_idx, destination_idx in edge_connections:
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
        edge = (min(origin, destination), max(origin, destination))
        if edge in visited_edges:
            continue

        strand = [origin, destination]
        visited_edges.add(edge)

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


def _load_exported_network_json(path: str) -> dict:
    """Load exported JSON and rebuild the stats inputs expected by analysis helpers."""
    data = load_network_json_payload(path)
    vertices = cast("dict", data.get("vertices", {}))
    edges = cast("dict", data.get("edges", {}))
    network = cast("dict", data.get("network", {}))

    vertex_positions = np.asarray(vertices.get("positions", []), dtype=float)
    edge_connections = _normalize_exported_edge_connections(edges.get("connections", []))
    vertex_radii = np.asarray(vertices.get("radii_microns", []), dtype=float)
    if len(vertex_radii) != len(vertex_positions):
        vertex_radii = np.zeros(len(vertex_positions), dtype=float)

    if not network.get("strands"):
        network["strands"] = _build_strands_from_edge_connections(
            edge_connections,
            vertex_count=len(vertex_positions),
        )
    if len(np.asarray(network.get("bifurcations", []), dtype=int)) == 0:
        graph = nx.Graph()
        graph.add_nodes_from(range(len(vertex_positions)))
        graph.add_edges_from(edge_connections.tolist())
        network["bifurcations"] = np.fromiter(
            (node for node, degree in graph.degree() if degree > 2),
            dtype=int,
        )

    return {
        "metadata": data.get("metadata", {}),
        "vertices": {
            **vertices,
            "positions": vertex_positions,
            "radii_microns": vertex_radii,
        },
        "edges": {
            **edges,
            "connections": edge_connections,
        },
        "network": network,
        "parameters": data.get("parameters", {}),
        "summary": data.get("summary", {}),
        "image_shape": tuple(
            data.get("image_shape", _infer_image_shape_from_vertices(vertex_positions))
        ),
    }


def _load_exported_results(input_path: str) -> dict:
    """Validate and load exported JSON results for analyze/plot commands."""
    _require_existing_file(input_path)
    return _load_exported_network_json(input_path)
