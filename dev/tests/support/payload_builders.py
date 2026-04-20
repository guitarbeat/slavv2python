"""Reusable payload builders for tests."""

from __future__ import annotations

from typing import Any

import numpy as np


def _coerce_positions(positions: Any) -> np.ndarray:
    array = np.asarray(positions, dtype=float)
    if array.size == 0:
        return np.empty((0, 3), dtype=float)
    array = np.atleast_2d(array)
    if array.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    return array


def _coerce_connections(connections: Any) -> np.ndarray:
    array = np.asarray(connections, dtype=int)
    if array.size == 0:
        return np.empty((0, 2), dtype=int)
    array = np.atleast_2d(array)
    if array.shape[1] != 2:
        raise ValueError("connections must have shape (N, 2)")
    return array


def _coerce_vector(
    values: Any | None,
    *,
    size: int,
    default: float,
    dtype: Any,
) -> np.ndarray:
    if values is None:
        return np.full((size,), default, dtype=dtype)
    array = np.asarray(values, dtype=dtype).reshape(-1)
    if size == 0 and array.size == 0:
        return array
    if array.size == 1 and size > 1:
        return np.repeat(array, size).astype(dtype, copy=False)
    if array.size != size:
        raise ValueError(f"expected vector of size {size}, got {array.size}")
    return array


def _copy_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
    return dict(payload) if payload is not None else {}


def build_energy_result(
    *,
    energy: Any | None = None,
    scale_indices: Any | None = None,
    image_shape: tuple[int, int, int] = (4, 4, 4),
    lumen_radius_pixels: Any | None = None,
    lumen_radius_microns: Any | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a minimal energy-stage payload for tests."""
    if energy is None:
        energy_array = np.zeros(image_shape, dtype=np.float32)
    else:
        energy_array = np.asarray(energy, dtype=np.float32)
        image_shape = tuple(int(value) for value in energy_array.shape)

    if scale_indices is None:
        scale_array = np.zeros(image_shape, dtype=np.int16)
    else:
        scale_array = np.asarray(scale_indices, dtype=np.int16)

    payload = {
        "energy": energy_array,
        "scale_indices": scale_array,
        "image_shape": image_shape,
        "lumen_radius_pixels": np.asarray(
            lumen_radius_pixels if lumen_radius_pixels is not None else [1.0],
            dtype=np.float32,
        ).reshape(-1),
        "lumen_radius_microns": np.asarray(
            lumen_radius_microns if lumen_radius_microns is not None else [1.0],
            dtype=np.float32,
        ).reshape(-1),
        "energy_sign": -1.0,
    }
    payload.update(_copy_payload(overrides))
    return payload


def build_vertices_payload(
    *,
    positions: Any | None = None,
    radii_microns: Any | None = None,
    radii_pixels: Any | None = None,
    energies: Any | None = None,
    scales: Any | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a minimal vertices payload for tests."""
    pos = _coerce_positions(
        positions
        if positions is not None
        else [[0.0, 0.0, 0.0], [0.0, 4.0, 0.0], [3.0, 4.0, 0.0]]
    )
    count = int(pos.shape[0])
    radii_microns_array = _coerce_vector(
        radii_microns,
        size=count,
        default=1.0,
        dtype=np.float32,
    )
    radii_pixels_array = _coerce_vector(
        radii_pixels,
        size=count,
        default=1.0,
        dtype=np.float32,
    )
    payload = {
        "positions": pos,
        "radii_microns": radii_microns_array,
        "radii_pixels": radii_pixels_array,
        "radii": radii_microns_array.copy(),
        "energies": _coerce_vector(
            energies,
            size=count,
            default=-1.0,
            dtype=np.float32,
        ),
        "scales": _coerce_vector(
            scales,
            size=count,
            default=0,
            dtype=np.int16,
        ),
        "count": count,
    }
    payload.update(_copy_payload(overrides))
    return payload


def build_edges_payload(
    *,
    traces: list[Any] | None = None,
    connections: Any | None = None,
    energies: Any | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a minimal edges payload for tests."""
    trace_arrays = [
        np.asarray(trace, dtype=float)
        for trace in (
            traces
            if traces is not None
            else [
                [[0.0, 0.0, 0.0], [0.0, 4.0, 0.0]],
                [[0.0, 4.0, 0.0], [3.0, 4.0, 0.0]],
            ]
        )
    ]
    payload = {
        "traces": trace_arrays,
        "connections": _coerce_connections(
            connections if connections is not None else [[0, 1], [1, 2]]
        ),
        "energies": _coerce_vector(
            energies,
            size=len(trace_arrays),
            default=-1.0,
            dtype=np.float32,
        ),
    }
    payload.update(_copy_payload(overrides))
    return payload


def build_network_payload(
    *,
    strands: list[list[int]] | None = None,
    bifurcations: Any | None = None,
    vertex_degrees: Any | None = None,
    edges: dict[str, Any] | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a minimal network payload for tests."""
    default_strands = strands if strands is not None else [[0, 1, 2]]

    degree_array = None
    if vertex_degrees is not None:
        degree_array = np.asarray(vertex_degrees, dtype=int).reshape(-1)
    elif edges is not None:
        edge_connections = _coerce_connections(edges.get("connections", []))
        if edge_connections.size:
            vertex_count = int(edge_connections.max()) + 1
            degree_array = np.zeros((vertex_count,), dtype=int)
            for start, end in edge_connections:
                degree_array[int(start)] += 1
                degree_array[int(end)] += 1
    if degree_array is None:
        degree_array = np.array([1, 2, 1], dtype=int)

    payload = {
        "strands": default_strands,
        "bifurcations": np.asarray(
            bifurcations if bifurcations is not None else [],
            dtype=int,
        ),
        "vertex_degrees": degree_array,
    }
    payload.update(_copy_payload(overrides))
    return payload


def build_processing_results(
    *,
    vertices: dict[str, Any] | None = None,
    edges: dict[str, Any] | None = None,
    network: dict[str, Any] | None = None,
    parameters: dict[str, Any] | None = None,
    energy_data: dict[str, Any] | None = None,
    vertex_positions: Any | None = None,
    edge_traces: list[Any] | None = None,
    edge_connections: Any | None = None,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a complete processing results payload for tests."""
    vertex_payload = (
        build_vertices_payload(positions=vertex_positions) if vertices is None else dict(vertices)
    )
    edge_payload = (
        build_edges_payload(traces=edge_traces, connections=edge_connections)
        if edges is None
        else dict(edges)
    )
    network_payload = (
        build_network_payload(edges=edge_payload) if network is None else dict(network)
    )
    payload = {
        "vertices": vertex_payload,
        "edges": edge_payload,
        "network": network_payload,
        "parameters": dict(parameters or {"microns_per_voxel": [1.0, 1.0, 1.0]}),
        "energy_data": energy_data if energy_data is not None else build_energy_result(),
    }
    payload.update(_copy_payload(overrides))
    return payload


__all__ = [
    "build_edges_payload",
    "build_energy_result",
    "build_network_payload",
    "build_processing_results",
    "build_vertices_payload",
]
