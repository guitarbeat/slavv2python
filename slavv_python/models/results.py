"""Typed models for SLAVV pipeline result payloads."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Mapping


def _copy_mapping(mapping: dict[str, Any] | None) -> dict[str, Any]:
    return dict(mapping) if mapping is not None else {}


def _coerce_array(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    return np.asarray(value, dtype=dtype) if dtype is not None else np.asarray(value)


@dataclass
class EnergyResult:
    """Typed energy-stage payload."""

    energy: np.ndarray
    scale_indices: np.ndarray
    image_shape: tuple[int, ...]
    lumen_radius_pixels: np.ndarray
    lumen_radius_microns: np.ndarray
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EnergyResult:
        payload_copy = _copy_mapping(payload)
        energy = _coerce_array(payload_copy.pop("energy", []), dtype=np.float32)
        scale_indices = _coerce_array(payload_copy.pop("scale_indices", []), dtype=np.int16)
        image_shape = tuple(
            payload_copy.pop("image_shape", tuple(int(value) for value in energy.shape))
        )
        lumen_radius_pixels = _coerce_array(
            payload_copy.pop("lumen_radius_pixels", []), dtype=np.float32
        )
        lumen_radius_microns = _coerce_array(
            payload_copy.pop("lumen_radius_microns", []), dtype=np.float32
        )
        return cls(
            energy=energy,
            scale_indices=scale_indices,
            image_shape=image_shape,
            lumen_radius_pixels=lumen_radius_pixels,
            lumen_radius_microns=lumen_radius_microns,
            extra=payload_copy,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "energy": self.energy.copy(),
            "scale_indices": self.scale_indices.copy(),
            "image_shape": self.image_shape,
            "lumen_radius_pixels": self.lumen_radius_pixels.copy(),
            "lumen_radius_microns": self.lumen_radius_microns.copy(),
            **self.extra,
        }


@dataclass
class VertexSet:
    """Typed vertex payload."""

    positions: np.ndarray
    radii_microns: np.ndarray
    radii_pixels: np.ndarray
    energies: np.ndarray
    scales: np.ndarray
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> VertexSet:
        payload_copy = _copy_mapping(payload)
        positions = _coerce_array(payload_copy.pop("positions", []))
        legacy_radii = _coerce_array(payload_copy.get("radii", []), dtype=np.float32)
        radii_microns = _coerce_array(payload_copy.pop("radii_microns", []), dtype=np.float32)
        radii_pixels = _coerce_array(payload_copy.pop("radii_pixels", []), dtype=np.float32)
        if radii_microns.size == 0 and legacy_radii.size != 0:
            radii_microns = legacy_radii.copy()
        if radii_pixels.size == 0 and legacy_radii.size != 0:
            radii_pixels = legacy_radii.copy()
        energies = _coerce_array(payload_copy.pop("energies", []), dtype=np.float32)
        scales = _coerce_array(payload_copy.pop("scales", []), dtype=np.int16)
        return cls(
            positions=positions,
            radii_microns=radii_microns,
            radii_pixels=radii_pixels,
            energies=energies,
            scales=scales,
            extra=payload_copy,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "positions": self.positions.copy(),
            "radii_microns": self.radii_microns.copy(),
            "radii_pixels": self.radii_pixels.copy(),
            "energies": self.energies.copy(),
            "scales": self.scales.copy(),
            **self.extra,
        }


@dataclass
class EdgeSet:
    """Typed edge payload."""

    traces: list[np.ndarray]
    connections: np.ndarray
    energies: np.ndarray
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EdgeSet:
        payload_copy = _copy_mapping(payload)
        traces = [np.asarray(trace, dtype=float) for trace in payload_copy.pop("traces", [])]
        connections = _coerce_array(payload_copy.pop("connections", []), dtype=np.int32)
        energies = _coerce_array(payload_copy.pop("energies", []), dtype=np.float32)
        return cls(
            traces=traces,
            connections=connections,
            energies=energies,
            extra=payload_copy,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "traces": [trace.copy() for trace in self.traces],
            "connections": self.connections.copy(),
            "energies": self.energies.copy(),
            **self.extra,
        }


@dataclass
class NetworkResult:
    """Typed network payload."""

    strands: list[Any]
    bifurcations: np.ndarray
    vertex_degrees: np.ndarray
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> NetworkResult:
        payload_copy = _copy_mapping(payload)
        strands = list(payload_copy.pop("strands", []))
        bifurcations = _coerce_array(payload_copy.pop("bifurcations", []), dtype=np.int32)
        vertex_degrees = _coerce_array(payload_copy.pop("vertex_degrees", []), dtype=np.int32)
        return cls(
            strands=strands,
            bifurcations=bifurcations,
            vertex_degrees=vertex_degrees,
            extra=payload_copy,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "strands": list(self.strands),
            "bifurcations": self.bifurcations.copy(),
            "vertex_degrees": self.vertex_degrees.copy(),
            **self.extra,
        }


@dataclass
class PipelineResult:
    """Typed wrapper for the full pipeline result payload."""

    parameters: dict[str, Any]
    energy_data: EnergyResult | None = None
    vertices: VertexSet | None = None
    edges: EdgeSet | None = None
    network: NetworkResult | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PipelineResult:
        payload_copy = _copy_mapping(payload)
        parameters = _copy_mapping(payload_copy.pop("parameters", {}))
        energy_payload = payload_copy.pop("energy_data", None)
        vertices_payload = payload_copy.pop("vertices", None)
        edges_payload = payload_copy.pop("edges", None)
        network_payload = payload_copy.pop("network", None)
        return cls(
            parameters=parameters,
            energy_data=(
                EnergyResult.from_dict(energy_payload) if isinstance(energy_payload, dict) else None
            ),
            vertices=VertexSet.from_dict(vertices_payload)
            if isinstance(vertices_payload, dict)
            else None,
            edges=EdgeSet.from_dict(edges_payload) if isinstance(edges_payload, dict) else None,
            network=NetworkResult.from_dict(network_payload)
            if isinstance(network_payload, dict)
            else None,
            extra=payload_copy,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "parameters": dict(self.parameters),
            **self.extra,
        }
        if self.energy_data is not None:
            payload["energy_data"] = self.energy_data.to_dict()
        if self.vertices is not None:
            payload["vertices"] = self.vertices.to_dict()
        if self.edges is not None:
            payload["edges"] = self.edges.to_dict()
        if self.network is not None:
            payload["network"] = self.network.to_dict()
        return payload


def normalize_pipeline_result(processing_results: Mapping[str, Any]) -> PipelineResult:
    """Normalize mapping-like pipeline payloads through the typed adapter."""
    return PipelineResult.from_dict(dict(processing_results))
