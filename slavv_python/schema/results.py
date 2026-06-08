"""Typed models for SLAVV pipeline result payloads."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path


def _copy_mapping(mapping: dict[str, Any] | None) -> dict[str, Any]:
    return dict(mapping) if mapping is not None else {}


def _coerce_array(value: Any, *, dtype: Any | None = None) -> np.ndarray:
    return np.asarray(value, dtype=dtype) if dtype is not None else np.asarray(value)


def _coerce_energy_array(value: Any) -> np.ndarray:
    energy = np.asarray(value)
    if energy.dtype == np.float64:
        return energy
    return np.asarray(value, dtype=np.float32)


def _coerce_float_metadata_array(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype == np.float64:
        return array
    return np.asarray(value, dtype=np.float32)


@dataclass
class EnergyResult:
    """Typed energy-stage payload with authoritative construction and persistence."""

    energy: np.ndarray
    scale_indices: np.ndarray
    image_shape: tuple[int, ...]
    lumen_radius_pixels: np.ndarray
    lumen_radius_microns: np.ndarray
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        energy: np.ndarray,
        scale_indices: np.ndarray,
        lumen_radius_pixels: np.ndarray,
        lumen_radius_microns: np.ndarray,
        image_shape: tuple[int, ...] | None = None,
        **extra: Any,
    ) -> EnergyResult:
        """Authoritative factory with dtype coercion."""
        energy = _coerce_energy_array(energy)
        scale_indices = _coerce_array(scale_indices, dtype=np.int16)
        if image_shape is None:
            image_shape = energy.shape
        return cls(
            energy=energy,
            scale_indices=scale_indices,
            image_shape=image_shape,
            lumen_radius_pixels=_coerce_float_metadata_array(lumen_radius_pixels),
            lumen_radius_microns=_coerce_float_metadata_array(lumen_radius_microns),
            extra=extra,
        )

    def save(self, path: Path | str) -> None:
        """Persist energy data using atomic joblib dump."""
        from slavv_python.engine.state.io import atomic_joblib_dump

        atomic_joblib_dump(self.to_dict(), path)

    @classmethod
    def load(cls, path: Path | str) -> EnergyResult:
        """Load energy data from persistent storage."""
        from slavv_python.utils.safe_unpickle import safe_load

        payload = safe_load(path)
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EnergyResult:
        payload_copy = _copy_mapping(payload)
        energy = _coerce_energy_array(payload_copy.pop("energy", []))
        scale_indices = _coerce_array(payload_copy.pop("scale_indices", []), dtype=np.int16)
        image_shape = tuple(
            payload_copy.pop("image_shape", tuple(int(value) for value in energy.shape))
        )
        lumen_radius_pixels = _coerce_float_metadata_array(
            payload_copy.pop("lumen_radius_pixels", [])
        )
        lumen_radius_microns = _coerce_float_metadata_array(
            payload_copy.pop("lumen_radius_microns", [])
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
    """Typed vertex payload with authoritative construction and persistence."""

    positions: np.ndarray
    radii_microns: np.ndarray
    radii_pixels: np.ndarray
    energies: np.ndarray
    scales: np.ndarray
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        vertex_positions: np.ndarray,
        vertex_scales: np.ndarray,
        vertex_energies: np.ndarray,
        lumen_radius_pixels: np.ndarray,
        lumen_radius_microns: np.ndarray,
        **extra: Any,
    ) -> VertexSet:
        """Authoritative factory that handles dtype coercion and shape validation."""
        positions = _coerce_array(vertex_positions, dtype=np.float32)
        scales = _coerce_array(vertex_scales, dtype=np.int16)
        energies = _coerce_array(vertex_energies, dtype=np.float32)

        if len(positions) != len(scales) or len(positions) != len(energies):
            raise ValueError(
                f"Vertex attribute mismatch: positions({len(positions)}), "
                f"scales({len(scales)}), energies({len(energies)})"
            )

        radii_pixels = _coerce_array(lumen_radius_pixels[scales], dtype=np.float32)
        radii_microns = _coerce_array(lumen_radius_microns[scales], dtype=np.float32)

        return cls(
            positions=positions,
            radii_microns=radii_microns,
            radii_pixels=radii_pixels,
            energies=energies,
            scales=scales,
            extra=extra,
        )

    def save(self, path: Path | str) -> None:
        """Persist the vertex set using atomic joblib dump."""
        from slavv_python.engine.state.io import atomic_joblib_dump

        atomic_joblib_dump(self.to_dict(), path)

    @classmethod
    def load(cls, path: Path | str) -> VertexSet:
        """Load a vertex set from a joblib-persisted dict."""
        from slavv_python.utils.safe_unpickle import safe_load

        payload = safe_load(path)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected dict at {path}, got {type(payload)}")
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> VertexSet:
        payload_copy = _copy_mapping(payload)
        positions = _coerce_array(payload_copy.pop("positions", []), dtype=np.float32)
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
    """Typed edge payload with authoritative construction and persistence."""

    traces: list[np.ndarray]
    connections: np.ndarray
    energies: np.ndarray
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        traces: list[np.ndarray],
        connections: np.ndarray,
        energies: np.ndarray,
        **extra: Any,
    ) -> EdgeSet:
        """Authoritative factory with validation and coercion."""
        traces = [np.asarray(trace, dtype=np.float32) for trace in traces]
        connections = _coerce_array(connections, dtype=np.int32)
        energies = _coerce_array(energies, dtype=np.float32)
        return cls(
            traces=traces,
            connections=connections,
            energies=energies,
            extra=extra,
        )

    def save(self, path: Path | str) -> None:
        """Persist the edge set using atomic joblib dump."""
        from slavv_python.engine.state.io import atomic_joblib_dump

        atomic_joblib_dump(self.to_dict(), path)

    @classmethod
    def load(cls, path: Path | str) -> EdgeSet:
        """Load an edge set from persistent storage."""
        from slavv_python.utils.safe_unpickle import safe_load

        payload = safe_load(path)
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EdgeSet:
        payload_copy = _copy_mapping(payload)
        traces = [np.asarray(trace, dtype=np.float32) for trace in payload_copy.pop("traces", [])]
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
    """Typed network payload with authoritative construction and persistence."""

    strands: list[Any]
    bifurcations: np.ndarray
    vertex_degrees: np.ndarray
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        strands: list[Any],
        bifurcations: np.ndarray,
        vertex_degrees: np.ndarray,
        **extra: Any,
    ) -> NetworkResult:
        """Authoritative factory with validation."""
        return cls(
            strands=list(strands),
            bifurcations=_coerce_array(bifurcations, dtype=np.int32),
            vertex_degrees=_coerce_array(vertex_degrees, dtype=np.int32),
            extra=extra,
        )

    def save(self, path: Path | str) -> None:
        """Persist the network result using atomic joblib dump."""
        from slavv_python.engine.state.io import atomic_joblib_dump

        atomic_joblib_dump(self.to_dict(), path)

    @classmethod
    def load(cls, path: Path | str) -> NetworkResult:
        """Load a network result from persistent storage."""
        from slavv_python.utils.safe_unpickle import safe_load

        payload = safe_load(path)
        return cls.from_dict(payload)

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
class PipelineResult(Mapping):
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
                energy_payload
                if isinstance(energy_payload, EnergyResult)
                else EnergyResult.from_dict(energy_payload)
                if isinstance(energy_payload, dict)
                else None
            ),
            vertices=(
                vertices_payload
                if isinstance(vertices_payload, VertexSet)
                else VertexSet.from_dict(vertices_payload)
                if isinstance(vertices_payload, dict)
                else None
            ),
            edges=(
                edges_payload
                if isinstance(edges_payload, EdgeSet)
                else EdgeSet.from_dict(edges_payload)
                if isinstance(edges_payload, dict)
                else None
            ),
            network=(
                network_payload
                if isinstance(network_payload, NetworkResult)
                else NetworkResult.from_dict(network_payload)
                if isinstance(network_payload, dict)
                else None
            ),
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

    def __getitem__(self, key: str) -> Any:
        return self.to_dict()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_dict())

    def __len__(self) -> int:
        return len(self.to_dict())


def normalize_pipeline_result(processing_results: Mapping[str, Any]) -> PipelineResult:
    """Normalize mapping-like pipeline payloads through the typed adapter."""
    return PipelineResult.from_dict(dict(processing_results))
