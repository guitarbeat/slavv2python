"""Edge discovery strategy seam: tracing, frontier, and manifest typing."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from typing_extensions import Protocol

import numpy as np
from scipy.spatial import cKDTree

from slavv_python.pipeline.energy.provenance import is_exact_compatible_energy_origin
from slavv_python.pipeline.edges.candidate_generation import (
    generate_directional_candidates,
    generate_watershed_candidates,
    sort_candidates_by_quality,
)
from slavv_python.pipeline.vertices.painting import paint_vertex_image

if TYPE_CHECKING:
    from collections.abc import Callable

    from slavv_python.engine.state import StageController
    from slavv_python.schema.results import EnergyResult, VertexSet

logger = logging.getLogger(__name__)


from slavv_python.pipeline.policy import PipelinePolicy


def _use_matlab_frontier_tracer(energy_data: dict[str, Any], params: dict[str, Any]) -> bool:
    """Enable the MATLAB-style frontier tracer for exact-compatible energy reruns."""
    policy = PipelinePolicy.from_params(params)
    if policy.internal_grid_alignment != "matlab":
        return False
    return is_exact_compatible_energy_origin(energy_data.get("energy_origin"))


def resolve_lumen_radius_pixels_axes(
    energy_data: Any,
    microns_per_voxel: np.ndarray,
    policy: PipelinePolicy | None = None,
) -> np.ndarray:
    """Return per-axis pixel radii for modern and legacy Energy checkpoints."""
    dtype = policy.precision if policy else np.float64
    raw_axes = energy_data.extra.get("lumen_radius_pixels_axes")
    if raw_axes is not None:
        return np.asarray(raw_axes, dtype=dtype)

    lumen_radius_pixels = np.asarray(energy_data.lumen_radius_pixels, dtype=dtype)
    if lumen_radius_pixels.size > 0:
        return np.repeat(lumen_radius_pixels.reshape(-1, 1), 3, axis=1)

    lumen_radius_microns = np.asarray(energy_data.lumen_radius_microns, dtype=dtype)
    if lumen_radius_microns.size == 0:
        return np.zeros((0, 3), dtype=dtype)
    voxel_size = np.asarray(microns_per_voxel, dtype=dtype).reshape(1, 3)
    return (lumen_radius_microns.reshape(-1, 1) / voxel_size).astype(dtype)


@dataclass
class CandidateManifest:
    """Typed edge-candidate payload produced by a discovery strategy."""

    traces: list[np.ndarray] = field(default_factory=list)
    connections: np.ndarray = field(default_factory=lambda: np.zeros((0, 2), dtype=np.int32))
    metrics: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.float64))
    energy_traces: list[np.ndarray] = field(default_factory=list)
    scale_traces: list[np.ndarray] = field(default_factory=list)
    origin_indices: np.ndarray = field(default_factory=lambda: np.zeros((0,), dtype=np.int32))
    connection_sources: list[str] = field(default_factory=list)
    frontier_lifecycle_events: list[dict[str, Any]] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def empty(cls) -> CandidateManifest:
        return cls()

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> CandidateManifest:
        payload_copy = dict(payload)
        traces = [np.asarray(trace, dtype=np.float64) for trace in payload_copy.pop("traces", [])]
        connections = np.asarray(payload_copy.pop("connections", []), dtype=np.int32).reshape(-1, 2)
        metrics = np.asarray(payload_copy.pop("metrics", []), dtype=np.float64).reshape(-1)
        energy_traces = [
            np.asarray(trace, dtype=np.float64) for trace in payload_copy.pop("energy_traces", [])
        ]
        scale_traces = [
            np.asarray(trace, dtype=np.int16) for trace in payload_copy.pop("scale_traces", [])
        ]
        origin_indices = np.asarray(payload_copy.pop("origin_indices", []), dtype=np.int32).reshape(
            -1
        )
        connection_sources = [str(value) for value in payload_copy.pop("connection_sources", [])]
        frontier_lifecycle_events = list(payload_copy.pop("frontier_lifecycle_events", []) or [])
        diagnostics = dict(payload_copy.pop("diagnostics", {}) or {})
        return cls(
            traces=traces,
            connections=connections,
            metrics=metrics,
            energy_traces=energy_traces,
            scale_traces=scale_traces,
            origin_indices=origin_indices,
            connection_sources=connection_sources,
            frontier_lifecycle_events=frontier_lifecycle_events,
            diagnostics=diagnostics,
            extra=payload_copy,
        )

    def to_payload(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "traces": self.traces,
            "connections": self.connections,
            "metrics": self.metrics,
            "energy_traces": self.energy_traces,
            "scale_traces": self.scale_traces,
            "origin_indices": self.origin_indices,
            "connection_sources": self.connection_sources,
            "diagnostics": self.diagnostics,
            **self.extra,
        }
        if self.frontier_lifecycle_events:
            payload["frontier_lifecycle_events"] = self.frontier_lifecycle_events
        return payload


@dataclass
class EdgeDiscoveryContext:
    """Inputs shared by all edge discovery strategies."""

    energy_data: EnergyResult
    vertices: VertexSet
    params: dict[str, Any]
    stage_controller: StageController
    vertex_center_image: np.ndarray
    lumen_radius_pixels_axes: np.ndarray
    microns_per_voxel: np.ndarray
    heartbeat: Callable[[int, int], None] | None = None


class EdgeDiscovery(Protocol):
    """Strategy interface for generating edge candidates."""

    def discover(self, context: EdgeDiscoveryContext) -> CandidateManifest: ...


class MaintainedTracingDiscovery:
    """Maintained tracing workflow with painted vertex occupancy."""

    def discover(self, context: EdgeDiscoveryContext) -> CandidateManifest:
        energy_data = context.energy_data
        vertices = context.vertices
        energy = energy_data.energy
        vertex_positions = vertices.positions
        vertex_scales = vertices.scales
        lumen_radius_microns = energy_data.lumen_radius_microns
        energy_sign = float(energy_data.extra.get("energy_sign", -1.0))

        vertex_image = paint_vertex_image(
            vertex_positions,
            vertex_scales,
            context.lumen_radius_pixels_axes,
            energy.shape,
        )
        tree = cKDTree(vertex_positions * context.microns_per_voxel)
        max_vertex_radius = (
            float(np.max(lumen_radius_microns)) if len(lumen_radius_microns) > 0 else 0.0
        )
        payload = generate_directional_candidates(
            energy=energy,
            scale_indices=energy_data.scale_indices,
            vertex_positions=vertex_positions,
            vertex_scales=vertex_scales,
            lumen_radius_pixels=energy_data.lumen_radius_pixels,
            lumen_radius_microns=lumen_radius_microns,
            microns_per_voxel=context.microns_per_voxel,
            vertex_center_image=context.vertex_center_image,
            vertex_image=vertex_image,
            tree=tree,
            max_search_radius=max_vertex_radius * 5.0,
            params=context.params,
            energy_sign=energy_sign,
        )
        return CandidateManifest.from_payload(payload)


class FrontierTracingDiscovery:
    """MATLAB-parity frontier tracer with post-finalize supplementation."""

    def discover(self, context: EdgeDiscoveryContext) -> CandidateManifest:
        energy_data = context.energy_data
        vertices = context.vertices
        energy = energy_data.energy
        vertex_positions = vertices.positions
        vertex_scales = vertices.scales
        lumen_radius_microns = energy_data.lumen_radius_microns
        energy_sign = float(energy_data.extra.get("energy_sign", -1.0))

        payload = generate_watershed_candidates(
            energy,
            energy_data.scale_indices,
            vertex_positions,
            vertex_scales,
            lumen_radius_microns,
            context.microns_per_voxel,
            context.vertex_center_image,
            context.params,
            heartbeat=context.heartbeat,
        )
        payload = sort_candidates_by_quality(
            payload,
            energy,
            energy_data.scale_indices,
            vertex_positions,
            energy_sign,
            context.params,
            context.microns_per_voxel,
        )
        return CandidateManifest.from_payload(payload)


def select_edge_discovery(energy_data: EnergyResult, params: dict[str, Any]) -> EdgeDiscovery:
    """Pick the discovery strategy for the current energy/parameters profile."""
    if _use_matlab_frontier_tracer(energy_data.to_dict(), params):
        return FrontierTracingDiscovery()
    return MaintainedTracingDiscovery()


def frontier_origin_counts(manifest: CandidateManifest) -> dict[int, int]:
    """Count frontier-sourced candidates per origin index."""
    counts: dict[int, int] = {}
    for index, origin_index in enumerate(manifest.origin_indices.reshape(-1)):
        if index >= len(manifest.connection_sources):
            continue
        if manifest.connection_sources[index] != "frontier":
            continue
        origin_key = int(origin_index)
        counts[origin_key] = counts.get(origin_key, 0) + 1
    return counts


def frontier_origin_counts_from_diagnostics(manifest: CandidateManifest) -> dict[int, int]:
    """Read normalized frontier per-origin counts from discovery diagnostics."""
    from slavv_python.pipeline.edges.audit import _normalize_candidate_origin_counts

    raw = manifest.diagnostics.get("frontier_per_origin_candidate_counts")
    normalized = _normalize_candidate_origin_counts(raw)
    return {int(origin_index): int(count) for origin_index, count in normalized.items()}


__all__ = [
    "CandidateManifest",
    "EdgeDiscovery",
    "EdgeDiscoveryContext",
    "FrontierTracingDiscovery",
    "MaintainedTracingDiscovery",
    "_use_matlab_frontier_tracer",
    "frontier_origin_counts",
    "frontier_origin_counts_from_diagnostics",
    "resolve_lumen_radius_pixels_axes",
    "select_edge_discovery",
]
