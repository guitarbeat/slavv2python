"""Edge Discovery strategy seam (ADR 0005).

Domain strategies (AGENTS glossary):

* **Tracing Discovery** — Paper Path directional centerline tracing
  (``TracingDiscovery``; legacy alias ``MaintainedTracingDiscovery``).
* **Watershed Discovery** — Exact Route MATLAB global watershed
  (``WatershedDiscovery``; legacy alias ``FrontierTracingDiscovery``).

Engine entry for Watershed Discovery is ``generate_watershed_candidates`` →
``matlab_get_edges_by_watershed``. That is the Certification path.

**Not** Edge Discovery for Certification: the skimage label-adjacency drivers in
``naive_watershed`` / ``extract_edges_watershed``.

Candidate payload ownership lives in ``candidate_manifest.CandidateManifest``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np
from scipy.spatial import cKDTree
from typing_extensions import Protocol

from slavv_python.pipeline.edges.audit import _normalize_candidate_origin_counts
from slavv_python.pipeline.edges.candidate_generation import (
    generate_directional_candidates,
    generate_watershed_candidates,
    sort_candidates_by_quality,
)
from slavv_python.pipeline.edges.candidate_manifest import (
    CandidateManifest,
    candidate_as_payload,
)
from slavv_python.pipeline.energy.provenance import is_exact_compatible_energy_origin
from slavv_python.pipeline.policy import PipelinePolicy
from slavv_python.pipeline.vertices.painting import paint_vertex_image

if TYPE_CHECKING:
    from collections.abc import Callable

    from slavv_python.engine.state import StageController
    from slavv_python.schema.results import EnergyResult, VertexSet

logger = logging.getLogger(__name__)


def _use_watershed_discovery(energy_data: dict[str, Any], params: dict[str, Any]) -> bool:
    """Return True when Exact Route Watershed Discovery should run.

    Requires MATLAB grid alignment and exact-compatible energy provenance.
    """
    policy = PipelinePolicy.from_params(params)
    if policy.internal_grid_alignment != "matlab":
        return False
    return is_exact_compatible_energy_origin(energy_data.get("energy_origin"))


# Legacy name — same predicate as ``_use_watershed_discovery``.
_use_matlab_frontier_tracer = _use_watershed_discovery


def resolve_lumen_radius_pixels_axes(
    energy_data: Any,
    microns_per_voxel: np.ndarray,
    policy: PipelinePolicy | None = None,
) -> np.ndarray:
    """Return per-axis pixel radii for modern and legacy Energy checkpoints."""
    dtype = policy.precision if policy else np.float64
    raw_axes = energy_data.extra.get("lumen_radius_pixels_axes")
    if raw_axes is not None:
        return cast("np.ndarray", np.asarray(raw_axes, dtype=dtype))

    lumen_radius_pixels = np.asarray(energy_data.lumen_radius_pixels, dtype=dtype)
    if lumen_radius_pixels.size > 0:
        return cast("np.ndarray", np.repeat(lumen_radius_pixels.reshape(-1, 1), 3, axis=1))

    lumen_radius_microns = np.asarray(energy_data.lumen_radius_microns, dtype=dtype)
    if lumen_radius_microns.size == 0:
        return np.zeros((0, 3), dtype=dtype)
    voxel_size = np.asarray(microns_per_voxel, dtype=dtype).reshape(1, 3)
    return cast("np.ndarray", (lumen_radius_microns.reshape(-1, 1) / voxel_size).astype(dtype))


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


class TracingDiscovery:
    """Tracing Discovery (Paper Path): directional centerline candidates.

    Paints vertex occupancy and expands via maintained directional tracing.
    """

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


class WatershedDiscovery:
    """Watershed Discovery (Exact Route): MATLAB global watershed candidates.

    Implements Certification Edge Discovery via ``generate_watershed_candidates``.
    Not the skimage label-adjacency path in ``extract_edges_watershed``.
    """

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
        return CandidateManifest.from_payload(
            sort_candidates_by_quality(
                payload,
                energy,
                energy_data.scale_indices,
                vertex_positions,
                energy_sign,
                context.params,
                context.microns_per_voxel,
            )
        )


# Legacy class names — prefer TracingDiscovery / WatershedDiscovery.
MaintainedTracingDiscovery = TracingDiscovery
FrontierTracingDiscovery = WatershedDiscovery


def select_edge_discovery(energy_data: EnergyResult, params: dict[str, Any]) -> EdgeDiscovery:
    """Pick Tracing Discovery or Watershed Discovery for this energy/params profile."""
    if _use_watershed_discovery(energy_data.to_dict(), params):
        return WatershedDiscovery()
    return TracingDiscovery()


def frontier_origin_counts(manifest: CandidateManifest) -> dict[int, int]:
    """Count candidates tagged connection_source ``frontier`` per origin index.

    The wire-format source string remains ``\"frontier\"`` for checkpoint compatibility;
    those candidates are produced by Watershed Discovery.
    """
    return manifest.frontier_origin_counts()


def frontier_origin_counts_from_diagnostics(manifest: CandidateManifest) -> dict[int, int]:
    """Read per-origin Watershed Discovery counts from discovery diagnostics."""
    raw = manifest.diagnostics.get("frontier_per_origin_candidate_counts")
    normalized = _normalize_candidate_origin_counts(raw)
    return {int(origin_index): int(count) for origin_index, count in normalized.items()}


__all__ = [
    "CandidateManifest",
    "EdgeDiscovery",
    "EdgeDiscoveryContext",
    "FrontierTracingDiscovery",  # legacy alias of WatershedDiscovery
    "MaintainedTracingDiscovery",  # legacy alias of TracingDiscovery
    "TracingDiscovery",
    "WatershedDiscovery",
    "_use_matlab_frontier_tracer",  # legacy alias of _use_watershed_discovery
    "_use_watershed_discovery",
    "candidate_as_payload",
    "frontier_origin_counts",
    "frontier_origin_counts_from_diagnostics",
    "resolve_lumen_radius_pixels_axes",
    "select_edge_discovery",
]
