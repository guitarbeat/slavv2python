"""Deep module: candidates + Energy/Vertices Stage Results to Edge Set.

Owns the post-Edge Discovery sequence used by EdgeManager and residual scripts:
choose -> optional bridge vertices -> finalize. Watershed Discovery remains a
separate seam; this module does not re-run discovery.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np

from slavv_python.pipeline.edges.bridge_insertion import add_vertices_to_edges_matlab_style
from slavv_python.pipeline.edges.discovery import (
    _use_watershed_discovery,
    resolve_lumen_radius_pixels_axes,
)
from slavv_python.pipeline.edges.finalize import finalize_edges_matlab_style
from slavv_python.pipeline.edges.selection import choose_edges_for_workflow
from slavv_python.pipeline.policy import PipelinePolicy
from slavv_python.schema.results import EdgeSet, EnergyResult, VertexSet

if TYPE_CHECKING:
    from slavv_python.pipeline.edges.candidate_manifest import CandidateManifest


def select_and_finalize_edge_set(
    candidates: CandidateManifest | dict[str, Any] | Any,
    energy_data: EnergyResult,
    vertices: VertexSet,
    params: dict[str, Any],
    *,
    apply_bridge_vertices: bool | None = None,
) -> EdgeSet:
    """Turn an Edge Discovery candidate surface into a finalized Edge Set.

    Args:
        candidates: ``CandidateManifest``, payload dict, or anything
            ``candidate_as_payload`` accepts (e.g. loaded ``candidates.pkl``).
        energy_data: Energy stage result.
        vertices: Vertices stage result (seed set before bridge insertion).
        params: Pipeline parameters (policy, microns, chooser knobs).
        apply_bridge_vertices: If None, follows Exact Route Watershed Discovery
            policy (``comparison_exact_network`` / frontier). If False, skip
            MATLAB-style bridge vertex insertion even on the Exact Route.

    Returns:
        Finalized ``EdgeSet`` after cleanup selection and finalize.
    """
    policy = PipelinePolicy.from_params(params)
    precision = policy.precision
    microns_per_voxel = np.asarray(
        params.get("microns_per_voxel", [1.0, 1.0, 1.0]),
        dtype=precision,
    )
    lumen_radius_microns = energy_data.lumen_radius_microns
    lumen_radius_pixels_axes = resolve_lumen_radius_pixels_axes(
        energy_data,
        microns_per_voxel,
        policy=policy,
    )
    size_of_image = tuple(int(v) for v in energy_data.energy.shape)
    if apply_bridge_vertices is None:
        apply_bridge_vertices = _use_watershed_discovery(energy_data.to_dict(), params)

    chosen = choose_edges_for_workflow(
        candidates,
        vertices.positions,
        vertices.scales,
        lumen_radius_microns,
        lumen_radius_pixels_axes,
        size_of_image,
        params,
        energy_map=energy_data.energy,
        scale_indices=energy_data.scale_indices,
    )
    chosen_payload = (
        chosen.to_dict() if hasattr(chosen, "to_dict") else cast("dict[str, Any]", chosen)
    )

    if apply_bridge_vertices:
        chosen_payload = add_vertices_to_edges_matlab_style(
            chosen_payload,
            vertices.to_dict(),
            energy=energy_data.energy,
            scale_indices=energy_data.scale_indices,
            microns_per_voxel=microns_per_voxel,
            lumen_radius_microns=lumen_radius_microns,
            lumen_radius_pixels_axes=lumen_radius_pixels_axes,
            size_of_image=size_of_image,
            params=params,
        )

    chosen_payload = finalize_edges_matlab_style(
        chosen_payload,
        lumen_radius_microns=lumen_radius_microns,
        microns_per_voxel=microns_per_voxel,
        size_of_image=size_of_image,
    )
    chosen_dict = cast("dict[str, Any]", chosen_payload)
    chosen_dict["lumen_radius_microns"] = np.asarray(
        lumen_radius_microns, dtype=np.float64
    ).copy()
    return EdgeSet.from_dict(chosen_dict)


__all__ = ["select_and_finalize_edge_set"]
