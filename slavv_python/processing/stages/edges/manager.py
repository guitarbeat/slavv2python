"""Consolidated edge extraction manager."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from slavv_python.processing.stages.edges.audit import (
    _normalize_candidate_origin_counts,
)
from slavv_python.processing.stages.edges.bridge_insertion import add_vertices_to_edges_matlab_style
from slavv_python.processing.stages.edges.candidate_generation import (
    _finalize_matlab_parity_candidates,
    _generate_edge_candidates,
    _generate_edge_candidates_matlab_frontier,
)
from slavv_python.processing.stages.edges.common import _use_matlab_frontier_tracer
from slavv_python.processing.stages.edges.finalize import finalize_edges_matlab_style
from slavv_python.processing.stages.edges.selection import choose_edges_for_workflow
from slavv_python.processing.stages.vertices.painting import (
    paint_vertex_center_image,
    paint_vertex_image,
)
from slavv_python.schema.results import EdgeSet, EnergyResult, VertexSet

if TYPE_CHECKING:
    from slavv_python.engine.state import StageController

logger = logging.getLogger(__name__)


class EdgeManager:
    """Consolidated manager for the edge discovery and selection lifecycle."""

    @classmethod
    def run_resumable(
        cls,
        energy_data: EnergyResult,
        vertices: VertexSet,
        params: dict[str, Any],
        stage_controller: StageController,
    ) -> EdgeSet:
        """Execute the full edge extraction lifecycle with resumability."""
        from slavv_python.engine.state.tracker import atomic_joblib_dump

        energy = energy_data.energy
        vertex_positions = vertices.positions
        vertex_scales = vertices.scales
        lumen_radius_microns = energy_data.lumen_radius_microns
        scale_indices = energy_data.scale_indices
        energy_sign = energy_data.extra.get("energy_sign", -1.0)
        microns_per_voxel = np.array(params.get("microns_per_voxel", [1.0, 1.0, 1.0]), dtype=float)

        if len(vertex_positions) == 0:
            return EdgeSet.create([], np.empty((0, 2), dtype=np.int32), np.empty((0,), dtype=np.float32))

        # 1. Setup paths and images
        candidate_manifest_path = stage_controller.artifact_path("candidates.pkl")
        candidate_audit_path = stage_controller.artifact_path("candidate_audit.json")
        chosen_manifest_path = stage_controller.artifact_path("chosen_edges.pkl")

        lumen_radius_pixels_axes = np.asarray(
            energy_data.extra.get(
                "lumen_radius_pixels_axes",
                np.repeat(
                    np.asarray(energy_data.lumen_radius_pixels, dtype=np.float32).reshape(-1, 1),
                    3,
                    axis=1,
                ),
            ),
            dtype=np.float32,
        )

        vertex_center_image = paint_vertex_center_image(vertex_positions, energy.shape)
        use_frontier = _use_matlab_frontier_tracer(energy_data.to_dict(), params)

        # 2. Candidate Generation
        stage_controller.begin(
            detail="Generating edge candidates",
            units_total=3,
            substage="generate_candidates",
        )

        if use_frontier:
            def _heartbeat(iteration_count: int, candidate_count: int) -> None:
                stage_controller.update(
                    units_total=3,
                    substage="generate_candidates",
                    detail=f"Tracing frontier (iters={iteration_count}, candidates={candidate_count})",
                )

            candidates = _generate_edge_candidates_matlab_frontier(
                energy,
                scale_indices,
                vertex_positions,
                vertex_scales,
                lumen_radius_microns,
                microns_per_voxel,
                vertex_center_image,
                params,
                heartbeat=_heartbeat,
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

            # Audit processing
            raw_origin_counts = _normalize_candidate_origin_counts(
                candidates.get("diagnostics", {}).get("frontier_per_origin_candidate_counts")
            )
            frontier_origin_counts = {int(k): int(v) for k, v in raw_origin_counts.items()}
        else:
            from scipy.spatial import cKDTree
            tree = cKDTree(vertex_positions * microns_per_voxel)
            vertex_image = paint_vertex_image(vertex_positions, vertex_scales, lumen_radius_pixels_axes, energy.shape)

            candidates = _generate_edge_candidates(
                energy=energy,
                scale_indices=scale_indices,
                vertex_positions=vertex_positions,
                vertex_scales=vertex_scales,
                lumen_radius_pixels=energy_data.lumen_radius_pixels,
                lumen_radius_microns=lumen_radius_microns,
                microns_per_voxel=microns_per_voxel,
                vertex_center_image=vertex_center_image,
                vertex_image=vertex_image,
                tree=tree,
                max_search_radius=np.max(lumen_radius_microns) * 5.0,
                params=params,
                energy_sign=energy_sign,
            )

            # Simplified audit for non-frontier
            frontier_origin_counts = {} # TODO: implement if needed

        # 3. Persistence & Audit
        atomic_joblib_dump(candidates, candidate_manifest_path)
        stage_controller.update(units_total=3, units_completed=1, substage="generate_candidates")

        # 4. Selection (Conflict Painting)
        chosen = choose_edges_for_workflow(
            candidates,
            vertex_positions,
            vertex_scales,
            lumen_radius_microns,
            lumen_radius_pixels_axes,
            energy.shape,
            params,
        )

        # 5. Bridging (Structural Vertex Insertion)
        if use_frontier:
            chosen = add_vertices_to_edges_matlab_style(
                chosen,
                vertices.to_dict(),
                energy=energy,
                scale_indices=scale_indices,
                microns_per_voxel=microns_per_voxel,
                lumen_radius_microns=lumen_radius_microns,
                lumen_radius_pixels_axes=lumen_radius_pixels_axes,
                size_of_image=energy.shape,
                params=params,
            )

        # 6. Finalize & Build Result
        final_data = finalize_edges_matlab_style(
            chosen,
            lumen_radius_microns=lumen_radius_microns,
            microns_per_voxel=microns_per_voxel,
            size_of_image=energy.shape,
        )

        atomic_joblib_dump(final_data, chosen_manifest_path)
        stage_controller.update(units_total=3, units_completed=3, substage="choose_edges")

        return EdgeSet.create(
            traces=final_data["traces"],
            connections=final_data["connections"],
            energies=final_data["energies"],
            **{k: v for k, v in final_data.items() if k not in ("traces", "connections", "energies")}
        )
