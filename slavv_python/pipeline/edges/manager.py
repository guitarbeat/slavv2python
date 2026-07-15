"""Consolidated edge extraction manager."""

from __future__ import annotations

import logging
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from slavv_python.pipeline.edges.artifacts import resolve_edge_candidate_persistence
from slavv_python.pipeline.edges.audit import (
    _build_edge_candidate_audit,
    _normalize_candidate_origin_counts,
)
from slavv_python.pipeline.edges.candidate_manifest import (
    CandidateManifest,
    candidate_as_payload,
)
from slavv_python.pipeline.edges.discovery import (
    EdgeDiscoveryContext,
    _use_watershed_discovery,
    frontier_origin_counts,
    frontier_origin_counts_from_diagnostics,
    resolve_lumen_radius_pixels_axes,
    select_edge_discovery,
)
from slavv_python.pipeline.edges.payloads import _empty_edges_result
from slavv_python.pipeline.edges.selection_workflow import select_and_finalize_edge_set
from slavv_python.pipeline.policy import PipelinePolicy
from slavv_python.pipeline.vertices.painting import paint_vertex_center_image
from slavv_python.schema.results import EdgeSet, EnergyResult, VertexSet

if TYPE_CHECKING:
    from slavv_python.engine.state import StageController

logger = logging.getLogger(__name__)


class _NullStageController:
    """No-op stage handle for ephemeral edge extraction."""

    run_context = None

    def begin(self, **_kwargs: Any) -> None:
        return None

    def update(self, **_kwargs: Any) -> None:
        return None

    def complete(self, **_kwargs: Any) -> None:
        return None

    def artifact_path(self, file_name: str) -> Path:
        raise RuntimeError(f"artifact_path({file_name!r}) called during ephemeral edge extraction")


class EdgeManager:
    """Consolidated manager for edge discovery, selection, and resumable persistence."""

    @classmethod
    def run(
        cls,
        energy_data: EnergyResult,
        vertices: VertexSet,
        params: dict[str, Any],
    ) -> EdgeSet:
        """Extract edges without run-directory checkpointing or parity audit artifacts.

        Args:
            energy_data: Result from the energy stage.
            vertices: Result from the vertices stage.
            params: Pipeline parameters.

        Returns:
            EdgeSet: The extracted and filtered edges.
        """
        return cls._run_tracing(energy_data, vertices, params, stage_controller=None)

    @classmethod
    def run_resumable(
        cls,
        energy_data: EnergyResult,
        vertices: VertexSet,
        params: dict[str, Any],
        stage_controller: StageController,
    ) -> EdgeSet:
        """Execute the full edge extraction lifecycle with resumability and audit artifacts.

        Args:
            energy_data: Result from the energy stage.
            vertices: Result from the vertices stage.
            params: Pipeline parameters.
            stage_controller: Controller for managing stage state and artifacts.

        Returns:
            EdgeSet: The extracted and filtered edges.
        """
        return cls._run_tracing(energy_data, vertices, params, stage_controller=stage_controller)

    @classmethod
    def discover_candidates(
        cls,
        energy_data: EnergyResult,
        vertices: VertexSet,
        params: dict[str, Any],
        *,
        heartbeat: Any | None = None,
    ) -> CandidateManifest:
        """Run edge discovery only (no selection/finalize) through the discovery strategy seam."""
        if len(vertices.positions) == 0:
            return CandidateManifest.empty()

        policy = PipelinePolicy.from_params(params)
        microns_per_voxel = np.array(
            params.get("microns_per_voxel", [1.0, 1.0, 1.0]),
            dtype=policy.precision,
        )
        lumen_radius_pixels_axes = resolve_lumen_radius_pixels_axes(
            energy_data,
            microns_per_voxel,
            policy=policy,
        )
        vertex_center_image = paint_vertex_center_image(
            vertices.positions, energy_data.energy.shape
        )
        discovery = select_edge_discovery(energy_data, params)
        return discovery.discover(
            EdgeDiscoveryContext(
                energy_data=energy_data,
                vertices=vertices,
                params=params,
                stage_controller=cast("StageController", _NullStageController()),
                vertex_center_image=vertex_center_image,
                lumen_radius_pixels_axes=lumen_radius_pixels_axes,
                microns_per_voxel=microns_per_voxel,
                heartbeat=heartbeat,
            )
        )

    @classmethod
    def _run_tracing(
        cls,
        energy_data: EnergyResult,
        vertices: VertexSet,
        params: dict[str, Any],
        *,
        stage_controller: StageController | None,
    ) -> EdgeSet:
        resumable = stage_controller is not None
        handle: StageController | _NullStageController = (
            stage_controller if stage_controller is not None else _NullStageController()
        )

        policy = PipelinePolicy.from_params(params)
        energy = energy_data.energy
        vertex_positions = vertices.positions
        microns_per_voxel = np.array(
            params.get("microns_per_voxel", [1.0, 1.0, 1.0]),
            dtype=policy.precision,
        )

        if len(vertex_positions) == 0:
            return EdgeSet.from_dict(_empty_edges_result(vertex_positions))

        lumen_radius_pixels_axes = resolve_lumen_radius_pixels_axes(
            energy_data,
            microns_per_voxel,
            policy=policy,
        )

        logger.info("Creating vertex center lookup image...")
        vertex_center_image = paint_vertex_center_image(vertex_positions, energy.shape)
        logger.info("Vertex center lookup image created")

        use_watershed = _use_watershed_discovery(energy_data.to_dict(), params)
        discovery = select_edge_discovery(energy_data, params)

        if resumable:
            handle.begin(
                detail=(
                    "Generating edge candidates through Watershed Discovery (Exact Route)"
                    if use_watershed
                    else "Generating edge candidates through Tracing Discovery (Paper Path)"
                ),
                units_total=3,
                units_completed=0,
                substage="generate_candidates",
                resumed=False,
            )

        heartbeat = None
        if use_watershed and resumable:

            def heartbeat(iteration_count: int, candidate_count: int) -> None:
                handle.update(
                    units_total=3,
                    units_completed=0,
                    substage="generate_candidates",
                    detail=(
                        "Generating edge candidates through Watershed Discovery (Exact Route) "
                        f"(iterations={iteration_count}, candidates={candidate_count})"
                    ),
                    resumed=False,
                )

        manifest = discovery.discover(
            EdgeDiscoveryContext(
                energy_data=energy_data,
                vertices=vertices,
                params=params,
                stage_controller=cast("StageController", handle),
                vertex_center_image=vertex_center_image,
                lumen_radius_pixels_axes=lumen_radius_pixels_axes,
                microns_per_voxel=microns_per_voxel,
                heartbeat=heartbeat,
            )
        )
        if resumable:
            from slavv_python.engine.state.io import atomic_joblib_dump, atomic_write_json

            if use_watershed:
                frontier_counts = frontier_origin_counts_from_diagnostics(manifest)
            else:
                frontier_counts = frontier_origin_counts(manifest)

            supplement_origin_counts = _normalize_candidate_origin_counts(
                manifest.diagnostics.get("watershed_per_origin_candidate_counts")
            )
            candidate_audit = _build_edge_candidate_audit(
                manifest,
                len(vertex_positions),
                use_frontier_tracer=use_watershed,
                frontier_origin_counts=frontier_counts,
                supplement_origin_counts={
                    int(origin_index): int(count)
                    for origin_index, count in (supplement_origin_counts or {}).items()
                },
            )
            atomic_write_json(handle.artifact_path("candidate_audit.json"), candidate_audit)

            handle.update(
                units_total=3,
                units_completed=0,
                substage="persist_candidates",
                detail="Writing edge candidate artifacts",
                resumed=False,
            )
            candidates_payload = candidate_as_payload(manifest)
            atomic_joblib_dump(candidates_payload, handle.artifact_path("candidates.pkl"))
            if use_watershed and stage_controller is not None:
                run_context = stage_controller.run_context
                if run_context is not None:
                    resolve_edge_candidate_persistence(
                        params,
                        use_frontier=True,
                    ).write_candidate_checkpoint(
                        run_context.checkpoints_dir,
                        candidates_payload,
                        include_debug_maps=bool(params.get("parity_include_debug_maps", False)),
                    )
            handle.update(
                units_total=3,
                units_completed=1,
                substage="persist_candidates",
                detail="Wrote edge candidate artifacts",
                resumed=False,
            )

        if resumable:
            handle.update(
                units_total=3,
                units_completed=1,
                substage="choose_edges",
                detail="Choosing, bridging, and finalizing edges",
                resumed=False,
            )

        # Post-Edge Discovery: single deep module shared with residual scripts
        edge_set = select_and_finalize_edge_set(
            manifest,
            energy_data,
            vertices,
            params,
            apply_bridge_vertices=use_watershed,
        )
        chosen_dict = edge_set.to_dict()

        if resumable:
            from slavv_python.engine.state.io import atomic_joblib_dump, atomic_write_json
            from slavv_python.pipeline.edges.frontier_events import (
                _build_frontier_candidate_lifecycle,
            )

            if use_watershed and manifest.frontier_lifecycle_events:
                candidate_lifecycle = _build_frontier_candidate_lifecycle(
                    candidate_as_payload(manifest),
                    chosen_dict.get("chosen_candidate_indices"),
                )
                atomic_write_json(
                    handle.artifact_path("candidate_lifecycle.json"),
                    candidate_lifecycle,
                )

            atomic_joblib_dump(chosen_dict, handle.artifact_path("chosen_edges.pkl"))
            handle.update(
                units_total=3,
                units_completed=3,
                substage="finalize_edges",
                detail="Finalized edges",
                resumed=False,
            )
        else:
            logger.info(
                "Extracted %d chosen edges from %d traced candidates",
                len(chosen_dict.get("traces", [])),
                chosen_dict.get("diagnostics", {}).get("candidate_traced_edge_count", 0),
            )

        return EdgeSet.from_dict(chosen_dict)

    @classmethod
    def run_watershed_resumable(
        cls,
        energy_data: EnergyResult,
        vertices: VertexSet,
        params: dict[str, Any],
        stage_controller: StageController,
    ) -> EdgeSet:
        """Delegate watershed resumable extraction (per-label units).

        Args:
            energy_data: Result from the energy stage.
            vertices: Result from the vertices stage.
            params: Pipeline parameters.
            stage_controller: Controller for managing stage state and artifacts.

        Returns:
            EdgeSet: The extracted edges using watershed logic.
        """
        from slavv_python.pipeline.edges import resumable as watershed_resumable

        return cast(
            "EdgeSet",
            watershed_resumable.extract_edges_watershed_resumable(
                energy_data,
                vertices,
                params,
                stage_controller,
            ),
        )


__all__ = ["CandidateManifest", "EdgeManager"]
