"""Consolidated edge extraction manager."""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING, Any, cast

from slavv_python.pipeline.edges.audit import (
    _build_edge_candidate_audit,
    _normalize_candidate_origin_counts,
)
from slavv_python.pipeline.edges.candidate_manifest import (
    CandidateManifest,
    candidate_as_payload,
)
from slavv_python.pipeline.edges.discovery import (
    _use_watershed_discovery,
    frontier_origin_counts,
    frontier_origin_counts_from_diagnostics,
    prepare_edge_discovery_context,
    select_edge_discovery,
)
from slavv_python.pipeline.edges.execution import EdgeExecutionArtifacts
from slavv_python.pipeline.edges.payloads import _empty_edges_result
from slavv_python.pipeline.edges.selection_workflow import select_and_finalize_edge_set
from slavv_python.schema.results import EdgeSet, EnergyResult, VertexSet

if TYPE_CHECKING:
    from slavv_python.engine.state import StageController

logger = logging.getLogger(__name__)


def _timing_writer_authorized(stage_controller: StageController | None) -> bool:
    """Return whether the current process owns the parity writer lease.

    A ``StageController`` permits normal resumable artifact persistence, but it
    is not itself proof that this process is the authorized parity writer.  The
    split timing artifact is certification metadata, so require the hardened
    run-local lease and an exact PID match.
    """
    if stage_controller is None:
        return False
    from slavv_python.analytics.parity.runs.writer_lease import load_writer_lease

    lease = load_writer_lease(stage_controller.run_context.run_root)
    if not lease or lease.get("status") != "running":
        return False
    try:
        return int(lease["pid"]) == os.getpid()
    except (KeyError, TypeError, ValueError):
        return False


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

        discovery = select_edge_discovery(energy_data, params)
        context = prepare_edge_discovery_context(
            energy_data,
            vertices,
            params,
            stage_controller=cast("StageController", _NullStageController()),
            heartbeat=heartbeat,
        )
        return discovery.discover(context)

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
        artifacts = EdgeExecutionArtifacts(
            handle.artifact_path,
            authorized=stage_controller is not None,
            timing_authorized=_timing_writer_authorized(stage_controller),
        )

        vertex_positions = vertices.positions
        if len(vertex_positions) == 0:
            return EdgeSet.from_dict(_empty_edges_result(vertex_positions))

        logger.info("Creating vertex center lookup image...")
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

        from slavv_python.analytics.parity.utils import now_iso
        from slavv_python.analytics.performance.edge_timing import EdgeTimingRecord

        execution_started_at = now_iso()
        t_initialization_start = time.perf_counter()
        context = prepare_edge_discovery_context(
            energy_data,
            vertices,
            params,
            stage_controller=cast("StageController", handle),
            heartbeat=heartbeat,
        )
        initialization_elapsed = time.perf_counter() - t_initialization_start
        logger.info("Vertex center lookup image created")
        t_discovery_start = time.perf_counter()
        manifest = discovery.discover(context)
        t_discovery_end = time.perf_counter()
        discovery_elapsed = t_discovery_end - t_discovery_start
        logger.info("Edge discovery completed in %.2f seconds", discovery_elapsed)
        discovery_completed_at = now_iso()
        if resumable:
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
            artifacts.write_candidates(manifest, candidate_audit)

            # Persist the expensive Discovery boundary before Selection starts.
            # If Selection is interrupted, the completed Discovery measurement
            # remains available and the Candidate Set remains resumable.
            artifacts.write_timing(
                EdgeTimingRecord(
                    initialization_seconds=initialization_elapsed,
                    discovery_seconds=discovery_elapsed,
                    selection_seconds=None,
                    candidate_count=len(manifest.connections),
                    edge_count=None,
                    exact_route=use_watershed,
                    writer_authorized=artifacts.timing_authorized,
                    started_at=execution_started_at,
                    discovery_completed_at=discovery_completed_at,
                    completed_at=None,
                ).to_payload()
            )

            handle.update(
                units_total=3,
                units_completed=0,
                substage="persist_candidates",
                detail="Writing edge candidate artifacts",
                resumed=False,
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
        t_selection_start = time.perf_counter()
        edge_set = select_and_finalize_edge_set(
            manifest,
            energy_data,
            vertices,
            params,
            apply_bridge_vertices=use_watershed,
        )
        t_selection_end = time.perf_counter()
        selection_elapsed = t_selection_end - t_selection_start
        logger.info("Edge selection and finalization completed in %.2f seconds", selection_elapsed)

        chosen_dict = edge_set.to_dict()

        if resumable:
            timing_payload = EdgeTimingRecord(
                initialization_seconds=initialization_elapsed,
                discovery_seconds=discovery_elapsed,
                selection_seconds=selection_elapsed,
                candidate_count=len(manifest.connections),
                edge_count=len(chosen_dict.get("traces", [])),
                exact_route=use_watershed,
                writer_authorized=artifacts.timing_authorized,
                started_at=execution_started_at,
                discovery_completed_at=discovery_completed_at,
                completed_at=now_iso(),
            ).to_payload()
            artifacts.write_timing(timing_payload)
            candidate_lifecycle = None
            from slavv_python.pipeline.edges.frontier_events import (
                _build_frontier_candidate_lifecycle,
            )

            if use_watershed and manifest.frontier_lifecycle_events:
                candidate_lifecycle = _build_frontier_candidate_lifecycle(
                    candidate_as_payload(manifest),
                    chosen_dict.get("chosen_candidate_indices"),
                )
            artifacts.write_final(chosen_dict, candidate_lifecycle=candidate_lifecycle)
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
