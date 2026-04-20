"""
Main pipeline orchestration for SLAVV.
Coordinates the energy, tracing, and graph construction steps.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, cast

from .. import utils
from ..runtime import ProgressEvent, RunContext
from ..runtime.run_state import PREPROCESS_STAGE
from ..workflows import (
    PipelineStageStep,
    emit_progress,
    finalize_pipeline_results,
    prepare_pipeline_run,
    preprocess_image,
    resolve_stage_with_checkpoint,
    run_pipeline_stage_sequence,
    validate_stage_control,
)
from . import edges as edge_ops
from . import energy, graph
from . import vertices as vertex_ops

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


class SLAVVProcessor:
    """Main class for SLAVV vectorization processing"""

    def __init__(self):
        self.energy_data = None
        self.vertices = None
        self.edges = None
        self.network = None

    def process_image(
        self,
        image: np.ndarray,
        parameters: dict[str, Any],
        progress_callback: Callable[[float, str], None] | None = None,
        event_callback: Callable[[ProgressEvent], None] | None = None,
        run_dir: str | None = None,
        checkpoint_dir: str | None = None,
        stop_after: str | None = None,
        force_rerun_from: str | None = None,
    ) -> dict[str, Any]:
        """Complete SLAVV processing pipeline.

        MATLAB Equivalent: `vectorize_V200.m` (with resume capability)

        Args:
            image: 3D input image array (y, x, z)
            parameters: Dictionary of processing parameters
            progress_callback: Optional callable receiving ``(fraction, stage)``
                updates as the pipeline advances from 0.0 to 1.0.
            event_callback: Optional structured callback receiving ``ProgressEvent``
                snapshots whenever run state changes.
            run_dir: Optional structured run directory. When provided, staged
                metadata and artifacts are persisted under this root.
            checkpoint_dir: Optional directory path. If provided, intermediate steps
                (Energy, Vertices, Edges, Network) will be saved/loaded from this directory.
                Enables resuming crashed runs or inspecting intermediate results.
            stop_after: Optional string ('energy', 'vertices', 'edges', 'network').
                The pipeline will return early after completing the specified stage.
            force_rerun_from: Optional string ('energy', 'vertices', 'edges', 'network').
                If provided, checkpoints for this stage and all subsequent stages will be
                ignored, forcing recalculation.
        Returns:
            Dictionary containing all processing results
        """
        if image.ndim != 3 or 0 in image.shape:
            raise ValueError("Input image must be a non-empty 3D array")
        validate_stage_control(stop_after, "stop_after")
        validate_stage_control(force_rerun_from, "force_rerun_from")

        logger.info("Starting SLAVV processing pipeline")

        emit_progress(progress_callback, 0.0, "start")

        prepared_run = prepare_pipeline_run(
            image,
            parameters,
            run_dir=run_dir,
            checkpoint_dir=checkpoint_dir,
            stop_after=stop_after,
            force_rerun_from=force_rerun_from,
            event_callback=event_callback,
            run_context_factory=RunContext,
        )
        parameters = prepared_run.parameters
        run_context = prepared_run.run_context
        force_rerun = prepared_run.force_rerun

        results = {"parameters": parameters}
        image = preprocess_image(image, parameters, run_context)
        emit_progress(progress_callback, 0.2, PREPROCESS_STAGE)
        if early_result := run_pipeline_stage_sequence(
            results,
            steps=[
                PipelineStageStep(
                    result_key="energy_data",
                    stage_name="energy",
                    progress_fraction=0.4,
                    resolve_fn=lambda: self._resolve_energy_stage(
                        image,
                        parameters,
                        run_context,
                        force_rerun["energy"],
                    ),
                ),
                PipelineStageStep(
                    result_key="vertices",
                    stage_name="vertices",
                    progress_fraction=0.6,
                    resolve_fn=lambda: self._resolve_vertices_stage(
                        cast("dict[str, Any]", results["energy_data"]),
                        parameters,
                        run_context,
                        force_rerun["vertices"],
                    ),
                ),
                PipelineStageStep(
                    result_key="edges",
                    stage_name="edges",
                    progress_fraction=0.8,
                    resolve_fn=lambda: self._resolve_edges_stage(
                        cast("dict[str, Any]", results["energy_data"]),
                        cast("dict[str, Any]", results["vertices"]),
                        parameters,
                        run_context,
                        force_rerun["edges"],
                    ),
                ),
                PipelineStageStep(
                    result_key="network",
                    stage_name="network",
                    progress_fraction=1.0,
                    resolve_fn=lambda: self._resolve_network_stage(
                        cast("dict[str, Any]", results["edges"]),
                        cast("dict[str, Any]", results["vertices"]),
                        parameters,
                        run_context,
                        force_rerun["network"],
                    ),
                ),
            ],
            progress_callback=progress_callback,
            stop_after=stop_after,
            run_context=run_context,
        ):
            return early_result

        logger.info("SLAVV processing pipeline completed")
        if run_context is not None:
            run_context.finalize_run(stop_after=stop_after)
        return finalize_pipeline_results(results)

    def _resolve_energy_stage(
        self,
        image: np.ndarray,
        parameters: dict[str, Any],
        run_context: RunContext | None,
        force_rerun: bool,
    ) -> dict[str, Any]:
        return resolve_stage_with_checkpoint(
            run_context=run_context,
            force_rerun=force_rerun,
            stage_name="energy",
            cached_log_label="Energy Field",
            cached_detail="Loaded energy checkpoint",
            success_detail="Energy field ready",
            fallback_fn=lambda: self.calculate_energy_field(image, parameters),
            compute_fn=lambda controller: energy.calculate_energy_field_resumable(
                image,
                parameters,
                controller,
                utils.get_chunking_lattice,
            ),
            logger=logger,
        )

    def _resolve_vertices_stage(
        self,
        energy_data: dict[str, Any],
        parameters: dict[str, Any],
        run_context: RunContext | None,
        force_rerun: bool,
    ) -> dict[str, Any]:
        return resolve_stage_with_checkpoint(
            run_context=run_context,
            force_rerun=force_rerun,
            stage_name="vertices",
            cached_log_label="Vertices",
            cached_detail="Loaded vertex checkpoint",
            success_detail="Vertices extracted",
            fallback_fn=lambda: self.extract_vertices(energy_data, parameters),
            compute_fn=lambda controller: vertex_ops.extract_vertices_resumable(
                energy_data,
                parameters,
                controller,
            ),
            logger=logger,
        )

    def _resolve_edges_stage(
        self,
        energy_data: dict[str, Any],
        vertices: dict[str, Any],
        parameters: dict[str, Any],
        run_context: RunContext | None,
        force_rerun: bool,
    ) -> dict[str, Any]:
        edge_method = parameters.get("edge_method", "tracing")
        return resolve_stage_with_checkpoint(
            run_context=run_context,
            force_rerun=force_rerun,
            stage_name="edges",
            cached_log_label="Edges",
            cached_detail="Loaded edge checkpoint",
            success_detail="Edges extracted",
            fallback_fn=lambda: (
                self.extract_edges_watershed(energy_data, vertices, parameters)
                if edge_method == "watershed"
                else self.extract_edges(energy_data, vertices, parameters)
            ),
            compute_fn=lambda controller: (
                edge_ops.extract_edges_watershed_resumable(
                    energy_data,
                    vertices,
                    parameters,
                    controller,
                )
                if edge_method == "watershed"
                else edge_ops.extract_edges_resumable(
                    energy_data,
                    vertices,
                    parameters,
                    controller,
                )
            ),
            logger=logger,
        )

    def _resolve_network_stage(
        self,
        edges: dict[str, Any],
        vertices: dict[str, Any],
        parameters: dict[str, Any],
        run_context: RunContext | None,
        force_rerun: bool,
    ) -> dict[str, Any]:
        return resolve_stage_with_checkpoint(
            run_context=run_context,
            force_rerun=force_rerun,
            stage_name="network",
            cached_log_label="Network",
            cached_detail="Loaded network checkpoint",
            success_detail="Network constructed",
            fallback_fn=lambda: self.construct_network(edges, vertices, parameters),
            compute_fn=lambda controller: graph.construct_network_resumable(
                edges,
                vertices,
                parameters,
                controller,
            ),
            logger=logger,
        )

    def calculate_energy_field(self, image: np.ndarray, params: dict[str, Any]) -> dict[str, Any]:
        """Calculate multi-scale energy field using Hessian. Delegates to ``energy`` module."""
        from .. import utils as utils_module

        return energy.calculate_energy_field(image, params, utils_module.get_chunking_lattice)

    def extract_vertices(
        self, energy_data: dict[str, Any], params: dict[str, Any]
    ) -> dict[str, Any]:
        """Extract vertices as local extrema. Delegates to ``vertices`` module."""
        result = vertex_ops.extract_vertices(energy_data, params)
        return cast("dict[str, Any]", result)

    def extract_edges(
        self, energy_data: dict[str, Any], vertices: dict[str, Any], params: dict[str, Any]
    ) -> dict[str, Any]:
        """Extract edges by tracing. Delegates to ``edges`` module."""
        return edge_ops.extract_edges(energy_data, vertices, params)

    def extract_edges_watershed(
        self, energy_data: dict[str, Any], vertices: dict[str, Any], params: dict[str, Any]
    ) -> dict[str, Any]:
        """Extract edges by watershed. Delegates to ``edges`` module."""
        return edge_ops.extract_edges_watershed(energy_data, vertices, params)

    def construct_network(
        self, edges: dict[str, Any], vertices: dict[str, Any], params: dict[str, Any]
    ) -> dict[str, Any]:
        """Construct network from traces. Delegates to ``graph`` module."""
        return graph.construct_network(edges, vertices, params)
