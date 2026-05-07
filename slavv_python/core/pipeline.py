"""
Main pipeline orchestration for SLAVV.
Coordinates the energy, tracing, and network construction steps.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, cast

from .. import utils
from ..runtime import ProgressEvent, RunContext
from ..runtime.run_state import PREPROCESS_STAGE
from ..workflows import (
    build_standard_pipeline_steps,
    emit_progress,
    finalize_pipeline_results,
    prepare_pipeline_run,
    preprocess_image,
    resolve_edges_stage,
    resolve_energy_stage,
    resolve_network_stage,
    resolve_vertices_stage,
    run_pipeline_stage_sequence,
    validate_stage_control,
)
from . import edges as edge_ops
from . import energy
from . import network as network_ops
from . import vertices as vertex_ops

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


class SlavvPipeline:
    """Main class for SLAVV vectorization processing."""

    def __init__(self):
        self.energy_data = None
        self.vertices = None
        self.edges = None
        self.network = None

    def run(
        self,
        image: np.ndarray,
        parameters: dict[str, Any],
        progress_callback: Callable[[float, str], None] | None = None,
        event_callback: Callable[[ProgressEvent], None] | None = None,
        run_dir: str | None = None,
        stop_after: str | None = None,
        force_rerun_from: str | None = None,
    ) -> dict[str, Any]:
        """Complete SLAVV processing pipeline.

        Args:
            image: 3D input image array (y, x, z)
            parameters: Dictionary of processing parameters
            progress_callback: Optional callable receiving ``(fraction, stage)``
                updates as the pipeline advances from 0.0 to 1.0.
            event_callback: Optional structured callback receiving ``ProgressEvent``
                snapshots whenever run state changes.
            run_dir: Optional structured run directory. When provided, staged
                metadata and artifacts are persisted under this root.
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
        early_result = run_pipeline_stage_sequence(
            results,
            steps=build_standard_pipeline_steps(
                resolve_energy=lambda: self._resolve_energy_stage(
                    image,
                    parameters,
                    run_context,
                    force_rerun["energy"],
                ),
                resolve_vertices=lambda: self._resolve_vertices_stage(
                    cast("dict[str, Any]", results["energy_data"]),
                    parameters,
                    run_context,
                    force_rerun["vertices"],
                ),
                resolve_edges=lambda: self._resolve_edges_stage(
                    cast("dict[str, Any]", results["energy_data"]),
                    cast("dict[str, Any]", results["vertices"]),
                    parameters,
                    run_context,
                    force_rerun["edges"],
                ),
                resolve_network=lambda: self._resolve_network_stage(
                    cast("dict[str, Any]", results["edges"]),
                    cast("dict[str, Any]", results["vertices"]),
                    parameters,
                    run_context,
                    force_rerun["network"],
                ),
            ),
            progress_callback=progress_callback,
            stop_after=stop_after,
            run_context=run_context,
        )
        if early_result:
            return cast("dict[str, Any]", early_result)

        logger.info("SLAVV processing pipeline completed")
        if run_context is not None:
            run_context.finalize_run(stop_after=stop_after)
        return cast("dict[str, Any]", finalize_pipeline_results(results))

    def _resolve_energy_stage(
        self,
        image: np.ndarray,
        parameters: dict[str, Any],
        run_context: RunContext | None,
        force_rerun: bool,
    ) -> dict[str, Any]:
        return cast(
            "dict[str, Any]",
            resolve_energy_stage(
                run_context=run_context,
                force_rerun=force_rerun,
                fallback_fn=lambda: self.compute_energy(image, parameters),
                resumable_fn=lambda controller: energy.calculate_energy_field_resumable(
                    image,
                    parameters,
                    controller,
                    utils.get_chunking_lattice,
                ),
                logger=logger,
            ),
        )

    def _resolve_vertices_stage(
        self,
        energy_data: dict[str, Any],
        parameters: dict[str, Any],
        run_context: RunContext | None,
        force_rerun: bool,
    ) -> dict[str, Any]:
        return cast(
            "dict[str, Any]",
            resolve_vertices_stage(
                run_context=run_context,
                force_rerun=force_rerun,
                fallback_fn=lambda: self.extract_vertices(energy_data, parameters),
                resumable_fn=lambda controller: vertex_ops.extract_vertices_resumable(
                    energy_data,
                    parameters,
                    controller,
                ),
                logger=logger,
            ),
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
        return cast(
            "dict[str, Any]",
            resolve_edges_stage(
                run_context=run_context,
                force_rerun=force_rerun,
                edge_method=edge_method,
                tracing_fallback_fn=lambda: self.extract_edges(energy_data, vertices, parameters),
                watershed_fallback_fn=lambda: self.extract_edges_watershed(
                    energy_data,
                    vertices,
                    parameters,
                ),
                tracing_resumable_fn=lambda controller: edge_ops.extract_edges_resumable(
                    energy_data,
                    vertices,
                    parameters,
                    controller,
                ),
                watershed_resumable_fn=lambda controller: (
                    edge_ops.extract_edges_watershed_resumable(
                        energy_data,
                        vertices,
                        parameters,
                        controller,
                    )
                ),
                logger=logger,
            ),
        )

    def _resolve_network_stage(
        self,
        edges: dict[str, Any],
        vertices: dict[str, Any],
        parameters: dict[str, Any],
        run_context: RunContext | None,
        force_rerun: bool,
    ) -> dict[str, Any]:
        return cast(
            "dict[str, Any]",
            resolve_network_stage(
                run_context=run_context,
                force_rerun=force_rerun,
                fallback_fn=lambda: self.build_network(edges, vertices, parameters),
                resumable_fn=lambda controller: network_ops.construct_network_resumable(
                    edges,
                    vertices,
                    parameters,
                    controller,
                ),
                logger=logger,
            ),
        )

    def compute_energy(self, image: np.ndarray, params: dict[str, Any]) -> dict[str, Any]:
        """Calculate the multi-scale energy field."""
        from slavv_python import utils as utils_module

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

    def build_network(
        self, edges: dict[str, Any], vertices: dict[str, Any], params: dict[str, Any]
    ) -> dict[str, Any]:
        """Construct the final network from traced edges and vertices."""
        return cast("dict[str, Any]", network_ops.construct_network(edges, vertices, params))


class SLAVVProcessor(SlavvPipeline):
    """Legacy compatibility class for SlavvPipeline."""

    def calculate_energy_field(
        self,
        image: np.ndarray,
        params: dict[str, Any],
        get_chunking_lattice_func: Callable | None = None,
    ) -> dict[str, Any]:
        """Legacy compatibility method for compute_energy."""
        return energy.calculate_energy_field(image, params, get_chunking_lattice_func)

    def construct_network(
        self,
        edges: dict[str, Any],
        vertices: dict[str, Any],
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Legacy compatibility method for build_network."""
        return self.build_network(edges, vertices, params)

    def process_image(
        self,
        image: np.ndarray,
        params: dict[str, Any],
        progress_callback: Callable[[float, str], None] | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Legacy compatibility method for run."""
        return self.run(image, params, progress_callback=progress_callback, **kwargs)
