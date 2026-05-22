"""
Main pipeline orchestration for SLAVV.
Coordinates the energy, tracing, and network construction steps.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, cast

from slavv_python import utils
from slavv_python.engine.executor import StageExecutor
from slavv_python.engine.context import RunContext
from slavv_python.engine.state.models import ProgressEvent
from slavv_python.engine.state.tracker import PREPROCESS_STAGE
from slavv_python.processing.stages import edges as edge_ops
from slavv_python.processing.stages import energy
from slavv_python.processing.stages import network as network_ops
from slavv_python.processing.stages import vertices as vertex_ops
from slavv_python.workflows import (
    emit_progress,
    finalize_pipeline_results,
    prepare_pipeline_run,
    preprocess_image,
    validate_stage_control,
)

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
        """Complete SLAVV processing pipeline."""
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
        executor = StageExecutor(run_context, progress_callback, results)

        image = preprocess_image(image, parameters, run_context)
        emit_progress(progress_callback, 0.2, PREPROCESS_STAGE)

        # 1. Energy
        executor.execute(
            "energy", "energy_data", 0.4,
            compute_fn=lambda c: energy.calculate_energy_field_resumable(
                image, parameters, c, utils.get_chunking_lattice
            ),
            fallback_fn=lambda: self.compute_energy(image, parameters),
            force_rerun=force_rerun["energy"],
            log_label="Energy Field"
        )
        if stop_after == "energy":
            return self._finalize_run(run_context, results, stop_after)

        # 2. Vertices
        executor.execute(
            "vertices", "vertices", 0.6,
            compute_fn=lambda c: vertex_ops.extract_vertices_resumable(
                results["energy_data"], parameters, c
            ),
            fallback_fn=lambda: self.extract_vertices(results["energy_data"], parameters),
            force_rerun=force_rerun["vertices"],
        )
        if stop_after == "vertices":
            return self._finalize_run(run_context, results, stop_after)

        # 3. Edges
        from slavv_python.processing.stages.edges.manager import EdgeManager
        executor.execute(
            "edges", "edges", 0.8,
            compute_fn=lambda c: EdgeManager.run_resumable(
                results["energy_data"], results["vertices"], parameters, c
            ),
            fallback_fn=lambda: self.extract_edges(results["energy_data"], results["vertices"], parameters),
            force_rerun=force_rerun["edges"],
        )
        if stop_after == "edges":
            return self._finalize_run(run_context, results, stop_after)

        # 4. Network
        executor.execute(
            "network", "network", 1.0,
            compute_fn=lambda c: network_ops.construct_network_resumable(
                results["edges"], results["vertices"], parameters, c
            ),
            fallback_fn=lambda: self.build_network(results["edges"], results["vertices"], parameters),
            force_rerun=force_rerun["network"],
        )

        logger.info("SLAVV processing pipeline completed")
        return self._finalize_run(run_context, results, stop_after)

    def _finalize_run(self, run_context, results, stop_after):
        """Finalize state and return normalized results."""
        if run_context is not None:
            run_context.finalize_run(stop_after=stop_after)
        return cast("dict[str, Any]", finalize_pipeline_results(results))

    def compute_energy(self, image: np.ndarray, params: dict[str, Any]) -> dict[str, Any]:
        """Calculate the multi-scale energy field."""
        from slavv_python import utils as utils_module

        return energy.compute_energy(image, params, utils_module.get_chunking_lattice)

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
        return cast("dict[str, Any]", network_ops.build_network(edges, vertices, params))
