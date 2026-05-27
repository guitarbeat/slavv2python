"""
Main pipeline orchestration for SLAVV.
Coordinates the energy, tracing, and network construction steps.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, cast

from slavv_python import utils
from slavv_python.engine.context import RunContext
from slavv_python.engine.executor import StageExecutor
from slavv_python.engine.state.run_state import RunState
from slavv_python.engine.state.tracker import PREPROCESS_STAGE

if TYPE_CHECKING:
    from .state.models import ProgressEvent
from slavv_python.processing.stages import edges as edge_ops
from slavv_python.processing.stages import energy
from slavv_python.processing.stages import vertices as vertex_ops
from slavv_python.schema.results import (
    EdgeSet,
    EnergyResult,
    NetworkResult,
    PipelineResult,
    VertexSet,
)
from slavv_python.workflows import (
    emit_progress,
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
    ) -> PipelineResult:
        """Complete SLAVV processing pipeline."""
        if image.ndim != 3 or 0 in image.shape:
            raise ValueError("Input image must be a non-empty 3D array")
        validate_stage_control(stop_after, "stop_after")
        validate_stage_control(force_rerun_from, "force_rerun_from")

        emit_progress(progress_callback, 0.0, "start")

        parameters, run_context, force_rerun = RunContext.prepare(
            image,
            parameters,
            run_dir=run_dir,
            stop_after=stop_after,
            force_rerun_from=force_rerun_from,
            event_callback=event_callback,
        )

        run_state = RunState(parameters=parameters)
        executor = StageExecutor(run_context, progress_callback, run_state)

        image = preprocess_image(image, parameters, run_context)
        emit_progress(progress_callback, 0.2, PREPROCESS_STAGE)

        # 1. Energy
        executor.execute(
            "energy",
            "energy_data",
            0.4,
            compute_fn=lambda c: energy.calculate_energy_field_resumable(
                image, parameters, c, utils.get_chunking_lattice
            ),
            fallback_fn=lambda: self.compute_energy(image, parameters),
            force_rerun=force_rerun["energy"],
            log_label="Energy Field",
            schema_class=EnergyResult,
        )
        if stop_after == "energy":
            return self._finalize_run(run_context, run_state, stop_after)

        # 2. Vertices
        executor.execute(
            "vertices",
            "vertices",
            0.6,
            compute_fn=lambda c: vertex_ops.extract_vertices_resumable(
                run_state.energy_data, parameters, c
            ),
            fallback_fn=lambda: self.extract_vertices(run_state.energy_data, parameters),
            force_rerun=force_rerun["vertices"],
            schema_class=VertexSet,
        )
        if stop_after == "vertices":
            return self._finalize_run(run_context, run_state, stop_after)

        # 3. Edges
        from slavv_python.processing.stages.edges.manager import EdgeManager

        executor.execute(
            "edges",
            "edges",
            0.8,
            compute_fn=lambda c: EdgeManager.run_resumable(
                run_state.energy_data, run_state.vertices, parameters, c
            ),
            fallback_fn=lambda: self.extract_edges(
                run_state.energy_data,
                run_state.vertices,
                parameters,
            ),
            force_rerun=force_rerun["edges"],
            schema_class=EdgeSet,
        )
        if stop_after == "edges":
            return self._finalize_run(run_context, run_state, stop_after)

        # 4. Network
        from slavv_python.processing.stages.network.manager import NetworkManager

        executor.execute(
            "network",
            "network",
            1.0,
            compute_fn=lambda c: NetworkManager.run_resumable(
                run_state.edges, run_state.vertices, parameters, c
            ),
            fallback_fn=lambda: self.build_network(
                run_state.edges if run_state.edges is not None else {},
                run_state.vertices if run_state.vertices is not None else {},
                parameters,
            ),
            force_rerun=force_rerun["network"],
            schema_class=NetworkResult,
        )

        logger.info("SLAVV processing pipeline completed")
        return self._finalize_run(run_context, run_state, stop_after)

    def _finalize_run(
        self, run_context: RunContext | None, run_state: RunState, stop_after: str | None
    ) -> PipelineResult:
        """Finalize state and return typed pipeline results."""
        if run_context is not None:
            run_context.finalize_run(stop_after=stop_after)
        return run_state.to_pipeline_result()

    def compute_energy(self, image: np.ndarray, params: dict[str, Any]) -> dict[str, Any]:
        """Calculate the multi-scale energy field."""
        from slavv_python import utils as utils_module

        return cast(
            "dict[str, Any]",
            energy.compute_energy(image, params, utils_module.get_chunking_lattice),
        )

    def extract_vertices(
        self, energy_data: EnergyResult | dict[str, Any] | None, params: dict[str, Any]
    ) -> VertexSet:
        """Extract vertices as local extrema. Delegates to ``vertices`` module."""
        if energy_data is None:
            raise ValueError("energy_data is required before vertex extraction")
        typed_energy = (
            energy_data
            if isinstance(energy_data, EnergyResult)
            else EnergyResult.from_dict(energy_data)
        )
        return vertex_ops.extract_vertices(typed_energy, params)

    def extract_edges(
        self,
        energy_data: EnergyResult | dict[str, Any] | None,
        vertices: VertexSet | dict[str, Any] | None,
        params: dict[str, Any],
    ) -> EdgeSet:
        """Extract edges by tracing. Delegates to ``edges`` module."""
        if energy_data is None or vertices is None:
            raise ValueError("energy_data and vertices are required before edge extraction")
        typed_energy = (
            energy_data
            if isinstance(energy_data, EnergyResult)
            else EnergyResult.from_dict(energy_data)
        )
        typed_vertices = (
            vertices if isinstance(vertices, VertexSet) else VertexSet.from_dict(vertices)
        )
        return edge_ops.extract_edges(typed_energy, typed_vertices, params)

    def extract_edges_watershed(
        self, energy_data: dict[str, Any], vertices: dict[str, Any], params: dict[str, Any]
    ) -> dict[str, Any]:
        """Extract edges by watershed. Delegates to ``edges`` module."""
        return cast(
            "dict[str, Any]", edge_ops.extract_edges_watershed(energy_data, vertices, params)
        )

    def build_network(
        self,
        edges: dict[str, Any] | EdgeSet,
        vertices: dict[str, Any] | VertexSet,
        params: dict[str, Any],
    ) -> NetworkResult:
        """Construct the final network from traced edges and vertices."""
        from slavv_python.processing.stages.network.manager import NetworkManager

        typed_edges = edges if isinstance(edges, EdgeSet) else EdgeSet.from_dict(edges)
        typed_vertices = (
            vertices if isinstance(vertices, VertexSet) else VertexSet.from_dict(vertices)
        )
        return NetworkManager.run(typed_edges, typed_vertices, params)
