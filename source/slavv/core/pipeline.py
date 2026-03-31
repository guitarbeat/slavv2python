"""
Main pipeline orchestration for SLAVV.
Coordinates the energy, tracing, and graph construction steps.
"""

from __future__ import annotations

import logging
import tempfile
from typing import TYPE_CHECKING, Any, Callable, cast

from . import energy, graph, tracing
from .. import utils
from ..runtime import ProgressEvent, RunContext
from ..runtime.run_state import (
    PIPELINE_STAGES,
    STATUS_RUNNING,
    atomic_write_json,
    fingerprint_array,
    fingerprint_jsonable,
)

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)


def _validate_stage_control(stage_name: str | None, option_name: str) -> None:
    if stage_name is not None and stage_name not in PIPELINE_STAGES:
        valid = ", ".join(PIPELINE_STAGES)
        raise ValueError(f"{option_name} must be one of: {valid}")


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
        _validate_stage_control(stop_after, "stop_after")
        _validate_stage_control(force_rerun_from, "force_rerun_from")

        logger.info("Starting SLAVV processing pipeline")

        if progress_callback:
            progress_callback(0.0, "start")

        parameters = utils.validate_parameters(parameters)
        input_fingerprint = fingerprint_array(image)
        params_fingerprint = fingerprint_jsonable(parameters)
        effective_run_dir = run_dir
        if effective_run_dir is None and checkpoint_dir is None and event_callback is not None:
            effective_run_dir = tempfile.mkdtemp(prefix="slavv_run_")

        run_context = None
        if effective_run_dir is not None or checkpoint_dir is not None:
            run_context = RunContext(
                run_dir=effective_run_dir,
                checkpoint_dir=checkpoint_dir,
                input_fingerprint=input_fingerprint,
                params_fingerprint=params_fingerprint,
                target_stage=stop_after or "network",
                provenance={
                    "source": "pipeline",
                    "image_shape": list(image.shape),
                    "stop_after": stop_after or "network",
                },
                event_callback=event_callback,
                legacy=checkpoint_dir is not None and effective_run_dir is None,
            )
            run_context.ensure_resume_allowed(
                input_fingerprint=input_fingerprint,
                params_fingerprint=params_fingerprint,
                force_rerun_from=force_rerun_from,
            )
            if force_rerun_from in PIPELINE_STAGES:
                run_context.reset_pipeline_state_from(force_rerun_from)
            params_path = (
                run_context.run_root / "checkpoint_params.json"
                if run_context.legacy
                else run_context.metadata_dir / "validated_params.json"
            )
            atomic_write_json(params_path, parameters)
            run_context.mark_run_status(
                STATUS_RUNNING,
                current_stage="preprocess",
                detail="Starting SLAVV processing pipeline",
            )

        force_rerun = dict.fromkeys(PIPELINE_STAGES, False)
        if force_rerun_from in PIPELINE_STAGES:
            start_idx = PIPELINE_STAGES.index(force_rerun_from)
            for stage_name in PIPELINE_STAGES[start_idx:]:
                force_rerun[stage_name] = True

        results = {"parameters": parameters}

        try:
            image = utils.preprocess_image(image, parameters)
        except Exception as exc:
            if run_context is not None:
                run_context.fail_stage("preprocess", exc)
            raise
        if run_context is not None:
            run_context.mark_preprocess_complete()
        if progress_callback:
            progress_callback(0.2, "preprocess")

        energy_data = self._resolve_energy_stage(
            image, parameters, run_context, force_rerun["energy"]
        )
        results["energy_data"] = energy_data
        if progress_callback:
            progress_callback(0.4, "energy")

        if stop_after == "energy":
            logger.info("Pipeline stopped after 'energy' stage as requested.")
            if run_context is not None:
                run_context.finalize_run(stop_after=stop_after)
            return results

        vertices = self._resolve_vertices_stage(
            energy_data,
            parameters,
            run_context,
            force_rerun["vertices"],
        )
        results["vertices"] = vertices
        if progress_callback:
            progress_callback(0.6, "vertices")

        if stop_after == "vertices":
            logger.info("Pipeline stopped after 'vertices' stage as requested.")
            if run_context is not None:
                run_context.finalize_run(stop_after=stop_after)
            return results

        edges = self._resolve_edges_stage(
            energy_data,
            vertices,
            parameters,
            run_context,
            force_rerun["edges"],
        )
        results["edges"] = edges
        if progress_callback:
            progress_callback(0.8, "edges")

        if stop_after == "edges":
            logger.info("Pipeline stopped after 'edges' stage as requested.")
            if run_context is not None:
                run_context.finalize_run(stop_after=stop_after)
            return results

        network = self._resolve_network_stage(
            edges,
            vertices,
            parameters,
            run_context,
            force_rerun["network"],
        )
        results["network"] = network
        if progress_callback:
            progress_callback(1.0, "network")

        logger.info("SLAVV processing pipeline completed")
        if run_context is not None:
            run_context.finalize_run(stop_after=stop_after)
        return results

    @staticmethod
    def _stage_artifacts(stage_controller) -> dict[str, str]:
        artifacts = {}
        if stage_controller.stage_dir.exists():
            for artifact in stage_controller.stage_dir.iterdir():
                if artifact.is_file() and artifact.name != "resume_state.json":
                    artifacts[artifact.name] = str(artifact)
        return artifacts

    def _resolve_energy_stage(
        self,
        image: np.ndarray,
        parameters: dict[str, Any],
        run_context: RunContext | None,
        force_rerun: bool,
    ) -> dict[str, Any]:
        if run_context is None:
            return self.calculate_energy_field(image, parameters)
        controller = run_context.stage("energy")
        if controller.checkpoint_path.exists() and not force_rerun:
            logger.info("Loading cached Energy Field from %s", controller.checkpoint_path)
            energy_data = cast(dict[str, Any], controller.load_checkpoint())
            controller.complete(
                detail="Loaded energy checkpoint",
                artifacts=self._stage_artifacts(controller),
                resumed=True,
            )
            return energy_data
        try:
            energy_data = cast(
                dict[str, Any],
                energy.calculate_energy_field_resumable(
                    image,
                    parameters,
                    controller,
                    utils.get_chunking_lattice,
                ),
            )
            controller.save_checkpoint(energy_data)
            controller.complete(
                detail="Energy field ready",
                artifacts=self._stage_artifacts(controller),
            )
            return energy_data
        except Exception as exc:
            run_context.fail_stage("energy", exc)
            raise

    def _resolve_vertices_stage(
        self,
        energy_data: dict[str, Any],
        parameters: dict[str, Any],
        run_context: RunContext | None,
        force_rerun: bool,
    ) -> dict[str, Any]:
        if run_context is None:
            return self.extract_vertices(energy_data, parameters)
        controller = run_context.stage("vertices")
        if controller.checkpoint_path.exists() and not force_rerun:
            logger.info("Loading cached Vertices from %s", controller.checkpoint_path)
            vertices = cast(dict[str, Any], controller.load_checkpoint())
            controller.complete(
                detail="Loaded vertex checkpoint",
                artifacts=self._stage_artifacts(controller),
                resumed=True,
            )
            return vertices
        try:
            vertices = cast(
                dict[str, Any], tracing.extract_vertices_resumable(energy_data, parameters, controller)
            )
            controller.save_checkpoint(vertices)
            controller.complete(
                detail="Vertices extracted",
                artifacts=self._stage_artifacts(controller),
            )
            return vertices
        except Exception as exc:
            run_context.fail_stage("vertices", exc)
            raise

    def _resolve_edges_stage(
        self,
        energy_data: dict[str, Any],
        vertices: dict[str, Any],
        parameters: dict[str, Any],
        run_context: RunContext | None,
        force_rerun: bool,
    ) -> dict[str, Any]:
        if run_context is None:
            edge_method = parameters.get("edge_method", "tracing")
            if edge_method == "watershed":
                return self.extract_edges_watershed(energy_data, vertices, parameters)
            return self.extract_edges(energy_data, vertices, parameters)
        controller = run_context.stage("edges")
        if controller.checkpoint_path.exists() and not force_rerun:
            logger.info("Loading cached Edges from %s", controller.checkpoint_path)
            edges = cast(dict[str, Any], controller.load_checkpoint())
            controller.complete(
                detail="Loaded edge checkpoint",
                artifacts=self._stage_artifacts(controller),
                resumed=True,
            )
            return edges
        try:
            edge_method = parameters.get("edge_method", "tracing")
            if edge_method == "watershed":
                edges = cast(
                    dict[str, Any],
                    tracing.extract_edges_watershed_resumable(
                        energy_data,
                        vertices,
                        parameters,
                        controller,
                    ),
                )
            else:
                edges = cast(
                    dict[str, Any],
                    tracing.extract_edges_resumable(
                        energy_data,
                        vertices,
                        parameters,
                        controller,
                    ),
                )
            controller.save_checkpoint(edges)
            controller.complete(
                detail="Edges extracted",
                artifacts=self._stage_artifacts(controller),
            )
            return edges
        except Exception as exc:
            run_context.fail_stage("edges", exc)
            raise

    def _resolve_network_stage(
        self,
        edges: dict[str, Any],
        vertices: dict[str, Any],
        parameters: dict[str, Any],
        run_context: RunContext | None,
        force_rerun: bool,
    ) -> dict[str, Any]:
        if run_context is None:
            return self.construct_network(edges, vertices, parameters)
        controller = run_context.stage("network")
        if controller.checkpoint_path.exists() and not force_rerun:
            logger.info("Loading cached Network from %s", controller.checkpoint_path)
            network = cast(dict[str, Any], controller.load_checkpoint())
            controller.complete(
                detail="Loaded network checkpoint",
                artifacts=self._stage_artifacts(controller),
                resumed=True,
            )
            return network
        try:
            network = cast(
                dict[str, Any],
                graph.construct_network_resumable(edges, vertices, parameters, controller),
            )
            controller.save_checkpoint(network)
            controller.complete(
                detail="Network constructed",
                artifacts=self._stage_artifacts(controller),
            )
            return network
        except Exception as exc:
            run_context.fail_stage("network", exc)
            raise

    def calculate_energy_field(self, image: np.ndarray, params: dict[str, Any]) -> dict[str, Any]:
        """Calculate multi-scale energy field using Hessian. Delegates to ``energy`` module."""
        from .. import utils as utils_module

        return energy.calculate_energy_field(
            image, params, utils_module.get_chunking_lattice
        )

    def extract_vertices(
        self, energy_data: dict[str, Any], params: dict[str, Any]
    ) -> dict[str, Any]:
        """Extract vertices as local extrema. Delegates to ``tracing`` module."""
        return tracing.extract_vertices(energy_data, params)

    def extract_edges(
        self, energy_data: dict[str, Any], vertices: dict[str, Any], params: dict[str, Any]
    ) -> dict[str, Any]:
        """Extract edges by tracing. Delegates to ``tracing`` module."""
        return tracing.extract_edges(energy_data, vertices, params)

    def extract_edges_watershed(
        self, energy_data: dict[str, Any], vertices: dict[str, Any], params: dict[str, Any]
    ) -> dict[str, Any]:
        """Extract edges by watershed. Delegates to ``tracing`` module."""
        return tracing.extract_edges_watershed(
            energy_data, vertices, params
        )

    def construct_network(
        self, edges: dict[str, Any], vertices: dict[str, Any], params: dict[str, Any]
    ) -> dict[str, Any]:
        """Construct network from traces. Delegates to ``graph`` module."""
        return graph.construct_network(edges, vertices, params)

    # Legacy private methods exposed for compatibility/internal use
    # These static methods can be attached to the class if needed, or kept as module calls
    # Since original code used `self._method`, we can map them.

    @staticmethod
    def _spherical_structuring_element(radius, mpv):
        return energy.spherical_structuring_element(radius, mpv)

    @staticmethod
    def _trace_edge(*args, **kwargs):
        return tracing.trace_edge(*args, **kwargs)

    @staticmethod
    def _generate_edge_directions(*args, **kwargs):
        return tracing.generate_edge_directions(*args, **kwargs)

    @staticmethod
    def _estimate_vessel_directions(*args, **kwargs):
        return tracing.estimate_vessel_directions(*args, **kwargs)

    @staticmethod
    def _near_vertex(*args, **kwargs):
        return tracing.near_vertex(*args, **kwargs)

    @staticmethod
    def _find_terminal_vertex(*args, **kwargs):
        return tracing.find_terminal_vertex(*args, **kwargs)

    @staticmethod
    def _compute_gradient(*args, **kwargs):
        return tracing.compute_gradient(*args, **kwargs)

    @staticmethod
    def _in_bounds(*args, **kwargs):
        return tracing.in_bounds(*args, **kwargs)

    @staticmethod
    def _trace_strand(*args, **kwargs):
        """Legacy method - delegates to sparse implementation.

        Note: Signature differs from original dense implementation.
        Use graph.trace_strand_sparse directly for new code.
        """
        return graph.trace_strand_sparse(*args, **kwargs)

    @staticmethod
    def _trace_strand_sparse(*args, **kwargs):
        return graph.trace_strand_sparse(*args, **kwargs)

    @staticmethod
    def _sort_and_validate_strands_sparse(*args, **kwargs):
        return graph.sort_and_validate_strands_sparse(*args, **kwargs)
