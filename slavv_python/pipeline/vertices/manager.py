"""Consolidated vertex extraction manager."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from slavv_python.engine.state.io import atomic_joblib_dump
from slavv_python.pipeline.policy import PipelinePolicy
from slavv_python.pipeline.vertices.detection import (
    choose_vertices_matlab_style,
    crop_vertices_matlab_style,
    matlab_vertex_candidates,
)
from slavv_python.pipeline.vertices.results import (
    build_vertices_result,
    coerce_radius_axes,
    empty_vertices_result,
    sort_vertex_order,
)
from slavv_python.schema.results import EnergyResult, VertexSet
from slavv_python.utils.safe_unpickle import safe_load

if TYPE_CHECKING:
    from slavv_python.engine.state import StageController

logger = logging.getLogger(__name__)


class VertexManager:
    """Deep facade for Vertex Set scan, crop/sort, choose/paint, and resumable artifacts."""

    @classmethod
    def run(cls, energy_data: EnergyResult, params: dict[str, Any]) -> VertexSet:
        """Extract vertices without run-directory checkpointing.

        Args:
            energy_data: Result from the energy stage containing energy map and scale indices.
            params: Pipeline parameters including energy_upper_bound, space_strel_apothem, etc.

        Returns:
            VertexSet: The extracted and filtered vertices.
        """
        return cls._run(energy_data, params, stage_controller=None)

    @classmethod
    def run_resumable(
        cls,
        energy_data: EnergyResult,
        params: dict[str, Any],
        stage_controller: StageController,
    ) -> VertexSet:
        """Extract vertices with persisted scan, crop, and choose state.

        Args:
            energy_data: Result from the energy stage.
            params: Pipeline parameters.
            stage_controller: Controller for managing stage state and artifacts.

        Returns:
            VertexSet: The extracted and filtered vertices.
        """
        return cls._run(energy_data, params, stage_controller=stage_controller)

    @classmethod
    def _run(
        cls,
        energy_data: EnergyResult,
        params: dict[str, Any],
        *,
        stage_controller: StageController | None,
    ) -> VertexSet:
        """Internal dispatcher for vertex extraction.

        Args:
            energy_data: Result from the energy stage.
            params: Pipeline parameters.
            stage_controller: Optional stage controller for resumable execution.

        Returns:
            VertexSet: The extracted and filtered vertices.
        """
        resumable = stage_controller is not None
        context = cls._build_context(energy_data, params)

        if resumable:
            assert stage_controller is not None
            return cls._run_resumable(context, stage_controller)
        return cls._run_ephemeral(context)

    @classmethod
    def _build_context(cls, energy_data: EnergyResult, params: dict[str, Any]) -> dict[str, Any]:
        """Consolidate energy data and parameters into a flat context dictionary."""
        lumen_radius_pixels = energy_data.lumen_radius_pixels
        policy = PipelinePolicy.from_params(params)
        return {
            "policy": policy,
            "energy": energy_data.energy,
            "scale_indices": energy_data.scale_indices,
            "lumen_radius_pixels": lumen_radius_pixels,
            "lumen_radius_pixels_axes": coerce_radius_axes(
                lumen_radius_pixels,
                energy_data.extra.get("lumen_radius_pixels_axes"),
            ),
            "energy_sign": float(energy_data.extra.get("energy_sign", -1.0)),
            "lumen_radius_microns": energy_data.lumen_radius_microns,
            "energy_upper_bound": params.get("energy_upper_bound", 0.0),
            "space_strel_apothem": params.get("space_strel_apothem", 1),
            "length_dilation_ratio": params.get("length_dilation_ratio", 1.0),
            "max_voxels_per_node": params.get("max_voxels_per_node", 6000),
            "block_size": int(params.get("resume_vertex_block_size", 256)),
            "n_jobs": int(params.get("n_jobs", 1)),
            "image_shape": energy_data.energy.shape,
        }

    @classmethod
    def _run_ephemeral(cls, context: dict[str, Any]) -> VertexSet:
        """Execute vertex extraction in-memory without persistence."""
        logger.info("Extracting vertices")
        policy = context["policy"]

        vertex_positions, vertex_scales, vertex_energies = matlab_vertex_candidates(
            context["energy"],
            context["scale_indices"],
            context["energy_sign"],
            context["energy_upper_bound"],
            context["space_strel_apothem"],
            context["lumen_radius_pixels_axes"][0],
            context["max_voxels_per_node"],
            policy=policy,
        )

        vertex_positions, vertex_scales, vertex_energies = crop_vertices_matlab_style(
            vertex_positions,
            vertex_scales,
            vertex_energies,
            context["image_shape"],
            context["lumen_radius_pixels_axes"],
            context["length_dilation_ratio"],
            policy=policy,
        )

        if len(vertex_positions) == 0:
            logger.info("Extracted 0 vertices")
            return empty_vertices_result()

        sort_indices = sort_vertex_order(
            vertex_positions,
            vertex_energies,
            context["image_shape"],
            context["energy_sign"],
        )
        vertex_positions = vertex_positions[sort_indices]
        vertex_scales = vertex_scales[sort_indices]
        vertex_energies = vertex_energies[sort_indices]

        chosen_mask = choose_vertices_matlab_style(
            vertex_positions,
            vertex_scales,
            context["image_shape"],
            context["lumen_radius_pixels_axes"],
            context["length_dilation_ratio"],
            policy=policy,
        )
        vertex_positions = vertex_positions[chosen_mask]
        vertex_scales = vertex_scales[chosen_mask]
        vertex_energies = vertex_energies[chosen_mask]

        logger.info("Extracted %s vertices", len(vertex_positions))
        return build_vertices_result(
            vertex_positions,
            vertex_scales,
            vertex_energies,
            context["lumen_radius_pixels"],
            context["lumen_radius_microns"],
        )

    @classmethod
    def _run_resumable(
        cls, context: dict[str, Any], stage_controller: StageController
    ) -> VertexSet:
        """Execute vertex extraction with checkpointing to disk.

        Args:
            context: Consolidated extraction context.
            stage_controller: Controller for managing stage state and artifacts.

        Returns:
            VertexSet: The extracted and filtered vertices.
        """
        candidate_path = stage_controller.artifact_path("candidates.pkl")
        cropped_path = stage_controller.artifact_path("cropped_candidates.pkl")
        chosen_mask_path = stage_controller.artifact_path("chosen_mask.pkl")
        choose_state = stage_controller.load_state()

        stage_controller.begin(
            detail="Scanning vertex candidates",
            units_total=3,
            substage="candidate_scan",
        )
        policy = context["policy"]
        if not candidate_path.exists():
            positions, scales, energies = matlab_vertex_candidates(
                context["energy"],
                context["scale_indices"],
                context["energy_sign"],
                context["energy_upper_bound"],
                context["space_strel_apothem"],
                context["lumen_radius_pixels_axes"][0],
                context["max_voxels_per_node"],
                n_jobs=context["n_jobs"],
                policy=policy,
            )
            atomic_joblib_dump(
                {"positions": positions, "scales": scales, "energies": energies},
                candidate_path,
            )
        stage_controller.update(units_total=3, units_completed=1, substage="candidate_scan")

        candidate_data = safe_load(candidate_path)
        vertex_positions = candidate_data["positions"]
        vertex_scales = candidate_data["scales"]
        vertex_energies = candidate_data["energies"]
        if len(vertex_positions) == 0:
            return cls._empty_vertex_set(context)

        if not cropped_path.exists():
            positions, scales, energies = crop_vertices_matlab_style(
                vertex_positions,
                vertex_scales,
                vertex_energies,
                context["image_shape"],
                context["lumen_radius_pixels_axes"],
                context["length_dilation_ratio"],
                policy=policy,
            )
            sort_indices = sort_vertex_order(
                positions,
                energies,
                context["image_shape"],
                context["energy_sign"],
            )
            atomic_joblib_dump(
                {
                    "positions": positions[sort_indices],
                    "scales": scales[sort_indices],
                    "energies": energies[sort_indices],
                },
                cropped_path,
            )
        stage_controller.update(units_total=3, units_completed=2, substage="crop_sort")

        ordered = safe_load(cropped_path)
        vertex_positions = ordered["positions"]
        vertex_scales = ordered["scales"]
        vertex_energies = ordered["energies"]
        if len(vertex_positions) == 0:
            return cls._empty_vertex_set(context)

        block_size = context["block_size"]
        chosen_mask = (
            safe_load(chosen_mask_path)
            if chosen_mask_path.exists()
            else np.zeros(len(vertex_positions), dtype=bool)
        )
        next_index = int(choose_state.get("next_index", 0))
        total_blocks = max(1, int(np.ceil(len(vertex_positions) / max(block_size, 1))))
        completed_blocks = min(total_blocks, next_index // max(block_size, 1))
        stage_controller.update(
            units_total=3 + total_blocks,
            units_completed=2 + completed_blocks,
            substage="choose_paint",
            detail=f"Vertex choose/paint {next_index}/{len(vertex_positions)}",
            resumed=next_index > 0,
        )

        for block_start in range(next_index, len(vertex_positions), max(block_size, 1)):
            block_end = min(len(vertex_positions), block_start + max(block_size, 1))
            chosen_mask = choose_vertices_matlab_style(
                vertex_positions,
                vertex_scales,
                context["image_shape"],
                context["lumen_radius_pixels_axes"],
                context["length_dilation_ratio"],
                start_index=block_start,
                end_index=block_end,
                chosen_mask=chosen_mask,
                policy=policy,
            )
            atomic_joblib_dump(chosen_mask, chosen_mask_path)
            stage_controller.save_state({"next_index": block_end, "block_size": block_size})
            completed_blocks = min(
                total_blocks,
                (block_end + block_size - 1) // max(block_size, 1),
            )
            stage_controller.update(
                units_total=3 + total_blocks,
                units_completed=2 + completed_blocks,
                substage="choose_paint",
                detail=f"Vertex choose/paint {block_end}/{len(vertex_positions)}",
                resumed=next_index > 0,
            )

        stage_controller.update(
            units_total=3 + total_blocks,
            units_completed=3 + total_blocks,
            substage="finalize",
        )
        return VertexSet.create(
            vertex_positions[chosen_mask],
            vertex_scales[chosen_mask],
            vertex_energies[chosen_mask],
            context["lumen_radius_pixels"],
            context["lumen_radius_microns"],
        )

    @staticmethod
    def _empty_vertex_set(context: dict[str, Any]) -> VertexSet:
        """Create an empty VertexSet with consistent spatial parameters.

        Args:
            context: Consolidated extraction context.

        Returns:
            VertexSet: An empty vertex set result.
        """
        return VertexSet.create(
            np.empty((0, 3), dtype=np.float64),
            np.empty((0,), dtype=np.int16),
            np.empty((0,), dtype=np.float64),
            context["lumen_radius_pixels"],
            context["lumen_radius_microns"],
        )
