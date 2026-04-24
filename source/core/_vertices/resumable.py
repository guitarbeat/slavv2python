"""Resumable vertex extraction workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from ...runtime.run_state import atomic_joblib_dump
from ...utils.safe_unpickle import safe_load
from .candidates import (
    choose_vertices_matlab_style,
    crop_vertices_matlab_style,
    matlab_vertex_candidates,
)
from .payloads import (
    build_vertices_result,
    coerce_radius_axes,
    empty_vertices_result,
    sort_vertex_order,
)

if TYPE_CHECKING:
    from source.runtime import StageController


def extract_vertices_resumable(
    energy_data: dict[str, Any],
    params: dict[str, Any],
    stage_controller: StageController,
) -> dict[str, Any]:
    """Extract vertices with persisted MATLAB-style scan, crop, and choose state."""
    energy = energy_data["energy"]
    scale_indices = energy_data["scale_indices"]
    lumen_radius_pixels = energy_data["lumen_radius_pixels"]
    lumen_radius_pixels_axes = coerce_radius_axes(
        lumen_radius_pixels,
        energy_data.get("lumen_radius_pixels_axes"),
    )
    energy_sign = energy_data.get("energy_sign", -1.0)
    lumen_radius_microns = energy_data["lumen_radius_microns"]
    energy_upper_bound = params.get("energy_upper_bound", 0.0)
    space_strel_apothem = params.get("space_strel_apothem", 1)
    length_dilation_ratio = params.get("length_dilation_ratio", 1.0)
    max_voxels_per_node = params.get("max_voxels_per_node", 6000)
    block_size = int(params.get("resume_vertex_block_size", 256))

    candidate_path = stage_controller.artifact_path("candidates.pkl")
    cropped_path = stage_controller.artifact_path("cropped_candidates.pkl")
    chosen_mask_path = stage_controller.artifact_path("chosen_mask.pkl")
    choose_state = stage_controller.load_state()

    stage_controller.begin(
        detail="Scanning MATLAB-style vertex candidates",
        units_total=3,
        substage="candidate_scan",
    )
    if not candidate_path.exists():
        positions, scales, energies = matlab_vertex_candidates(
            energy,
            scale_indices,
            energy_sign,
            energy_upper_bound,
            space_strel_apothem,
            lumen_radius_pixels_axes[0],
            max_voxels_per_node,
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
        return empty_vertices_result()

    if not cropped_path.exists():
        positions, scales, energies = crop_vertices_matlab_style(
            vertex_positions,
            vertex_scales,
            vertex_energies,
            energy.shape,
            lumen_radius_pixels_axes,
            length_dilation_ratio,
        )
        sort_indices = sort_vertex_order(positions, energies, energy.shape, energy_sign)
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
        return empty_vertices_result()

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
            energy.shape,
            lumen_radius_pixels_axes,
            length_dilation_ratio,
            start_index=block_start,
            end_index=block_end,
            chosen_mask=chosen_mask,
        )
        atomic_joblib_dump(chosen_mask, chosen_mask_path)
        stage_controller.save_state({"next_index": block_end, "block_size": block_size})
        completed_blocks = min(total_blocks, (block_end + block_size - 1) // max(block_size, 1))
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
    return build_vertices_result(
        vertex_positions[chosen_mask],
        vertex_scales[chosen_mask],
        vertex_energies[chosen_mask],
        lumen_radius_pixels,
        lumen_radius_microns,
    )


