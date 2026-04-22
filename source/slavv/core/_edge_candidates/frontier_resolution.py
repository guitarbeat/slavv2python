"""Frontier parent/child resolution helpers."""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from .common import (
    Int32Array,
    _path_max_energy_from_linear_indices,
)


def _prune_frontier_indices_beyond_found_vertices(
    candidate_coords: np.ndarray,
    origin_position_microns: np.ndarray,
    displacement_vectors: list[np.ndarray],
    microns_per_voxel: np.ndarray,
) -> np.ndarray:
    """Remove frontier voxels that lie beyond an already-found terminal direction."""
    if len(candidate_coords) == 0 or not displacement_vectors:
        return candidate_coords

    vectors_from_origin = (
        candidate_coords.astype(np.float64) * microns_per_voxel - origin_position_microns
    )
    indices_beyond: np.ndarray = np.zeros((len(candidate_coords),), dtype=bool)
    for displacement in displacement_vectors:
        indices_beyond |= np.sum(displacement * vectors_from_origin, axis=1) > 1.0
    kept_coords: Int32Array = candidate_coords[~indices_beyond]
    return cast("np.ndarray", kept_coords)


def _resolve_frontier_edge_connection(
    current_path_linear: list[int],
    terminal_vertex_idx: int,
    seed_origin_idx: int,
    edge_paths_linear: list[list[int]],
    edge_pairs: list[tuple[int, int]],
    pointer_index_map: dict[int, int],
    energy: np.ndarray,
    shape: tuple[int, int, int],
) -> tuple[int | None, int | None]:
    """Resolve MATLAB-style parent/child validity for a frontier-found terminal."""
    origin_idx, terminal_idx, _resolution_reason, _resolution_debug = (
        _normalize_frontier_resolution_result(
            _resolve_frontier_edge_connection_details(
                current_path_linear,
                terminal_vertex_idx,
                seed_origin_idx,
                edge_paths_linear,
                edge_pairs,
                pointer_index_map,
                energy,
                shape,
            )
        )
    )
    return origin_idx, terminal_idx


def _normalize_frontier_resolution_result(
    result: tuple[int | None, int | None, str] | tuple[int | None, int | None, str, dict[str, Any]],
) -> tuple[int | None, int | None, str, dict[str, Any]]:
    """Normalize frontier-resolution returns to the enriched four-field form."""
    if len(result) == 4:
        origin_idx, terminal_idx, resolution_reason, resolution_debug = cast(
            "tuple[int | None, int | None, str, dict[str, Any]]",
            result,
        )
        return origin_idx, terminal_idx, resolution_reason, dict(resolution_debug)
    origin_idx, terminal_idx, resolution_reason = cast(
        "tuple[int | None, int | None, str]",
        result,
    )
    return origin_idx, terminal_idx, resolution_reason, {}


def _resolve_frontier_edge_connection_details(
    current_path_linear: list[int],
    terminal_vertex_idx: int,
    seed_origin_idx: int,
    edge_paths_linear: list[list[int]],
    edge_pairs: list[tuple[int, int]],
    pointer_index_map: dict[int, int],
    energy: np.ndarray,
    shape: tuple[int, int, int],
) -> tuple[int | None, int | None, str, dict[str, Any]]:
    """Resolve MATLAB-style parent/child validity for a frontier-found terminal."""
    root_index = current_path_linear[-1]
    root_pointer = int(pointer_index_map.get(root_index, 0))
    parent_index = -root_pointer if root_pointer < 0 else 0
    resolution_debug: dict[str, Any] = {
        "root_linear_index": int(root_index),
        "root_pointer": root_pointer,
        "parent_edge_index": int(parent_index),
        "current_path_length": len(current_path_linear),
    }

    if parent_index == 0:
        return seed_origin_idx, terminal_vertex_idx, "accepted_seed_origin", resolution_debug

    parent_path = edge_paths_linear[parent_index - 1]
    resolution_debug["parent_path_length"] = len(parent_path)
    parent_pointers = {
        -int(pointer_index_map.get(index, 0))
        for index in parent_path
        if int(pointer_index_map.get(index, 0)) < 0
    }
    parent_pointers.discard(0)
    parent_pointers.discard(parent_index)
    resolution_debug["parent_child_edge_indices"] = sorted(
        int(pointer) for pointer in parent_pointers
    )
    if parent_pointers:
        return None, None, "rejected_parent_has_child", resolution_debug

    parent_terminal, parent_origin = edge_pairs[parent_index - 1]
    resolution_debug["parent_terminal_vertex_index"] = int(parent_terminal)
    resolution_debug["parent_origin_vertex_index"] = int(parent_origin)
    if parent_terminal < 0 or parent_origin < 0:
        return None, None, "rejected_parent_invalid", resolution_debug

    parent_energy = _path_max_energy_from_linear_indices(parent_path, energy, shape)
    child_energy = _path_max_energy_from_linear_indices(current_path_linear, energy, shape)
    resolution_debug["parent_path_max_energy"] = float(parent_energy)
    resolution_debug["child_path_max_energy"] = float(child_energy)
    if child_energy <= parent_energy:
        return None, None, "rejected_child_better_than_parent", resolution_debug

    if root_index not in parent_path:
        return None, None, "rejected_root_missing_from_parent", resolution_debug

    bifurcation_index = parent_path.index(root_index)
    resolution_debug["bifurcation_path_index"] = int(bifurcation_index)
    parent_1 = parent_path[:bifurcation_index]
    parent_2 = parent_path[bifurcation_index + 1 :]
    parent_1_energy = (
        _path_max_energy_from_linear_indices(parent_1, energy, shape)
        if parent_1
        else float("-inf")
    )
    resolution_debug["parent_terminal_half_length"] = len(parent_1)
    resolution_debug["parent_terminal_half_max_energy"] = float(parent_1_energy)
    half_candidates: list[tuple[int, float]] = [(parent_terminal, parent_1_energy)]
    if parent_2:
        parent_2_energy = _path_max_energy_from_linear_indices(parent_2, energy, shape)
        resolution_debug["parent_origin_half_length"] = len(parent_2)
        resolution_debug["parent_origin_half_max_energy"] = float(parent_2_energy)
        half_candidates.append((parent_origin, parent_2_energy))
    else:
        resolution_debug["parent_origin_half_length"] = 0
        resolution_debug["parent_origin_half_max_energy"] = None
    origin_vertex_idx = min(half_candidates, key=lambda item: item[1])[0]
    resolution_debug["resolved_origin_vertex_index"] = int(origin_vertex_idx)
    if origin_vertex_idx < 0:
        return None, None, "rejected_parent_origin_invalid", resolution_debug
    if origin_vertex_idx == parent_terminal:
        return (
            origin_vertex_idx,
            terminal_vertex_idx,
            "accepted_parent_terminal_half",
            resolution_debug,
        )
    return (
        origin_vertex_idx,
        terminal_vertex_idx,
        "accepted_parent_origin_half",
        resolution_debug,
    )


def _frontier_parent_child_outcome_from_reason(resolution_reason: str) -> str | None:
    """Map a frontier resolution reason onto parent/child lifecycle language."""
    if resolution_reason == "rejected_parent_has_child":
        return "parent_has_child"
    if resolution_reason == "rejected_child_better_than_parent":
        return "child_better_than_parent"
    if resolution_reason.startswith("accepted_parent_"):
        return "accepted_parent_child_resolution"
    return None


def _frontier_bifurcation_choice_from_reason(resolution_reason: str) -> str | None:
    """Map a frontier resolution reason onto the chosen parent half."""
    if resolution_reason == "accepted_parent_terminal_half":
        return "parent_terminal_half"
    if resolution_reason == "accepted_parent_origin_half":
        return "parent_origin_half"
    return None


def _frontier_claim_reassignment_from_reason(resolution_reason: str) -> str | None:
    """Map a frontier resolution reason onto claim-reassignment language."""
    if resolution_reason == "accepted_parent_terminal_half":
        return "reassigned_to_parent_terminal_half"
    if resolution_reason == "accepted_parent_origin_half":
        return "reassigned_to_parent_origin_half"
    return None


def _build_frontier_lifecycle_event(
    *,
    seed_origin_idx: int,
    terminal_vertex_idx: int,
    origin_idx: int | None,
    terminal_idx: int | None,
    resolution_reason: str,
    resolution_debug: dict[str, Any] | None,
    terminal_hit_sequence: int,
    local_candidate_index: int | None = None,
) -> dict[str, Any]:
    """Create a serializable frontier lifecycle event entry."""
    emitted_endpoint_pair: list[int] | None = None
    if (
        origin_idx is not None
        and terminal_idx is not None
        and origin_idx >= 0
        and terminal_idx >= 0
    ):
        start_vertex, end_vertex = int(origin_idx), int(terminal_idx)
        emitted_endpoint_pair = (
            [start_vertex, end_vertex] if start_vertex < end_vertex else [end_vertex, start_vertex]
        )

    survived_candidate_manifest = emitted_endpoint_pair is not None
    claim_reassigned = (
        survived_candidate_manifest
        and origin_idx is not None
        and int(origin_idx) != seed_origin_idx
    )
    claim_reassignment_reason = _frontier_claim_reassignment_from_reason(resolution_reason)
    return {
        "seed_origin_index": seed_origin_idx,
        "terminal_vertex_index": terminal_vertex_idx,
        "resolved_origin_index": None if origin_idx is None else origin_idx,
        "resolved_terminal_index": None if terminal_idx is None else terminal_idx,
        "emitted_endpoint_pair": emitted_endpoint_pair,
        "resolution_reason": resolution_reason,
        "resolution_debug": dict(resolution_debug or {}),
        "rejection_reason": (None if survived_candidate_manifest else resolution_reason),
        "parent_child_outcome": _frontier_parent_child_outcome_from_reason(resolution_reason),
        "bifurcation_choice": _frontier_bifurcation_choice_from_reason(resolution_reason),
        "claim_reassigned": claim_reassigned,
        "claim_reassignment_reason": (claim_reassignment_reason if claim_reassigned else None),
        "survived_candidate_manifest": survived_candidate_manifest,
        "origin_candidate_local_index": (
            None if local_candidate_index is None else local_candidate_index
        ),
        "manifest_candidate_index": None,
        "chosen_final_edge": False,
        "final_survival_stage": (
            "candidate_manifest" if survived_candidate_manifest else "pre_manifest_rejection"
        ),
        "terminal_hit_sequence": terminal_hit_sequence,
    }
