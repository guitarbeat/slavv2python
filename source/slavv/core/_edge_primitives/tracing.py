"""Tracing helpers for edge primitives."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
from typing_extensions import TypeAlias

from .lookup import near_vertex, vertex_at_position
from .terminals import _finalize_traced_edge

if TYPE_CHECKING:
    from scipy.spatial import cKDTree

TraceMetadata: TypeAlias = "dict[str, Any]"
TraceEdgeResult: TypeAlias = "list[np.ndarray] | tuple[list[np.ndarray], TraceMetadata]"


def trace_edge(
    energy: np.ndarray,
    start_pos: np.ndarray,
    direction: np.ndarray,
    step_size: float,
    max_edge_energy: float,
    vertex_positions: np.ndarray,
    vertex_scales: np.ndarray,
    lumen_radius_pixels: np.ndarray,
    lumen_radius_microns: np.ndarray,
    max_steps: int,
    microns_per_voxel: np.ndarray,
    energy_sign: float,
    discrete_steps: bool = False,
    vertex_center_image: np.ndarray | None = None,
    vertex_image: np.ndarray | None = None,
    tree: cKDTree | None = None,
    max_search_radius: float = 0.0,
    origin_vertex_idx: int | None = None,
    return_metadata: bool = False,
) -> list[np.ndarray] | tuple[list[np.ndarray], dict[str, Any]]:
    """Trace an edge through the energy field with adaptive step sizing."""
    if vertex_center_image is None:
        vertex_center_image = vertex_image

    trace: list[np.ndarray] = [np.asarray(start_pos, dtype=np.float32).copy()]
    stop_reason: str = "max_steps"
    direct_terminal_vertex: int | None = None

    def finish(reason: str, terminal_vertex: int | None = None) -> TraceEdgeResult:
        finalized_trace, metadata = _finalize_traced_edge(
            trace,
            stop_reason=reason,
            direct_terminal_vertex=terminal_vertex,
            vertex_center_image=vertex_center_image,
            vertex_positions=vertex_positions,
            vertex_scales=vertex_scales,
            lumen_radius_microns=lumen_radius_microns,
            microns_per_voxel=microns_per_voxel,
            origin_vertex=origin_vertex_idx if origin_vertex_idx is not None else -1,
            tree=tree,
            max_search_radius=max_search_radius,
        )
        return (finalized_trace, metadata) if return_metadata else finalized_trace

    current_pos_y, current_pos_x, current_pos_z = (
        float(start_pos[0]),
        float(start_pos[1]),
        float(start_pos[2]),
    )
    current_dir_y, current_dir_x, current_dir_z = (
        float(direction[0]),
        float(direction[1]),
        float(direction[2]),
    )

    inv_mpv_2x_y = 1.0 / (2.0 * float(microns_per_voxel[0]))
    inv_mpv_2x_x = 1.0 / (2.0 * float(microns_per_voxel[1]))
    inv_mpv_2x_z = 1.0 / (2.0 * float(microns_per_voxel[2]))

    dim_y: int = int(energy.shape[0])
    dim_x: int = int(energy.shape[1])
    dim_z: int = int(energy.shape[2])
    dim_y_minus_2: int = dim_y - 2
    dim_x_minus_2: int = dim_x - 2
    dim_z_minus_2: int = dim_z - 2

    res_v: TraceEdgeResult = finish("bounds")
    if dim_y < 3 or dim_x < 3 or dim_z < 3:
        return res_v

    pos_y = math.floor(current_pos_y)
    pos_x = math.floor(current_pos_x)
    pos_z = math.floor(current_pos_z)
    prev_energy = energy[pos_y, pos_x, pos_z]
    if not math.isfinite(prev_energy):
        return finish("nan")

    for _ in range(max_steps):
        attempt = 0
        while attempt < 10:
            next_pos_y = current_pos_y + current_dir_y * step_size
            next_pos_x = current_pos_x + current_dir_x * step_size
            next_pos_z = current_pos_z + current_dir_z * step_size
            if not (
                math.isfinite(next_pos_y)
                and math.isfinite(next_pos_x)
                and math.isfinite(next_pos_z)
            ):
                return finish("nan")

            if discrete_steps:
                r_next_pos_y = round(next_pos_y)
                r_next_pos_x = round(next_pos_x)
                r_next_pos_z = round(next_pos_z)
                if (
                    r_next_pos_y == round(current_pos_y)
                    and r_next_pos_x == round(current_pos_x)
                    and r_next_pos_z == round(current_pos_z)
                ):
                    return finish("max_steps")
                next_pos_y, next_pos_x, next_pos_z = (
                    float(r_next_pos_y),
                    float(r_next_pos_x),
                    float(r_next_pos_z),
                )

            if (
                next_pos_y < 0
                or next_pos_y >= dim_y
                or next_pos_x < 0
                or next_pos_x >= dim_x
                or next_pos_z < 0
                or next_pos_z >= dim_z
            ):
                return finish("bounds")

            pos_y = math.floor(next_pos_y)
            pos_x = math.floor(next_pos_x)
            pos_z = math.floor(next_pos_z)
            current_energy = energy[pos_y, pos_x, pos_z]
            if not math.isfinite(current_energy):
                return finish("nan")

            if (energy_sign < 0 and current_energy > max_edge_energy) or (
                energy_sign > 0 and current_energy < max_edge_energy
            ):
                return finish("energy_threshold")
            if (energy_sign < 0 and current_energy > prev_energy) or (
                energy_sign > 0 and current_energy < prev_energy
            ):
                step_size *= 0.5
                if step_size < 0.5:
                    return finish("energy_rise_step_halving")
                attempt += 1
                continue
            break

        current_pos_y, current_pos_x, current_pos_z = next_pos_y, next_pos_x, next_pos_z
        current_pos_arr = np.array([current_pos_y, current_pos_x, current_pos_z], dtype=np.float32)
        trace.append(current_pos_arr)

        prev_energy = current_energy

        pos_y_r: int = round(current_pos_y)
        pos_x_r: int = round(current_pos_x)
        pos_z_r: int = round(current_pos_z)

        gp_y: int = pos_y_r
        if gp_y < 1:
            gp_y = 1
        elif gp_y > dim_y_minus_2:
            gp_y = dim_y_minus_2

        gp_x: int = pos_x_r
        if gp_x < 1:
            gp_x = 1
        elif gp_x > dim_x_minus_2:
            gp_x = dim_x_minus_2

        gp_z: int = pos_z_r
        if gp_z < 1:
            gp_z = 1
        elif gp_z > dim_z_minus_2:
            gp_z = dim_z_minus_2

        grad_y = (energy[gp_y + 1, gp_x, gp_z] - energy[gp_y - 1, gp_x, gp_z]) * inv_mpv_2x_y
        grad_x = (energy[gp_y, gp_x + 1, gp_z] - energy[gp_y, gp_x - 1, gp_z]) * inv_mpv_2x_x
        grad_z = (energy[gp_y, gp_x, gp_z + 1] - energy[gp_y, gp_x, gp_z - 1]) * inv_mpv_2x_z

        grad_norm = math.sqrt(grad_y**2 + grad_x**2 + grad_z**2)

        if grad_norm > 1e-12:
            dot_prod = grad_y * current_dir_y + grad_x * current_dir_x + grad_z * current_dir_z

            perp_grad_y = grad_y - current_dir_y * dot_prod
            perp_grad_x = grad_x - current_dir_x * dot_prod
            perp_grad_z = grad_z - current_dir_z * dot_prod

            sign = 1.0 if energy_sign >= 0 else -1.0
            current_dir_y = current_dir_y - sign * perp_grad_y
            current_dir_x = current_dir_x - sign * perp_grad_x
            current_dir_z = current_dir_z - sign * perp_grad_z

            norm = math.sqrt(current_dir_y**2 + current_dir_x**2 + current_dir_z**2)
            if norm > 1e-12:
                inv_norm = 1.0 / norm
                current_dir_y *= inv_norm
                current_dir_x *= inv_norm
                current_dir_z *= inv_norm

        if vertex_center_image is not None:
            terminal_vertex_idx = vertex_at_position(current_pos_arr, vertex_center_image)
        else:
            terminal_vertex_idx = near_vertex(
                current_pos_arr,
                vertex_positions,
                vertex_scales,
                lumen_radius_microns,
                microns_per_voxel,
                tree=tree,
                max_search_radius=max_search_radius,
            )
        if origin_vertex_idx is not None and terminal_vertex_idx == origin_vertex_idx:
            terminal_vertex_idx = None
        if terminal_vertex_idx is not None:
            direct_terminal_vertex = int(terminal_vertex_idx)
            stop_reason = "direct_terminal_hit"
            break

    final_result: TraceEdgeResult = finish(stop_reason, direct_terminal_vertex)
    return final_result
