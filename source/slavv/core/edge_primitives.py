"""Low-level edge tracing primitives for SLAVV."""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import scipy.ndimage as ndi
from skimage import feature
from typing_extensions import TypeAlias

from ._edge_payloads import _empty_stop_reason_counts
from .energy import compute_gradient_impl

if TYPE_CHECKING:
    from scipy.spatial import cKDTree

logger = logging.getLogger(__name__)

Int16Array: TypeAlias = "np.ndarray"
Int32Array: TypeAlias = "np.ndarray"
Int64Array: TypeAlias = "np.ndarray"
Float32Array: TypeAlias = "np.ndarray"
Float64Array: TypeAlias = "np.ndarray"
BoolArray: TypeAlias = "np.ndarray"
TraceMetadata: TypeAlias = "dict[str, Any]"
TraceEdgeResult: TypeAlias = "list[np.ndarray] | tuple[list[np.ndarray], TraceMetadata]"


def in_bounds(pos: np.ndarray, shape: tuple[int, ...]) -> bool:
    """Check if the floored position lies within array bounds."""
    # Optimization for 3D case which is the bottleneck in tracing
    if len(shape) == 3:
        res_3d: bool = 0 <= pos[0] < shape[0] and 0 <= pos[1] < shape[1] and 0 <= pos[2] < shape[2]
        return res_3d

    pos_int = np.floor(pos).astype(int)
    res: bool = np.all((pos_int >= 0) & (pos_int < np.array(shape)))  # type: ignore[assignment]
    return res


def compute_gradient(
    energy: np.ndarray, pos: np.ndarray, microns_per_voxel: np.ndarray
) -> np.ndarray:
    """Compute gradient at ``pos`` using central differences (wrapper for implementation)."""
    pos_int = np.round(pos).astype(np.int64)
    # Ensure proper dtypes for Numba compatibility (if enabled in impl)
    energy_arr = np.ascontiguousarray(energy, dtype=np.float64)
    mpv_arr = np.asarray(microns_per_voxel, dtype=np.float64)
    res: np.ndarray = compute_gradient_impl(energy_arr, pos_int, mpv_arr)
    return res


def generate_edge_directions(n_directions: int, seed: int | None = None) -> np.ndarray:
    """Generate uniformly distributed unit vectors on the sphere.

    Parameters
    ----------
    n_directions : int
        Number of direction vectors to generate.
    seed : int, optional
        Random seed for reproducibility. If None, uses unseeded RNG.
    """
    if n_directions <= 0:
        res_empty: np.ndarray = np.empty((0, 3), dtype=np.float64)
        return res_empty
    if n_directions == 1:
        res_single: np.ndarray = np.array([[0, 0, 1]], dtype=np.float64)
        return res_single

    # Generate random points from a 3D standard normal distribution
    rng = np.random.default_rng(seed)
    points = rng.standard_normal((n_directions, 3))
    # Normalize to unit vectors
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    res_arr: np.ndarray = (points / norms).astype(np.float64)
    return res_arr


def vertex_at_position(pos: np.ndarray, vertex_image: np.ndarray) -> int | None:
    """
    Fast O(1) vertex lookup using pre-computed vertex volume image.

    Parameters
    ----------
    pos : np.ndarray
        Position in voxel coordinates [y, x, z]
    vertex_image : np.ndarray
        Volume where each voxel contains vertex index (1-indexed) or 0

    Returns
    -------
    vertex_idx : Optional[int]
        Vertex index (0-indexed) if position is within a vertex region, None otherwise
    """
    pos_int = np.floor(pos).astype(int)

    # Check bounds
    if not np.all((pos_int >= 0) & (pos_int < np.array(vertex_image.shape))):
        return None

    vertex_id = vertex_image[pos_int[0], pos_int[1], pos_int[2]]

    return int(vertex_id - 1) if vertex_id > 0 else None


def near_vertex(
    pos: np.ndarray,
    vertex_positions: np.ndarray,
    vertex_scales: np.ndarray,
    lumen_radius_microns: np.ndarray,
    microns_per_voxel: np.ndarray,
    tree: cKDTree | None = None,
    max_search_radius: float = 0.0,
) -> int | None:
    """Return the index of a nearby vertex if within its physical radius; otherwise None

    Uses a tolerance of 0.5 voxels to account for traces ending near but not exactly at vertices.
    """
    # Tolerance: 0.5 voxels in physical units (use average voxel size)
    tolerance_microns = 0.5 * np.mean(microns_per_voxel)

    if tree is not None:
        # Optimized spatial query
        pos_microns: Float64Array = np.asarray(pos * microns_per_voxel, dtype=np.float64)
        # Query candidates within max possible radius
        candidates = tree.query_ball_point(pos_microns, max_search_radius)
        for i in candidates:
            # Check specific radius for this candidate
            vertex_pos = vertex_positions[i]
            vertex_scale = vertex_scales[i]
            radius = lumen_radius_microns[vertex_scale]
            diff = pos_microns - (vertex_pos * microns_per_voxel)
            if np.linalg.norm(diff) <= radius + tolerance_microns:
                return int(i)
        return None
    # Fallback linear scan
    for i, (vertex_pos, vertex_scale) in enumerate(zip(vertex_positions, vertex_scales)):
        radius = lumen_radius_microns[vertex_scale]
        diff = (pos - vertex_pos) * microns_per_voxel
        if np.linalg.norm(diff) <= radius + tolerance_microns:
            return int(i)
    return None


def find_terminal_vertex(
    pos: np.ndarray,
    vertex_positions: np.ndarray,
    vertex_scales: np.ndarray,
    lumen_radius_microns: np.ndarray,
    microns_per_voxel: np.ndarray,
    tree: cKDTree | None = None,
    max_search_radius: float = 0.0,
) -> int | None:
    """Find the index of a terminal vertex near a given position, if any."""
    return near_vertex(
        pos,
        vertex_positions,
        vertex_scales,
        lumen_radius_microns,
        microns_per_voxel,
        tree=tree,
        max_search_radius=max_search_radius,
    )


def _clip_trace_indices(trace: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    """Convert a trace to clipped integer voxel indices."""
    clipped_coords: Int32Array = np.floor(np.asarray(trace, dtype=np.float32)[:, :3]).astype(
        np.int32,
        copy=False,
    )
    for axis in range(3):
        clipped_coords[:, axis] = np.clip(clipped_coords[:, axis], 0, shape[axis] - 1)
    return cast("np.ndarray", clipped_coords)


def _resolve_trace_terminal_vertex(
    edge_trace: list[np.ndarray] | np.ndarray,
    vertex_center_image: np.ndarray | None,
    vertex_positions: np.ndarray,
    vertex_scales: np.ndarray,
    lumen_radius_microns: np.ndarray,
    microns_per_voxel: np.ndarray,
    origin_vertex: int,
    tree: cKDTree | None = None,
    max_search_radius: float = 0.0,
    direct_terminal_vertex: int | None = None,
) -> tuple[int | None, str | None]:
    """Resolve a terminal vertex using MATLAB-style center hits plus tolerant fallback."""
    trace_array = np.asarray(edge_trace, dtype=np.float32).reshape(-1, 3)

    if direct_terminal_vertex is not None and direct_terminal_vertex != origin_vertex:
        return int(direct_terminal_vertex), "direct_hit"

    if len(trace_array) == 0:
        return None, None

    if vertex_center_image is not None:
        terminal_vertex = vertex_at_position(trace_array[-1], vertex_center_image)
        if terminal_vertex is not None and terminal_vertex != origin_vertex:
            return int(terminal_vertex), "direct_hit"

        for point in trace_array[-2::-1]:
            terminal_vertex = vertex_at_position(point, vertex_center_image)
            if terminal_vertex is not None and terminal_vertex != origin_vertex:
                return int(terminal_vertex), "reverse_center_hit"

    for point in trace_array[::-1]:
        terminal_vertex = near_vertex(
            point,
            vertex_positions,
            vertex_scales,
            lumen_radius_microns,
            microns_per_voxel,
            tree=tree,
            max_search_radius=max_search_radius,
        )
        if terminal_vertex is not None and terminal_vertex != origin_vertex:
            return int(terminal_vertex), "reverse_near_hit"

    return None, None


def _finalize_traced_edge(
    edge_trace: list[np.ndarray] | np.ndarray,
    *,
    stop_reason: str,
    direct_terminal_vertex: int | None,
    vertex_center_image: np.ndarray | None,
    vertex_positions: np.ndarray,
    vertex_scales: np.ndarray,
    lumen_radius_microns: np.ndarray,
    microns_per_voxel: np.ndarray,
    origin_vertex: int,
    tree: cKDTree | None = None,
    max_search_radius: float = 0.0,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Finalize a raw trace by resolving its terminal vertex and normalizing metadata."""
    trace_array = np.asarray(edge_trace, dtype=np.float32).reshape(-1, 3)
    final_trace = [point.copy() for point in trace_array]
    terminal_vertex, terminal_resolution = _resolve_trace_terminal_vertex(
        trace_array,
        vertex_center_image,
        vertex_positions,
        vertex_scales,
        lumen_radius_microns,
        microns_per_voxel,
        origin_vertex,
        tree=tree,
        max_search_radius=max_search_radius,
        direct_terminal_vertex=direct_terminal_vertex,
    )

    if terminal_vertex is not None:
        final_trace.append(np.asarray(vertex_positions[terminal_vertex], dtype=np.float32).copy())

    return final_trace, {
        "stop_reason": stop_reason,
        "terminal_vertex": terminal_vertex,
        "terminal_resolution": terminal_resolution,
    }


def _record_trace_diagnostics(
    diagnostics: dict[str, Any],
    trace_metadata: dict[str, Any],
) -> None:
    """Accumulate per-trace terminal-resolution and stop-reason diagnostics."""
    if stop_reason := trace_metadata.get("stop_reason"):
        stop_reason_counts = diagnostics.setdefault(
            "stop_reason_counts", _empty_stop_reason_counts()
        )
        stop_reason_counts[stop_reason] = int(stop_reason_counts.get(stop_reason, 0)) + 1

    terminal_resolution = trace_metadata.get("terminal_resolution")
    if terminal_resolution == "direct_hit":
        diagnostics["terminal_direct_hit_count"] += 1
    elif terminal_resolution == "reverse_center_hit":
        diagnostics["terminal_reverse_center_hit_count"] += 1
    elif terminal_resolution == "reverse_near_hit":
        diagnostics["terminal_reverse_near_hit_count"] += 1


def _trace_scale_series(edge_trace: np.ndarray, scale_indices: np.ndarray | None) -> np.ndarray:
    """Sample projected scale indices along an edge trace."""
    if scale_indices is None:
        empty_scale_trace: np.ndarray = np.zeros((len(edge_trace),), dtype=np.int16)
        return empty_scale_trace
    idx = _clip_trace_indices(edge_trace, scale_indices.shape)
    scale_trace: Int16Array = scale_indices[idx[:, 0], idx[:, 1], idx[:, 2]].astype(
        np.int16,
        copy=False,
    )
    return cast("np.ndarray", scale_trace)


def _trace_energy_series(edge_trace: np.ndarray, energy: np.ndarray) -> np.ndarray:
    """Sample projected energy values along an edge trace."""
    idx = _clip_trace_indices(edge_trace, energy.shape)
    energy_trace: Float32Array = energy[idx[:, 0], idx[:, 1], idx[:, 2]].astype(
        np.float32,
        copy=False,
    )
    return cast("np.ndarray", energy_trace)


def _edge_metric_from_energy_trace(energy_trace: np.ndarray) -> float:
    """Match MATLAB's current edge quality metric: minimum max-energy is best."""
    arr = np.asarray(energy_trace, dtype=np.float32)
    if arr.size == 0:
        return 0.0
    value = float(np.nanmax(arr))
    return -1000.0 if math.isnan(value) else value


def estimate_vessel_directions(
    energy: np.ndarray, pos: np.ndarray, radius: float, microns_per_voxel: np.ndarray
) -> np.ndarray:
    """Estimate vessel directions at a vertex via local Hessian analysis."""
    # Determine a small neighborhood around the vertex
    sigma = max(radius / 2.0, 1.0)
    center = np.round(pos).astype(int)
    r = int(max(1, np.ceil(sigma)))
    slices = tuple(slice(max(c - r, 0), min(c + r + 1, s)) for c, s in zip(center, energy.shape))
    patch = energy[slices]
    # Fallback to uniform directions if patch is too small
    if patch.ndim != 3 or min(patch.shape) < 3:
        return generate_edge_directions(2, seed=0)

    # Rescale patch to account for anisotropic voxel spacing
    scale = microns_per_voxel / microns_per_voxel.min()
    if not np.allclose(scale, 1):
        patch = ndi.zoom(patch, scale, order=1, mode="nearest")

    # --- EXPLANATION FOR JUNIOR DEVS ---
    # WHY: We need to find the direction of the vessel at this point.
    # HOW: We calculate the Hessian matrix (second-order partial derivatives) of intensity.
    #      The eigenvalues of the Hessian describe the local curvature:
    #      - Small eigenvector -> Direction of least curvature (along the vessel).
    #      - Large eigenvectors -> Direction of high curvature (across the vessel wall).
    #      We pick the eigenvector corresponding to the smallest absolute eigenvalue
    #      as the vessel direction.
    # -----------------------------------

    # Compute Hessian in the local patch and extract center values
    try:
        raw_hessian = feature.hessian_matrix(
            patch,
            sigma=sigma,
            mode="nearest",
            order="rc",
            use_gaussian_derivatives=False,
        )
    except TypeError:
        raw_hessian = feature.hessian_matrix(
            patch,
            sigma=sigma,
            mode="nearest",
            order="rc",
        )
    hessian_elems = [h * (radius**2) for h in raw_hessian]
    patch_center_arr: Int64Array = np.array(patch.shape, dtype=np.int64) // 2
    patch_center = tuple(int(value) for value in patch_center_arr.tolist())
    Hxx, Hxy, Hxz, Hyy, Hyz, Hzz = [h[patch_center] for h in hessian_elems]
    H = np.array(
        [
            [Hxx, Hxy, Hxz],
            [Hxy, Hyy, Hyz],
            [Hxz, Hyz, Hzz],
        ]
    )
    # Eigen decomposition to find principal axis
    try:
        w, v = np.linalg.eigh(H)
    except np.linalg.LinAlgError:
        return generate_edge_directions(2, seed=0)
    if not np.all(np.isfinite(w)):
        return generate_edge_directions(2, seed=0)

    # Fallback if eigenvalues are nearly isotropic or all zero
    w_abs = np.sort(np.abs(w))
    max_eig = w_abs[-1]
    if max_eig == 0 or (w_abs[1] - w_abs[0]) < 1e-6 * max_eig:
        return generate_edge_directions(2, seed=0)

    direction = v[:, np.argmin(np.abs(w))]
    norm = np.linalg.norm(direction)
    if norm == 0 or not np.isfinite(norm):
        return generate_edge_directions(2, seed=0)
    direction = direction / norm
    bidirectional: Float64Array = np.stack((direction, -direction))
    return cast("np.ndarray", bidirectional)


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

    # Tracing state
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

    # Scalarize position and direction
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

    # Precompute for optimized gradient calc
    inv_mpv_2x_y = 1.0 / (2.0 * float(microns_per_voxel[0]))
    inv_mpv_2x_x = 1.0 / (2.0 * float(microns_per_voxel[1]))
    inv_mpv_2x_z = 1.0 / (2.0 * float(microns_per_voxel[2]))

    # Precompute shape scalars
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
                # Rounding logic for discrete steps
                r_next_pos_y = round(next_pos_y)
                r_next_pos_x = round(next_pos_x)
                r_next_pos_z = round(next_pos_z)
                # Check if position changed
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

            # Inline bounds check for speed
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

        # Update current position
        current_pos_y, current_pos_x, current_pos_z = next_pos_y, next_pos_x, next_pos_z
        current_pos_arr = np.array([current_pos_y, current_pos_x, current_pos_z], dtype=np.float32)
        trace.append(current_pos_arr)

        prev_energy = current_energy

        # Optimized gradient computation
        # Use scalar args to avoid allocating arrays
        pos_y_r: int = round(current_pos_y)
        pos_x_r: int = round(current_pos_x)
        pos_z_r: int = round(current_pos_z)

        # Inline gradient computation to avoid function call and allocation
        # Manual clamping
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

        # Compute gradient components
        grad_y = (energy[gp_y + 1, gp_x, gp_z] - energy[gp_y - 1, gp_x, gp_z]) * inv_mpv_2x_y
        grad_x = (energy[gp_y, gp_x + 1, gp_z] - energy[gp_y, gp_x - 1, gp_z]) * inv_mpv_2x_x
        grad_z = (energy[gp_y, gp_x, gp_z + 1] - energy[gp_y, gp_x, gp_z - 1]) * inv_mpv_2x_z

        # Manual norm
        grad_norm = math.sqrt(grad_y**2 + grad_x**2 + grad_z**2)

        if grad_norm > 1e-12:
            # Project gradient onto plane perpendicular to current direction
            dot_prod = grad_y * current_dir_y + grad_x * current_dir_x + grad_z * current_dir_z

            perp_grad_y = grad_y - current_dir_y * dot_prod
            perp_grad_x = grad_x - current_dir_x * dot_prod
            perp_grad_z = grad_z - current_dir_z * dot_prod

            # Steer along ridge by opposing gradient direction
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
