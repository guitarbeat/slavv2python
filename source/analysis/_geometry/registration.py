from __future__ import annotations

from typing import Any, cast

import numpy as np
from scipy.spatial import cKDTree

from .vector_math import safe_normalize_rows


def evaluate_registration(
        vectors_after: np.ndarray,
        vectors_before: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Evaluate registration by finding pairwise best match scores between vector sets."""
    vectors_after_arr = np.asarray(vectors_after, dtype=float)
    vectors_before_arr = np.asarray(vectors_before, dtype=float)

    if vectors_after_arr.ndim != 2 or vectors_before_arr.ndim != 2:
        raise ValueError("Vectors must be 2D arrays")

    if vectors_after_arr.size == 0 or vectors_before_arr.size == 0:
        return 0.0, np.array([]), np.array([])

    normalized_after = safe_normalize_rows(vectors_after_arr)
    normalized_before = safe_normalize_rows(vectors_before_arr)
    sim_matrix = np.clip(normalized_before @ normalized_after.T, -1.0, 1.0)

    best_before_to_after = np.max(sim_matrix, axis=1)
    best_after_to_before = np.max(sim_matrix, axis=0)
    reg_score = float(np.sum(best_before_to_after) * np.sum(best_after_to_before))
    return reg_score, best_before_to_after, best_after_to_before


def transform_vector_set(
        positions: np.ndarray,
        *,
        matrix: np.ndarray | None = None,
        scale: list[float] | None = None,
        rotation: np.ndarray | None = None,
        translate: list[float] | None = None,
) -> np.ndarray:
    """Apply geometric transforms to a set of positions."""
    pts = np.asarray(positions, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("positions must have shape (N, 3)")
    if matrix is not None:
        transform_matrix = np.asarray(matrix, dtype=float)
        if transform_matrix.shape != (4, 4):
            raise ValueError("matrix must be 4x4")
        homo = np.c_[pts, np.ones((pts.shape[0], 1))]
        out = homo @ transform_matrix.T
        return cast("np.ndarray", out[:, :3])
    out = pts.copy()
    if scale is not None:
        scale_arr = np.asarray(scale, dtype=float)
        if scale_arr.shape != (3,):
            raise ValueError("scale must be length-3")
        out = out * scale_arr
    if rotation is not None:
        rotation_arr = np.asarray(rotation, dtype=float)
        if rotation_arr.shape != (3, 3):
            raise ValueError("rotation must be 3x3")
        out = out @ rotation_arr.T
    if translate is not None:
        translate_arr = np.asarray(translate, dtype=float)
        if translate_arr.shape != (3,):
            raise ValueError("translate must be length-3")
        out = out + translate_arr
    return cast("np.ndarray", out)


def icp_register_rigid(
        source: np.ndarray,
        target: np.ndarray,
        *,
        with_scale: bool = False,
        max_iters: int = 50,
        tol: float = 1e-6,
) -> tuple[np.ndarray, float]:
    """Iterative closest point (rigid) registration using Kabsch per iteration."""
    source_arr = np.asarray(source, dtype=float)
    target_arr = np.asarray(target, dtype=float)
    if (
            source_arr.ndim != 2
            or target_arr.ndim != 2
            or source_arr.shape[1] != 3
            or target_arr.shape[1] != 3
    ):
        raise ValueError("source and target must be (N,3) arrays")
    if len(source_arr) == 0 or len(target_arr) == 0:
        return np.eye(4), 0.0

    source_mean = source_arr.mean(axis=0)
    target_mean = target_arr.mean(axis=0)
    scale = 1.0
    rotation = np.eye(3)
    translate = target_mean - source_mean

    tree = cKDTree(target_arr)
    prev_err = np.inf

    for _ in range(max_iters):
        source_projected = (source_arr @ rotation.T) * scale + translate
        _dists, idx = tree.query(source_projected, k=1)
        target_matched = target_arr[idx]

        source_centered = source_arr - source_arr.mean(axis=0)
        target_centered = target_matched - target_matched.mean(axis=0)
        covariance = source_centered.T @ target_centered
        u_vals, singular_vals, v_transpose = np.linalg.svd(covariance)
        rotation_kabsch = v_transpose.T @ u_vals.T
        if np.linalg.det(rotation_kabsch) < 0:
            v_transpose[-1, :] *= -1
            rotation_kabsch = v_transpose.T @ u_vals.T
        scale_kabsch = 1.0
        if with_scale:
            denom = float(np.sum(source_centered ** 2))
            if denom > 0:
                scale_kabsch = float(np.sum(singular_vals) / denom)
        translate_kabsch = target_matched.mean(axis=0) - scale_kabsch * (
                rotation_kabsch @ source_arr.mean(axis=0)
        )

        rotation = rotation_kabsch @ rotation
        scale = scale_kabsch * scale
        translate = rotation_kabsch @ translate + translate_kabsch

        source_projected = (source_arr @ rotation.T) * scale + translate
        err = float(np.sqrt(np.mean(np.sum((source_projected - target_matched) ** 2, axis=1))))
        if abs(prev_err - err) < tol:
            prev_err = err
            break
        prev_err = err

    transform = np.eye(4)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = translate
    return transform, prev_err


def register_vector_sets(
        source: np.ndarray,
        target: np.ndarray,
        *,
        method: str = "rigid",
        with_scale: bool = False,
        return_error: bool = False,
) -> Any:
    """Register source points to target points and return a 4x4 transform."""
    source_arr = np.asarray(source, dtype=float)
    target_arr = np.asarray(target, dtype=float)
    if source_arr.shape != target_arr.shape or source_arr.ndim != 2 or source_arr.shape[1] != 3:
        raise ValueError("source and target must have shape (N, 3)")
    num_points = source_arr.shape[0]
    if method not in {"rigid", "affine"}:
        raise ValueError("method must be 'rigid' or 'affine'")

    if method == "affine":
        if num_points < 4:
            raise ValueError("affine registration requires at least 4 points")
        source_h = np.c_[source_arr, np.ones((num_points, 1))]
        affine_matrix, *_ = np.linalg.lstsq(source_h, target_arr, rcond=None)
        transform = np.eye(4)
        transform[:3, :3] = affine_matrix[:3, :].T
        transform[:3, 3] = affine_matrix[3, :]
        projected = source_h @ affine_matrix
        err = float(np.sqrt(np.mean(np.sum((projected - target_arr) ** 2, axis=1))))
        return (transform, err) if return_error else transform

    if num_points < 3:
        raise ValueError("rigid registration requires at least 3 points")
    source_mean = source_arr.mean(axis=0)
    target_mean = target_arr.mean(axis=0)
    source_centered = source_arr - source_mean
    target_centered = target_arr - target_mean
    covariance = source_centered.T @ target_centered
    u_vals, singular_vals, v_transpose = np.linalg.svd(covariance)
    rotation = v_transpose.T @ u_vals.T
    if np.linalg.det(rotation) < 0:
        v_transpose[-1, :] *= -1
        rotation = v_transpose.T @ u_vals.T
    scale = 1.0
    if with_scale:
        denom = float(np.sum(source_centered ** 2))
        if denom > 0:
            scale = float(np.sum(singular_vals) / denom)
    translate = target_mean - scale * (rotation @ source_mean)
    transform = np.eye(4)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = translate
    projected = (np.c_[source_arr, np.ones((num_points, 1))] @ transform.T)[:, :3]
    err = float(np.sqrt(np.mean(np.sum((projected - target_arr) ** 2, axis=1))))
    return (transform, err) if return_error else transform


def register_strands(
        vertices_a: np.ndarray,
        edges_a: np.ndarray,
        vertices_b: np.ndarray,
        edges_b: np.ndarray,
        *,
        method: str = "rigid",
        with_scale: bool = False,
        match_threshold: float = 2.0,
        max_iters: int = 50,
) -> dict[str, Any]:
    """Register and merge two networks (A onto B) and return the merged result."""
    vertices_a_arr = np.asarray(vertices_a, dtype=float)
    vertices_b_arr = np.asarray(vertices_b, dtype=float)
    edges_a_arr = np.atleast_2d(np.asarray(edges_a, dtype=int))
    edges_b_arr = np.atleast_2d(np.asarray(edges_b, dtype=int))

    if vertices_a_arr.size == 0:
        return {
            "vertices": vertices_b_arr.copy(),
            "edges": edges_b_arr.copy(),
            "transform": np.eye(4),
            "rms": 0.0,
        }
    if vertices_b_arr.size == 0:
        return {
            "vertices": vertices_a_arr.copy(),
            "edges": edges_a_arr.copy(),
            "transform": np.eye(4),
            "rms": 0.0,
        }

    if method == "rigid":
        transform, rms = icp_register_rigid(
            vertices_a_arr, vertices_b_arr, with_scale=with_scale, max_iters=max_iters
        )
    elif method == "affine":
        tree = cKDTree(vertices_b_arr)
        _, idx = tree.query(vertices_a_arr, k=1)
        transform, rms = register_vector_sets(
            vertices_a_arr, vertices_b_arr[idx], method="affine", return_error=True
        )
    else:
        raise ValueError("method must be 'rigid' or 'affine'")

    vertices_a_h = np.c_[vertices_a_arr, np.ones((vertices_a_arr.shape[0], 1))]
    vertices_a_transformed = (vertices_a_h @ transform.T)[:, :3]

    tree = cKDTree(vertices_b_arr)
    dists, idx = tree.query(vertices_a_transformed, k=1)
    merged_vertices = vertices_b_arr.tolist()
    a_to_merged = np.empty((vertices_a_arr.shape[0],), dtype=int)
    for row_idx, (distance, target_idx) in enumerate(zip(dists, idx)):
        if np.isfinite(distance) and distance <= match_threshold:
            a_to_merged[row_idx] = int(target_idx)
        else:
            a_to_merged[row_idx] = len(merged_vertices)
            merged_vertices.append(vertices_a_transformed[row_idx].tolist())

    merged_vertices_arr = np.asarray(merged_vertices, dtype=float)
    edges_a_mapped = np.vstack(
        [
            np.minimum(a_to_merged[edges_a_arr[:, 0]], a_to_merged[edges_a_arr[:, 1]]),
            np.maximum(a_to_merged[edges_a_arr[:, 0]], a_to_merged[edges_a_arr[:, 1]]),
        ]
    ).T
    edges_b_norm = np.vstack(
        [
            np.minimum(edges_b_arr[:, 0], edges_b_arr[:, 1]),
            np.maximum(edges_b_arr[:, 0], edges_b_arr[:, 1]),
        ]
    ).T

    all_edges = np.vstack([edges_b_norm, edges_a_mapped])
    all_edges = all_edges[all_edges[:, 0] != all_edges[:, 1]]
    if all_edges.size:
        view = np.ascontiguousarray(all_edges).view(
            [("a", all_edges.dtype), ("b", all_edges.dtype)]
        )
        _, idx_unique = np.unique(view, return_index=True)
        merged_edges = all_edges[np.sort(idx_unique)]
    else:
        merged_edges = all_edges

    return {
        "vertices": merged_vertices_arr,
        "edges": merged_edges.astype(int),
        "transform": transform,
        "rms": rms,
    }
