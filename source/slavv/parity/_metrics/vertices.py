from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
from scipy import stats
from scipy.spatial import cKDTree

from .counts import _infer_vertices_count
from .signatures import _as_position_array, _sample_counter_diff, _vertex_signatures


def match_vertices(
    matlab_positions: np.ndarray, python_positions: np.ndarray, distance_threshold: float = 3.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match vertices between MATLAB and Python using one-to-one nearest neighbors."""
    matlab_positions = _as_position_array(matlab_positions)
    python_positions = _as_position_array(python_positions)

    if matlab_positions.size == 0 or python_positions.size == 0:
        return np.array([]), np.array([]), np.array([])

    matlab_xyz = matlab_positions[:, :3]
    python_xyz = python_positions[:, :3]
    tree = cKDTree(python_xyz)
    distances, python_indices = tree.query(matlab_xyz)

    candidate_pairs = [
        (float(distance), int(matlab_index), int(python_index))
        for matlab_index, (distance, python_index) in enumerate(zip(distances, python_indices))
        if distance < distance_threshold
    ]
    candidate_pairs.sort()

    matched_matlab: list[int] = []
    matched_python: list[int] = []
    matched_distances: list[float] = []
    used_matlab: set[int] = set()
    used_python: set[int] = set()
    for distance, matlab_index, python_index in candidate_pairs:
        if matlab_index in used_matlab or python_index in used_python:
            continue
        used_matlab.add(matlab_index)
        used_python.add(python_index)
        matched_matlab.append(matlab_index)
        matched_python.append(python_index)
        matched_distances.append(distance)

    return (
        np.asarray(matched_matlab, dtype=np.int32),
        np.asarray(matched_python, dtype=np.int32),
        np.asarray(matched_distances, dtype=float),
    )


def compare_vertices(matlab_verts: dict[str, Any], python_verts: dict[str, Any]) -> dict[str, Any]:
    """Compare vertex information between MATLAB and Python."""
    comparison = {
        "matlab_count": _infer_vertices_count(matlab_verts),
        "python_count": _infer_vertices_count(python_verts),
        "count_difference": 0,
        "count_percent_difference": 0.0,
        "position_rmse": None,
        "matched_vertices": 0,
        "unmatched_matlab": 0,
        "unmatched_python": 0,
        "radius_correlation": None,
        "radius_stats": {},
        "exact_positions_scales_match": False,
        "exact_positions_scales_energies_match": False,
        "matlab_only_samples": [],
        "python_only_samples": [],
    }

    matlab_count = comparison["matlab_count"]
    python_count = comparison["python_count"]

    if matlab_count > 0 or python_count > 0:
        comparison["count_difference"] = abs(matlab_count - python_count)
        avg_count = (matlab_count + python_count) / 2.0
        if avg_count > 0:
            comparison["count_percent_difference"] = (comparison["count_difference"] / avg_count) * 100.0

    matlab_positions = _as_position_array(matlab_verts.get("positions", np.array([])))
    python_positions = _as_position_array(python_verts.get("positions", np.array([])))

    if matlab_positions.size > 0 and python_positions.size > 0:
        matlab_idx, python_idx, distances = match_vertices(matlab_positions, python_positions)
        unique_python_idx = np.unique(python_idx)

        comparison["matched_vertices"] = len(matlab_idx)
        comparison["unmatched_matlab"] = matlab_count - len(matlab_idx)
        comparison["unmatched_python"] = python_count - len(unique_python_idx)

        if len(distances) > 0:
            comparison["position_rmse"] = float(np.sqrt(np.mean(distances**2)))
            comparison["position_mean_distance"] = float(np.mean(distances))
            comparison["position_median_distance"] = float(np.median(distances))
            comparison["position_95th_percentile"] = float(np.percentile(distances, 95))

        matlab_radii = np.asarray(matlab_verts.get("radii", np.array([])))
        python_radii = np.asarray(python_verts.get("radii", np.array([])))

        if (
            len(matlab_idx) > 0
            and len(unique_python_idx) == len(matlab_idx)
            and matlab_radii.size > 0
            and python_radii.size > 0
        ):
            matched_matlab_radii = matlab_radii[matlab_idx]
            matched_python_radii = python_radii[python_idx]

            if len(matched_matlab_radii) > 1:
                pearson_r, pearson_p = stats.pearsonr(matched_matlab_radii, matched_python_radii)
                spearman_r, spearman_p = stats.spearmanr(matched_matlab_radii, matched_python_radii)
                comparison["radius_correlation"] = {
                    "pearson_r": float(pearson_r),
                    "pearson_p": float(pearson_p),
                    "spearman_r": float(spearman_r),
                    "spearman_p": float(spearman_p),
                }

            comparison["radius_stats"] = {
                "matlab_mean": float(np.mean(matched_matlab_radii)),
                "matlab_std": float(np.std(matched_matlab_radii)),
                "python_mean": float(np.mean(matched_python_radii)),
                "python_std": float(np.std(matched_python_radii)),
                "mean_difference": float(np.mean(matched_matlab_radii - matched_python_radii)),
                "rmse": float(np.sqrt(np.mean((matched_matlab_radii - matched_python_radii) ** 2))),
            }

    coords_scales_matlab, coords_scales_energy_matlab = _vertex_signatures(matlab_verts)
    coords_scales_python, coords_scales_energy_python = _vertex_signatures(python_verts)
    coords_scales_counter_matlab = Counter(coords_scales_matlab)
    coords_scales_counter_python = Counter(coords_scales_python)
    coords_scales_energy_counter_matlab = Counter(coords_scales_energy_matlab)
    coords_scales_energy_counter_python = Counter(coords_scales_energy_python)

    comparison["exact_positions_scales_match"] = coords_scales_counter_matlab == coords_scales_counter_python
    comparison["exact_positions_scales_energies_match"] = (
        coords_scales_energy_counter_matlab == coords_scales_energy_counter_python
    )
    comparison["matlab_only_samples"] = _sample_counter_diff(
        coords_scales_counter_matlab, coords_scales_counter_python
    )
    comparison["python_only_samples"] = _sample_counter_diff(
        coords_scales_counter_python, coords_scales_counter_matlab
    )

    return comparison
