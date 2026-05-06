"""Unit tests for exact imported-MATLAB artifact proof helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.io import savemat

from source.io.matlab_exact_proof import (
    compare_exact_artifacts,
    load_normalized_matlab_edge_input_vertices,
    load_normalized_matlab_stage,
    normalize_python_stage_payload,
    sync_exact_vertex_checkpoint_from_matlab,
)

if TYPE_CHECKING:
    from pathlib import Path


def _cell(items: list[np.ndarray]) -> np.ndarray:
    cell = np.empty((len(items),), dtype=object)
    for index, item in enumerate(items):
        cell[index] = item
    return cell


def _write_mat(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    savemat(path, payload)
    return path


def test_load_normalized_matlab_vertices_converts_one_based_indices(tmp_path):
    mat_path = _write_mat(
        tmp_path / "vertices.mat",
        {
            "vertex_space_subscripts": np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            "vertex_scale_subscripts": np.array([2, 4], dtype=np.int16),
            "vertex_energies": np.array([-3.5, -1.25], dtype=np.float64),
        },
    )

    payload = load_normalized_matlab_stage(mat_path, "vertices")

    np.testing.assert_array_equal(
        payload["positions"],
        np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float64),
    )
    np.testing.assert_array_equal(payload["scales"], np.array([1, 3], dtype=np.int64))
    np.testing.assert_array_equal(payload["energies"], np.array([-3.5, -1.25], dtype=np.float64))


def test_energy_stage_normalization_supports_dense_native_proof_surface(tmp_path):
    mat_path = _write_mat(
        tmp_path / "energy.mat",
        {
            "energy": np.array([[[[-1.0]], [[-2.0]]]], dtype=np.float64).reshape(1, 2, 1),
            "scale_indices": np.array([[[1], [2]]], dtype=np.int16),
            "energy_4d": np.array([[[[-1.0, -2.0]]]], dtype=np.float64),
            "lumen_radius_microns": np.array([1.0, 2.0], dtype=np.float64),
        },
    )
    python_payload = {
        "energy": np.array([[[-1.0], [-2.0]]], dtype=np.float32),
        "scale_indices": np.array([[[0], [1]]], dtype=np.int16),
        "energy_4d": np.array([[[[-1.0, -2.0]]]], dtype=np.float32),
        "lumen_radius_microns": np.array([1.0, 2.0], dtype=np.float32),
    }

    matlab_energy = load_normalized_matlab_stage(mat_path, "energy")
    python_energy = normalize_python_stage_payload("energy", python_payload)
    report = compare_exact_artifacts(
        {"energy": matlab_energy},
        {"energy": python_energy},
        ("energy",),
    )

    np.testing.assert_array_equal(matlab_energy["scale_indices"], python_energy["scale_indices"])
    assert report["passed"] is True


def test_load_normalized_matlab_edges_normalizes_bridge_payloads(tmp_path):
    mat_path = _write_mat(
        tmp_path / "edges.mat",
        {
            "edges2vertices": np.array([[1, 2]], dtype=np.int16),
            "edge_space_subscripts": _cell(
                [
                    np.array(
                        [
                            [2.0, 3.0, 4.0],
                            [3.0, 4.0, 5.0],
                        ],
                        dtype=np.float64,
                    )
                ]
            ),
            "edge_scale_subscripts": _cell([np.array([3.0, 2.5], dtype=np.float64)]),
            "edge_energies": _cell([np.array([-4.0, -3.0], dtype=np.float64)]),
            "mean_edge_energies": np.array([-3.5], dtype=np.float64),
            "bridge_vertex_space_subscripts": np.array([[4.0, 5.0, 6.0]], dtype=np.float64),
            "bridge_vertex_scale_subscripts": np.array([3], dtype=np.int16),
            "bridge_vertex_energies": np.array([-5.0], dtype=np.float64),
            "bridge_edges2vertices": np.array([[4, 0]], dtype=np.int16),
            "bridge_edge_space_subscripts": _cell(
                [
                    np.array(
                        [
                            [4.0, 5.0, 6.0],
                            [5.0, 6.0, 7.0],
                        ],
                        dtype=np.float64,
                    )
                ]
            ),
            "bridge_edge_scale_subscripts": _cell([np.array([3.0, 2.0], dtype=np.float64)]),
            "bridge_edge_energies": _cell([np.array([-6.0, -5.0], dtype=np.float64)]),
            "bridge_mean_edge_energies": np.array([-5.5], dtype=np.float64),
        },
    )

    payload = load_normalized_matlab_stage(mat_path, "edges")

    np.testing.assert_array_equal(payload["connections"], np.array([[0, 1]], dtype=np.int64))
    np.testing.assert_array_equal(
        payload["traces"][0],
        np.array([[1.0, 2.0, 3.0], [2.0, 3.0, 4.0]], dtype=np.float64),
    )
    np.testing.assert_array_equal(
        payload["scale_traces"][0],
        np.array([2.0, 1.5], dtype=np.float64),
    )
    np.testing.assert_array_equal(
        payload["bridge_vertex_positions"],
        np.array([[3.0, 4.0, 5.0]], dtype=np.float64),
    )
    np.testing.assert_array_equal(payload["bridge_vertex_scales"], np.array([2], dtype=np.int64))
    np.testing.assert_array_equal(
        payload["bridge_edges"]["connections"],
        np.array([[3, -1]], dtype=np.int64),
    )
    np.testing.assert_array_equal(
        payload["bridge_edges"]["traces"][0],
        np.array([[3.0, 4.0, 5.0], [4.0, 5.0, 6.0]], dtype=np.float64),
    )
    np.testing.assert_array_equal(
        payload["bridge_edges"]["energies"],
        np.array([-5.5], dtype=np.float64),
    )


def test_load_normalized_matlab_network_normalizes_empty_payloads(tmp_path):
    mat_path = _write_mat(
        tmp_path / "network.mat",
        {
            "strands2vertices": np.empty((0, 2), dtype=np.int16),
            "bifurcation_vertices": np.empty((0,), dtype=np.int16),
            "strand_subscripts": np.empty((0,), dtype=object),
            "strand_energies": np.empty((0,), dtype=object),
            "mean_strand_energies": np.empty((0,), dtype=np.float64),
            "vessel_directions": np.empty((0,), dtype=object),
        },
    )

    payload = load_normalized_matlab_stage(mat_path, "network")

    assert payload["strands"] == []
    np.testing.assert_array_equal(payload["bifurcations"], np.empty((0,), dtype=np.int64))
    assert payload["strand_subscripts"] == []
    assert payload["strand_energy_traces"] == []
    np.testing.assert_array_equal(payload["mean_strand_energies"], np.empty((0,), dtype=np.float64))
    assert payload["vessel_directions"] == []


def test_find_matlab_vector_paths_prefers_curated_vertices(tmp_path):
    from source.io.matlab_exact_proof import find_matlab_vector_paths

    vectors_dir = tmp_path / "batch" / "vectors"
    data_dir = tmp_path / "batch" / "data"
    vectors_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    _write_mat(
        data_dir / "energy_1.mat",
        {
            "energy": np.zeros((1, 1, 1), dtype=np.float64),
            "scale_indices": np.ones((1, 1, 1), dtype=np.int16),
            "energy_4d": np.zeros((1, 1, 1, 1), dtype=np.float64),
            "lumen_radius_microns": np.array([1.0], dtype=np.float64),
        },
    )
    _write_mat(
        vectors_dir / "vertices_1.mat",
        {
            "vertex_space_subscripts": np.array([[1.0, 2.0, 3.0]]),
            "vertex_scale_subscripts": np.array([2], dtype=np.int16),
            "vertex_energies": np.array([-1.0], dtype=np.float64),
        },
    )
    _write_mat(
        vectors_dir / "curated_vertices_1.mat",
        {
            "vertex_space_subscripts": np.array([[4.0, 5.0, 6.0]]),
            "vertex_scale_subscripts": np.array([3], dtype=np.int16),
            "vertex_energies": np.array([-2.0], dtype=np.float64),
        },
    )
    _write_mat(
        vectors_dir / "edges_1.mat",
        {
            "edges2vertices": np.empty((0, 2), dtype=np.int16),
            "edge_space_subscripts": np.empty((0,), dtype=object),
            "edge_scale_subscripts": np.empty((0,), dtype=object),
            "edge_energies": np.empty((0,), dtype=object),
            "mean_edge_energies": np.empty((0,), dtype=np.float64),
        },
    )
    _write_mat(
        vectors_dir / "network_1.mat",
        {
            "strands2vertices": np.empty((0, 2), dtype=np.int16),
            "bifurcation_vertices": np.empty((0,), dtype=np.int16),
            "strand_subscripts": np.empty((0,), dtype=object),
            "strand_energies": np.empty((0,), dtype=object),
            "mean_strand_energies": np.empty((0,), dtype=np.float64),
            "vessel_directions": np.empty((0,), dtype=object),
        },
    )

    paths = find_matlab_vector_paths(vectors_dir.parent, ("energy", "vertices", "edges", "network"))

    assert paths["energy"].name == "energy_1.mat"
    assert paths["vertices"].name == "curated_vertices_1.mat"


def test_load_normalized_matlab_edge_input_vertices_prefers_embedded_edge_surface(tmp_path):
    batch_dir = tmp_path / "batch"
    vectors_dir = batch_dir / "vectors"
    vectors_dir.mkdir(parents=True, exist_ok=True)
    _write_mat(
        vectors_dir / "curated_vertices_1.mat",
        {
            "vertex_space_subscripts": np.array([[100.0, 200.0, 10.0]], dtype=np.float64),
            "vertex_scale_subscripts": np.array([9.2], dtype=np.float64),
            "vertex_energies": np.array([-9.0], dtype=np.float64),
        },
    )
    _write_mat(
        vectors_dir / "edges_1.mat",
        {
            "edges2vertices": np.empty((0, 2), dtype=np.int16),
            "edge_space_subscripts": np.empty((0,), dtype=object),
            "edge_scale_subscripts": np.empty((0,), dtype=object),
            "edge_energies": np.empty((0,), dtype=object),
            "mean_edge_energies": np.empty((0,), dtype=np.float64),
            "vertex_space_subscripts": np.array(
                [[11.0, 21.0, 3.0], [31.0, 41.0, 5.0]], dtype=np.float64
            ),
            "vertex_scale_subscripts": np.array([2.6, 4.4], dtype=np.float64),
            "vertex_energies": np.array([-3.0, -5.0], dtype=np.float64),
        },
    )
    _write_mat(
        vectors_dir / "network_1.mat",
        {
            "strands2vertices": np.empty((0, 2), dtype=np.int16),
            "bifurcation_vertices": np.empty((0,), dtype=np.int16),
            "strand_subscripts": np.empty((0,), dtype=object),
            "strand_energies": np.empty((0,), dtype=object),
            "mean_strand_energies": np.empty((0,), dtype=np.float64),
            "vessel_directions": np.empty((0,), dtype=object),
        },
    )

    payload = load_normalized_matlab_edge_input_vertices(batch_dir)

    assert payload is not None
    np.testing.assert_array_equal(
        payload["positions"],
        np.array([[10.0, 20.0, 2.0], [30.0, 40.0, 4.0]], dtype=np.float64),
    )
    np.testing.assert_array_equal(payload["scales"], np.array([2, 3], dtype=np.int64))
    np.testing.assert_array_equal(payload["energies"], np.array([-3.0, -5.0], dtype=np.float64))


def test_sync_exact_vertex_checkpoint_from_matlab_overwrites_parity_fields(tmp_path):
    batch_dir = tmp_path / "batch"
    vectors_dir = batch_dir / "vectors"
    vectors_dir.mkdir(parents=True, exist_ok=True)
    _write_mat(
        vectors_dir / "curated_vertices_1.mat",
        {
            "vertex_space_subscripts": np.array(
                [[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float64
            ),
            "vertex_scale_subscripts": np.array([3, 5], dtype=np.int16),
            "vertex_energies": np.array([-9.0, -7.0], dtype=np.float64),
        },
    )
    _write_mat(
        vectors_dir / "edges_1.mat",
        {
            "edges2vertices": np.empty((0, 2), dtype=np.int16),
            "edge_space_subscripts": np.empty((0,), dtype=object),
            "edge_scale_subscripts": np.empty((0,), dtype=object),
            "edge_energies": np.empty((0,), dtype=object),
            "mean_edge_energies": np.empty((0,), dtype=np.float64),
        },
    )
    _write_mat(
        vectors_dir / "network_1.mat",
        {
            "strands2vertices": np.empty((0, 2), dtype=np.int16),
            "bifurcation_vertices": np.empty((0,), dtype=np.int16),
            "strand_subscripts": np.empty((0,), dtype=object),
            "strand_energies": np.empty((0,), dtype=object),
            "mean_strand_energies": np.empty((0,), dtype=np.float64),
            "vessel_directions": np.empty((0,), dtype=object),
        },
    )

    checkpoint_path = tmp_path / "checkpoint_vertices.pkl"
    from joblib import dump, load

    dump(
        {
            "positions": np.zeros((2, 3), dtype=np.float32),
            "scales": np.zeros((2,), dtype=np.int16),
            "energies": np.zeros((2,), dtype=np.float32),
            "radii_microns": np.array([1.0, 2.0], dtype=np.float32),
            "count": 2,
        },
        checkpoint_path,
    )

    updated = sync_exact_vertex_checkpoint_from_matlab(checkpoint_path, batch_dir)
    reloaded = load(checkpoint_path)

    expected_positions = np.array([[3.0, 4.0, 5.0], [6.0, 7.0, 8.0]], dtype=np.float32)
    expected_scales = np.array([2, 4], dtype=np.int16)
    expected_energies = np.array([-9.0, -7.0], dtype=np.float32)

    np.testing.assert_array_equal(updated["positions"], expected_positions)
    np.testing.assert_array_equal(updated["scales"], expected_scales)
    np.testing.assert_array_equal(updated["energies"], expected_energies)
    np.testing.assert_array_equal(reloaded["positions"], expected_positions)
    np.testing.assert_array_equal(reloaded["radii_microns"], np.array([1.0, 2.0], dtype=np.float32))


def test_compare_exact_artifacts_passes_on_exact_match():
    report = compare_exact_artifacts(
        {
            "vertices": {
                "positions": np.array([[0.0, 1.0, 2.0]], dtype=np.float64),
                "scales": np.array([1], dtype=np.int64),
                "energies": np.array([-2.0], dtype=np.float64),
            }
        },
        {
            "vertices": {
                "positions": np.array([[0.0, 1.0, 2.0]], dtype=np.float64),
                "scales": np.array([1], dtype=np.int64),
                "energies": np.array([-2.0], dtype=np.float64),
            }
        },
        ("vertices",),
    )

    assert report["passed"] is True
    assert report["first_failure"] is None


def test_compare_exact_artifacts_reports_ordering_mismatch():
    matlab_edges = {
        "connections": np.array([[0, 1], [1, 2]], dtype=np.int64),
        "traces": [],
        "scale_traces": [],
        "energy_traces": [],
        "energies": np.array([], dtype=np.float64),
        "bridge_vertex_positions": np.empty((0, 3), dtype=np.float64),
        "bridge_vertex_scales": np.empty((0,), dtype=np.int64),
        "bridge_vertex_energies": np.empty((0,), dtype=np.float64),
        "bridge_edges": {
            "connections": np.empty((0, 2), dtype=np.int64),
            "traces": [],
            "scale_traces": [],
            "energy_traces": [],
            "energies": np.empty((0,), dtype=np.float64),
        },
    }
    python_edges = dict(matlab_edges)
    python_edges["connections"] = np.array([[1, 2], [0, 1]], dtype=np.int64)

    report = compare_exact_artifacts(
        {"edges": matlab_edges},
        {"edges": python_edges},
        ("edges",),
    )

    assert report["passed"] is False
    assert report["first_failing_stage"] == "edges"
    assert report["first_failure"]["mismatch_type"] == "ordering mismatch"
    assert report["first_failure"]["field_path"] == "edges.connections"


def test_compare_exact_artifacts_reports_shape_mismatch():
    report = compare_exact_artifacts(
        {
            "network": {
                "strands": [np.array([0, 1], dtype=np.int64)],
                "bifurcations": np.array([], dtype=np.int64),
                "strand_subscripts": [np.array([[0.0, 0.0, 0.0, 0.0]], dtype=np.float64)],
                "strand_energy_traces": [np.array([-1.0], dtype=np.float64)],
                "mean_strand_energies": np.array([-1.0], dtype=np.float64),
                "vessel_directions": [np.array([[1.0, 0.0, 0.0]], dtype=np.float64)],
            }
        },
        {
            "network": {
                "strands": [np.array([0, 1], dtype=np.int64)],
                "bifurcations": np.array([], dtype=np.int64),
                "strand_subscripts": [np.array([[0.0, 0.0, 0.0]], dtype=np.float64)],
                "strand_energy_traces": [np.array([-1.0], dtype=np.float64)],
                "mean_strand_energies": np.array([-1.0], dtype=np.float64),
                "vessel_directions": [np.array([[1.0, 0.0, 0.0]], dtype=np.float64)],
            }
        },
        ("network",),
    )

    assert report["passed"] is False
    assert report["first_failure"]["mismatch_type"] == "shape mismatch"
    assert report["first_failure"]["field_path"] == "network.strand_subscripts[0]"


def test_compare_exact_artifacts_reports_missing_field():
    report = compare_exact_artifacts(
        {
            "vertices": {
                "positions": np.array([[0.0, 1.0, 2.0]], dtype=np.float64),
                "scales": np.array([1], dtype=np.int64),
                "energies": np.array([-2.0], dtype=np.float64),
            }
        },
        {
            "vertices": {
                "positions": np.array([[0.0, 1.0, 2.0]], dtype=np.float64),
                "scales": np.array([1], dtype=np.int64),
            }
        },
        ("vertices",),
    )

    assert report["passed"] is False
    assert report["first_failure"]["mismatch_type"] == "missing field"
    assert report["first_failure"]["field_path"] == "vertices.energies"
