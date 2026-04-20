"""Tests for visualizer-driven MAT and JSON export behavior."""

import json
from pathlib import Path

import numpy as np
from dev.tests.support.payload_builders import build_processing_results

from slavv.io import load_network_from_mat
from slavv.visualization import NetworkVisualizer


def test_mat_export_via_visualizer(tmp_path: Path) -> None:
    """Test exporting network via ``NetworkVisualizer`` and reloading."""
    processing_results = build_processing_results(
        vertex_positions=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        edge_connections=[[0, 1]],
        edge_traces=[np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=float)],
        network={
            "strands": [np.array([0])],
            "bifurcations": np.array([], dtype=int),
            "vertex_degrees": np.array([1, 1], dtype=int),
        },
        parameters={"voxel_spacing": [1.0, 1.0, 1.0]},
    )
    processing_results["vertices"]["energies"] = np.array([-1.0, -2.0], dtype=float)
    processing_results["vertices"]["radii_microns"] = np.array([1.0, 1.5], dtype=float)
    processing_results["vertices"]["radii_pixels"] = np.array([2.0, 3.0], dtype=float)
    processing_results["vertices"]["scales"] = np.array([0.5, 0.5], dtype=float)

    out_path = tmp_path / "network.mat"
    NetworkVisualizer().export_network_data(processing_results, out_path, format="mat")

    loaded = load_network_from_mat(out_path)
    assert np.array_equal(loaded.vertices, processing_results["vertices"]["positions"])
    assert np.array_equal(loaded.edges, processing_results["edges"]["connections"])
    assert np.array_equal(loaded.radii, processing_results["vertices"]["radii_microns"])


def test_mat_export_complex_params(tmp_path: Path) -> None:
    """Test MAT export sanitizes parameters with nested mixed-value content."""
    processing_results = build_processing_results(
        vertices={"positions": [], "energies": [], "scales": []},
        edges={"connections": [], "traces": []},
        network={},
        parameters={
            "voxel_spacing": [1.0, 1.0, 1.0],
            "metadata": {
                "date": "2023-01-01",
                "ignore_this": None,
                "settings": {"threshold": 0.5},
            },
            "flags": {True, False},
            "none_value": None,
        },
    )

    out_path = tmp_path / "complex.mat"
    NetworkVisualizer().export_network_data(processing_results, out_path, format="mat")

    assert out_path.exists()


def test_json_export_handles_numpy_scalars_and_paths(tmp_path: Path) -> None:
    """Test JSON export sanitizes numpy scalars nested inside object containers."""
    processing_results = build_processing_results(
        vertices={
            "positions": np.array([[0, 0, 0]], dtype=np.int32),
            "energies": np.array([np.float32(-1.0)]),
            "radii": np.array([np.float32(1.5)]),
        },
        edges={
            "connections": np.array([[np.int32(0), np.int32(0)]], dtype=object),
            "traces": [np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int32)],
            "metadata": {"best_scale": np.int32(3)},
        },
        network={
            "strands": [np.array([np.int32(0)], dtype=object)],
            "bifurcations": np.array([np.int32(0)], dtype=np.int32),
            "adjacency": {np.int32(0): {np.int32(1), np.int32(2)}},
            "graph_edges": {np.int32(0): (np.int32(3), np.int32(4))},
        },
        parameters={"threshold": np.float32(0.5), "path": Path("nested") / "output"},
    )

    out_path = tmp_path / "network.json"
    NetworkVisualizer().export_network_data(processing_results, out_path, format="json")

    exported = json.loads(out_path.read_text(encoding="utf-8"))
    assert exported["edges"]["metadata"]["best_scale"] == 3
    assert exported["network"]["bifurcations"] == [0]
    assert sorted(exported["network"]["adjacency"]["0"]) == [1, 2]
    assert exported["network"]["graph_edges"]["0"] == [3, 4]
    assert exported["parameters"]["path"] == str(Path("nested") / "output")
