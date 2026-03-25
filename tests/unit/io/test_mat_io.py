"""Consolidated tests for MAT and JSON network I/O."""

import json
from pathlib import Path

import numpy as np
from scipy.io import savemat

from slavv.io import load_network_from_mat
from slavv.visualization import NetworkVisualizer


def test_mat_roundtrip(tmp_path: Path) -> None:
    """Test loading a MAT file with network data."""
    vertices = np.array([[0, 0, 0], [1, 1, 1]], dtype=float)
    edges = np.array([[0, 1]], dtype=int)
    radii = np.array([1.0, 2.0], dtype=float)
    mat_path = tmp_path / "network.mat"
    savemat(mat_path, {"vertices": vertices, "edges": edges, "radii": radii})

    network = load_network_from_mat(mat_path)

    assert np.array_equal(network.vertices, vertices)
    assert np.array_equal(network.edges, edges)
    assert np.array_equal(network.radii, radii)


def test_mat_export_via_visualizer(tmp_path: Path) -> None:
    """Test exporting network via NetworkVisualizer and reloading."""
    vertices = {
        "positions": np.array([[0, 0, 0], [1, 1, 1]], dtype=float),
        "energies": np.array([-1.0, -2.0], dtype=float),
        "radii_microns": np.array([1.0, 1.5], dtype=float),
        "radii_pixels": np.array([2.0, 3.0], dtype=float),
        "scales": np.array([0.5, 0.5], dtype=float),
    }
    edges = {
        "connections": np.array([[0, 1]], dtype=int),
        "traces": [np.array([[0, 0, 0], [1, 1, 1]], dtype=float)],
    }
    network = {
        "strands": [np.array([0])],
        "bifurcations": np.array([], dtype=int),
        "vertex_degrees": np.array([1, 1], dtype=int),
    }
    processing_results = {
        "vertices": vertices,
        "edges": edges,
        "network": network,
        "parameters": {"voxel_spacing": [1.0, 1.0, 1.0]},
    }

    out_path = tmp_path / "network.mat"
    NetworkVisualizer().export_network_data(processing_results, out_path, format="mat")

    loaded = load_network_from_mat(out_path)
    assert np.array_equal(loaded.vertices, vertices["positions"])
    assert np.array_equal(loaded.edges, edges["connections"])
    assert np.array_equal(loaded.radii, vertices["radii_microns"])


def test_mat_export_complex_params(tmp_path: Path) -> None:
    """Test exporting parameters with mixed types (Bug 13)."""
    vertices = {"positions": [], "energies": [], "scales": []}
    edges = {"connections": [], "traces": []}
    network = {}

    # Complex parameters that would fail without sanitization
    parameters = {
        "voxel_spacing": [1.0, 1.0, 1.0],
        "metadata": {"date": "2023-01-01", "ignore_this": None, "settings": {"threshold": 0.5}},
        "flags": {True, False},  # Set (not JSON/MAT serializable usually)
        "none_value": None,
    }

    processing_results = {
        "vertices": vertices,
        "edges": edges,
        "network": network,
        "parameters": parameters,
    }

    out_path = tmp_path / "complex.mat"
    # Should not raise error
    NetworkVisualizer().export_network_data(processing_results, out_path, format="mat")

    assert out_path.exists()


def test_load_empty_mat_network_shapes(tmp_path: Path) -> None:
    """Test MAT import normalizes empty vertices/edges to 2D network shapes."""
    mat_path = tmp_path / "empty_network.mat"
    savemat(
        mat_path,
        {
            "vertices": np.empty((0, 3), dtype=float),
            "edges": np.empty((0, 2), dtype=int),
        },
    )

    network = load_network_from_mat(mat_path)

    assert network.vertices.shape == (0, 3)
    assert network.edges.shape == (0, 2)
    assert network.radii is None


def test_json_export_handles_numpy_scalars_and_paths(tmp_path: Path) -> None:
    """Test JSON export sanitizes numpy scalars nested inside object containers."""
    processing_results = {
        "vertices": {
            "positions": np.array([[0, 0, 0]], dtype=np.int32),
            "energies": np.array([np.float32(-1.0)]),
            "radii": np.array([np.float32(1.5)]),
        },
        "edges": {
            "connections": np.array([[np.int32(0), np.int32(0)]], dtype=object),
            "traces": [np.array([[0, 0, 0], [1, 1, 1]], dtype=np.int32)],
            "metadata": {"best_scale": np.int32(3)},
        },
        "network": {
            "strands": [np.array([np.int32(0)], dtype=object)],
            "bifurcations": np.array([np.int32(0)], dtype=np.int32),
            "adjacency": {np.int32(0): {np.int32(1), np.int32(2)}},
            "graph_edges": {np.int32(0): (np.int32(3), np.int32(4))},
        },
        "parameters": {"threshold": np.float32(0.5), "path": Path("nested") / "output"},
    }

    out_path = tmp_path / "network.json"
    NetworkVisualizer().export_network_data(processing_results, out_path, format="json")

    exported = json.loads(out_path.read_text(encoding="utf-8"))
    assert exported["edges"]["metadata"]["best_scale"] == 3
    assert exported["network"]["bifurcations"] == [0]
    assert sorted(exported["network"]["adjacency"]["0"]) == [1, 2]
    assert exported["network"]["graph_edges"]["0"] == [3, 4]
    assert exported["parameters"]["path"] == str(Path("nested") / "output")
