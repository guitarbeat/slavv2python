"""Tests for MAT network I/O."""

from pathlib import Path

import numpy as np
from dev.tests.support.network_builders import build_network_object
from scipy.io import savemat
from source.io import load_network_from_mat


def test_mat_roundtrip(tmp_path: Path) -> None:
    """Test loading a MAT file with network data."""
    network = build_network_object(
        vertices=[[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
        edges=[[0, 1]],
        radii=[1.0, 2.0],
    )
    mat_path = tmp_path / "network.mat"
    savemat(
        mat_path,
        {
            "vertices": network.vertices,
            "edges": network.edges,
            "radii": network.radii,
        },
    )

    loaded = load_network_from_mat(mat_path)

    assert np.array_equal(loaded.vertices, network.vertices)
    assert np.array_equal(loaded.edges, network.edges)
    assert np.array_equal(loaded.radii, network.radii)


def test_load_empty_mat_network_shapes(tmp_path: Path) -> None:
    """Test MAT import normalizes empty vertices and edges to 2D shapes."""
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


