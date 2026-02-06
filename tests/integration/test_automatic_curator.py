import pathlib
import sys

import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))
try:
    from slavv.ml_curator import AutomaticCurator
except ImportError:
    from source.slavv.analysis import AutomaticCurator


def test_automatic_vertex_curation_filters_low_energy_vertices():
    vertices = {
        "positions": np.array([[5, 5, 5], [5, 5, 5], [5, 5, 5]], dtype=float),
        "energies": np.array([-0.2, -0.05, -0.3]),
        "scales": np.ones(3),
        "radii_pixels": np.ones(3),
        "radii_microns": np.ones(3),
    }
    energy_data = {"energy": np.zeros((10, 10, 10)), "image_shape": (10, 10, 10)}
    params = {
        "vertex_energy_threshold": -0.1,
        "min_vertex_radius": 0.5,
        "boundary_margin": 1,
        "contrast_threshold": 0.0,
    }

    curator = AutomaticCurator()
    curated = curator.curate_vertices_automatic(vertices, energy_data, params)

    assert curated["positions"].shape[0] == 2
    np.testing.assert_array_equal(curated["original_indices"], [0, 2])


def test_automatic_edge_curation_filters_short_edges():
    edges = {
        "traces": [
            np.array([[0, 0, 0], [0, 0, 1]], dtype=float),
            np.array([[0, 0, 0], [0, 0, 2]], dtype=float),
        ],
        "connections": [(0, 1), (0, 1)],
        "vertex_positions": np.array([[0, 0, 0], [0, 0, 2]], dtype=float),
    }
    vertices = {"positions": np.array([[0, 0, 0], [0, 0, 2]], dtype=float)}
    params = {
        "min_edge_length": 1.5,
        "max_edge_tortuosity": 5.0,
        "max_connection_distance": 5.0,
    }

    curator = AutomaticCurator()
    curated = curator.curate_edges_automatic(edges, vertices, params)

    assert len(curated["traces"]) == 1
    np.testing.assert_array_equal(curated["original_indices"], [1])


def test_automatic_edge_curation_handles_curated_vertices():
    """Test edge curation with curated (reduced) vertex list."""
    # All vertices
    vertices = {
        "positions": np.array([[1, 1, 1], [5, 5, 5], [9, 9, 9]], dtype=float),
        "energies": np.array([-0.2, -0.2, -0.2]),
        "scales": np.ones(3),
        "radii_pixels": np.ones(3),
        "radii_microns": np.ones(3),
        "original_indices": np.arange(3),
    }

    # Edges connected to original vertex indices
    edges = {
        "traces": [
            np.array([[1, 1, 1], [5, 5, 5]], dtype=float),  # Connects 0 and 1
            np.array([[5, 5, 5], [9, 9, 9]], dtype=float),  # Connects 1 and 2
        ],
        "connections": [(0, 1), (1, 2)],
        "vertex_positions": vertices["positions"],
    }

    # Curated vertices, where vertex 0 is removed
    curated_vertices = {
        "positions": np.array([[5, 5, 5], [9, 9, 9]], dtype=float),
        "energies": np.array([-0.2, -0.2]),
        "scales": np.ones(2),
        "radii_pixels": np.ones(2),
        "radii_microns": np.ones(2),
        "original_indices": np.array([1, 2]),
    }

    params = {
        "min_edge_length": 0.1,
        "max_edge_tortuosity": 5.0,
        "max_connection_distance": 5.0,
    }

    curator = AutomaticCurator()
    # This call should fail with an IndexError before the fix
    curated_edges = curator.curate_edges_automatic(edges, curated_vertices, params)

    # After the fix, it should correctly identify and keep the valid edge (1->2)
    assert len(curated_edges["traces"]) == 1
    np.testing.assert_array_equal(curated_edges["original_indices"], [1])
