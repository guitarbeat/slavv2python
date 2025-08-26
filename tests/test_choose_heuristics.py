import sys
import pathlib
import numpy as np

sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / 'slavv-streamlit'))

from src.ml_curator import choose_vertices, choose_edges


def test_choose_vertices_thresholds():
    vertices = {
        'energies': np.array([-4.0, -0.5, -2.0]),
        'radii_microns': np.array([3.0, 5.0, 1.0]),
    }
    idx = choose_vertices(vertices, min_energy=1.0, min_radius=2.0, energy_sign=-1.0)
    assert idx.tolist() == [0]


def test_choose_edges_thresholds():
    edges = {
        'traces': [
            np.array([[0, 0, 0], [0, 0, 3]], dtype=float),
            np.array([[0, 0, 0], [0, 1, 0]], dtype=float),
        ],
        'energies': np.array([-4.0, -0.5], dtype=float),
    }
    idx = choose_edges(edges, min_energy=1.0, min_length=2.0, energy_sign=-1.0)
    assert idx.tolist() == [0]
