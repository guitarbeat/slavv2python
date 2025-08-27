import pathlib
import sys

import numpy as np

# Add source path for imports
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / 'slavv-streamlit' / 'src'))

from vectorization_core import calculate_network_statistics, calculate_surface_area


def test_calculate_network_statistics_basic():
    strands = [[0, 1, 2]]
    bifurcations = np.array([], dtype=np.int32)
    vertex_positions = np.array(
        [[0, 0, 0], [0, 1, 0], [0, 2, 0]], dtype=np.float32
    )
    radii = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    stats = calculate_network_statistics(
        strands, bifurcations, vertex_positions, radii, [1.0, 1.0, 1.0], (3, 3, 1)
    )
    assert stats["num_vertices"] == 3
    assert stats["num_strands"] == 1
    assert np.isclose(stats["total_length"], 2.0)
    assert np.isclose(stats["total_surface_area"], 4 * np.pi)
    assert np.isclose(stats["surface_area_density"], (4 * np.pi) / 9)
    assert np.isclose(stats["mean_tortuosity"], 1.0)
    assert np.isclose(stats["tortuosity_std"], 0.0)


def test_calculate_surface_area_direct():
    strands = [[0, 1, 2]]
    vertex_positions = np.array(
        [[0, 0, 0], [0, 1, 0], [0, 2, 0]], dtype=np.float32
    )
    radii = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    area = calculate_surface_area(strands, vertex_positions, radii, [1.0, 1.0, 1.0])
    assert np.isclose(area, 4 * np.pi)


def test_calculate_network_statistics_tortuosity():
    strands = [[0, 1, 2]]
    bifurcations = np.array([], dtype=np.int32)
    vertex_positions = np.array(
        [[0, 0, 0], [1, 0, 0], [1, 1, 0]], dtype=np.float32
    )
    radii = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    stats = calculate_network_statistics(
        strands, bifurcations, vertex_positions, radii, [1.0, 1.0, 1.0], (2, 2, 1)
    )
    expected_tortuosity = 2.0 / np.sqrt(2)
    assert np.isclose(stats["mean_tortuosity"], expected_tortuosity)
    assert np.isclose(stats["tortuosity_std"], 0.0)
