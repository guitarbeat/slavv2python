import pathlib
import sys

import numpy as np

# Add source path for imports
from slavv.analysis.geometry import (
    calculate_network_statistics,
    calculate_surface_area,
    calculate_vessel_volume,
)


def test_calculate_network_statistics_basic():
    strands = [[0, 1, 2]]
    bifurcations = np.array([], dtype=np.int32)
    vertex_positions = np.array(
        [[0, 0, 0], [0, 1, 0], [0, 2, 0]], dtype=np.float32
    )
    radii = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    edge_energies = np.array([-1.0, -2.0], dtype=np.float32)
    stats = calculate_network_statistics(
        strands,
        bifurcations,
        vertex_positions,
        radii,
        [1.0, 1.0, 1.0],
        (3, 3, 1),
        edge_energies,
    )
    assert stats["num_vertices"] == 3
    assert stats["num_strands"] == 1
    assert stats["num_edges"] == 2
    assert np.isclose(stats["total_length"], 2.0)
    assert np.isclose(stats["total_surface_area"], 4 * np.pi)
    assert np.isclose(stats["surface_area_density"], (4 * np.pi) / 9)
    assert np.isclose(stats["total_volume"], 2 * np.pi)
    assert np.isclose(stats["volume_fraction"], (2 * np.pi) / 9)
    assert np.isclose(stats["mean_tortuosity"], 1.0)
    assert np.isclose(stats["tortuosity_std"], 0.0)
    assert np.isclose(stats["mean_edge_energy"], -1.5)
    assert np.isclose(stats["edge_energy_std"], 0.5)
    assert np.isclose(stats["mean_edge_length"], 1.0)
    assert np.isclose(stats["edge_length_std"], 0.0)
    assert np.isclose(stats["mean_edge_radius"], 1.0)
    assert np.isclose(stats["edge_radius_std"], 0.0)
    assert np.isclose(stats["mean_degree"], 4 / 3)
    assert np.isclose(stats["degree_std"], np.std([1, 2, 1]))
    assert stats["num_connected_components"] == 1
    assert stats["num_endpoints"] == 2
    assert np.isclose(stats["avg_path_length"], 4 / 3)
    assert np.isclose(stats["clustering_coefficient"], 0.0)
    assert np.isclose(stats["network_diameter"], 2.0)
    assert np.isclose(stats["vertex_density"], 3 / 9)
    assert np.isclose(stats["edge_density"], 2 / 9)
    assert np.isclose(stats["betweenness_mean"], 1 / 3)
    assert np.isclose(stats["betweenness_std"], np.sqrt(2 / 9))
    assert np.isclose(stats["closeness_mean"], 7 / 9)
    assert np.isclose(stats["closeness_std"], np.sqrt(2) / 9)
    expected_eigen = np.array([0.5, np.sqrt(2) / 2, 0.5])
    assert np.isclose(stats["eigenvector_mean"], expected_eigen.mean())
    assert np.isclose(stats["eigenvector_std"], expected_eigen.std())
    assert np.isclose(stats["graph_density"], 2 / 3)


def test_calculate_surface_area_direct():
    strands = [[0, 1, 2]]
    vertex_positions = np.array(
        [[0, 0, 0], [0, 1, 0], [0, 2, 0]], dtype=np.float32
    )
    radii = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    area = calculate_surface_area(strands, vertex_positions, radii, [1.0, 1.0, 1.0])
    assert np.isclose(area, 4 * np.pi)


def test_calculate_vessel_volume_direct():
    strands = [[0, 1, 2]]
    vertex_positions = np.array(
        [[0, 0, 0], [0, 1, 0], [0, 2, 0]], dtype=np.float32
    )
    radii = np.array([1.0, 1.0, 1.0], dtype=np.float32)
    volume = calculate_vessel_volume(strands, vertex_positions, radii, [1.0, 1.0, 1.0])
    assert np.isclose(volume, 2 * np.pi)


def test_edge_radius_stats_varying():
    strands = [[0, 1, 2]]
    bifurcations = np.array([], dtype=np.int32)
    vertex_positions = np.array(
        [[0, 0, 0], [0, 1, 0], [0, 2, 0]], dtype=np.float32
    )
    radii = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    stats = calculate_network_statistics(
        strands, bifurcations, vertex_positions, radii, [1.0, 1.0, 1.0], (3, 3, 1)
    )
    assert np.isclose(stats["mean_edge_radius"], 2.0)
    assert np.isclose(stats["edge_radius_std"], 0.5)


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


def test_calculate_branching_angles():
    strands = [[0, 1], [0, 2], [0, 3]]
    bifurcations = np.array([0], dtype=np.int32)
    vertex_positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [np.sqrt(3) / 2, -0.5, 0.0],
            [-np.sqrt(3) / 2, -0.5, 0.0],
        ],
        dtype=np.float32,
    )
    radii = np.ones(4, dtype=np.float32)
    stats = calculate_network_statistics(
        strands, bifurcations, vertex_positions, radii, [1.0, 1.0, 1.0], (3, 3, 1)
    )
    assert np.isclose(stats["mean_branch_angle"], 120.0)
    assert np.isclose(stats["branch_angle_std"], 0.0, atol=1e-6)
