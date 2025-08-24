import pathlib
import sys

import numpy as np

# Add source path for imports
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / 'slavv-streamlit' / 'src'))

from vectorization_core import calculate_network_statistics


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
    assert stats["total_length"] > 0
