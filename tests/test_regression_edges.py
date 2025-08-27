import pathlib
import sys
import numpy as np

# Add source path for imports
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1] / 'slavv-streamlit' / 'src'))

from vectorization_core import SLAVVProcessor


def test_extract_edges_regression():
    expected_connections = np.array([[0, -1], [0, -1]], dtype=int)
    expected_traces = np.array(
        [
            [[10.0, 10.0, 10.0], [10.0, 14.0, 10.0], [10.0, 18.0, 10.0]],
            [[10.0, 10.0, 10.0], [10.0, 6.0, 10.0], [10.0, 2.0, 10.0]],
        ],
        dtype=float,
    )

    size = 21
    coords = np.indices((size, size, size))
    x = coords[1] - size // 2
    z = coords[2] - size // 2
    energy = -(x**2 + z**2).astype(float)

    energy_data = {
        'energy': energy,
        'lumen_radius_pixels': np.array([2.0], dtype=float),
        'lumen_radius_microns': np.array([2.0], dtype=float),
        'energy_sign': -1.0,
    }
    vertices = {'positions': np.array([[10.0, 10.0, 10.0]], dtype=float), 'scales': np.array([0], dtype=int)}
    params = {
        'number_of_edges_per_vertex': 2,
        'step_size_per_origin_radius': 2.0,
        'length_dilation_ratio': 5.0,
        'microns_per_voxel': [1.0, 1.0, 1.0],
    }

    processor = SLAVVProcessor()
    edges = processor.extract_edges(energy_data, vertices, params)
    connections = np.array(edges['connections'])
    traces = np.stack([np.array(t) for t in edges['traces']])

    assert np.array_equal(connections, expected_connections)
    assert np.allclose(traces, expected_traces)
