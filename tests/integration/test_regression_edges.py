import pathlib
import sys
import numpy as np

# Add source path for imports
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

try:
    from slavv.pipeline import SLAVVProcessor
except ImportError:
    from source.slavv.pipeline import SLAVVProcessor


from unittest.mock import patch


# Patching the correct location in tracing module if we wanted to force directions,
# but here we fix the energy field to match the expectation.
# We remove the mock because the hessian should correctly find the direction
# if the energy field is correct.
def test_extract_edges_regression():
    expected_connections = np.array([[0, -1], [0, -1]], dtype=int)
    # Expectation: Traces along Z-axis (varying index 2)
    # Start at [10, 10, 10].
    # Step size 4. Next point [10, 10, 14] or [10, 10, 6].
    expected_traces = np.array(
        [
            [[10.0, 10.0, 10.0], [10.0, 10.0, 14.0], [10.0, 10.0, 18.0]],
            [[10.0, 10.0, 10.0], [10.0, 10.0, 6.0], [10.0, 10.0, 2.0]],
        ],
        dtype=float,
    )

    size = 21
    coords = np.indices((size, size, size))
    # Corrected energy field for Z-axis ridge: -(x^2 + y^2)
    # y is coords[0], x is coords[1], z is coords[2]
    y = coords[0] - size // 2
    x = coords[1] - size // 2
    # z = coords[2] - size // 2 # Unused for Z-cylinder

    # Energy max at x=0, y=0 -> Ridge along Z
    energy = -(x**2 + y**2).astype(float)

    energy_data = {
        'energy': energy,
        'lumen_radius_pixels': np.array([2.0], dtype=float),
        'lumen_radius_microns': np.array([2.0], dtype=float),
        'lumen_radius_pixels_axes': np.array([[2.0, 2.0, 2.0]], dtype=float),
        'energy_sign': -1.0,
    }
    vertices = {'positions': np.array([[10.0, 10.0, 10.0]], dtype=float), 'scales': np.array([0], dtype=int)}
    params = {
        'number_of_edges_per_vertex': 2,
        'step_size_per_origin_radius': 2.0,
        'length_dilation_ratio': 5.0,
        'microns_per_voxel': [1.0, 1.0, 1.0],
        # Ensure we use hessian to find the Z-direction from energy
        'direction_method': 'hessian'
    }

    processor = SLAVVProcessor()
    edges = processor.extract_edges(energy_data, vertices, params)
    connections = np.array(edges['connections'])

    #Sort traces to match expectation (positive Z vs negative Z)
    traces = []
    for t in edges['traces']:
        t = np.array(t)
        traces.append(t)

    # Simple sort by end point Z coordinate to match expected order
    # Expected[0] ends at 18 (high Z)
    # Expected[1] ends at 2 (low Z)
    traces.sort(key=lambda arr: arr[-1, 2], reverse=True)
    traces = np.stack(traces)

    assert np.array_equal(connections, expected_connections)
    assert np.allclose(traces, expected_traces, atol=1e-5)
