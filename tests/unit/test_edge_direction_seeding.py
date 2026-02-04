import pathlib
import sys

import numpy as np

# Add source path for imports
try:
    from slavv.pipeline import SLAVVProcessor
except ImportError:
    from src.slavv.pipeline import SLAVVProcessor


from unittest.mock import patch


@patch(
    'src.slavv.tracing.estimate_vessel_directions',
    return_value=np.array([[0.0, 1.0, 0.0], [0.0, -1.0, 0.0]], dtype=float),
)
def test_extract_edges_seeds_directions_with_hessian(mock_generate_directions):
    processor = SLAVVProcessor()

    size = 21
    coords = np.indices((size, size, size))
    x = coords[1] - size // 2
    z = coords[2] - size // 2
    energy = -(x**2 + z**2).astype(float)

    vertex_pos = np.array([[10.0, 10.0, 10.0]], dtype=float)
    vertex_scales = np.array([0], dtype=int)
    energy_data = {
        "energy": energy,
        "lumen_radius_pixels": np.array([2.0], dtype=float),
        "lumen_radius_microns": np.array([2.0], dtype=float),
        "lumen_radius_pixels_axes": np.array([[2.0, 2.0, 2.0]], dtype=float),
        "energy_sign": -1.0,
    }
    vertices = {"positions": vertex_pos, "scales": vertex_scales}
    params = {
        "number_of_edges_per_vertex": 2,
        "step_size_per_origin_radius": 2.0,
        "length_dilation_ratio": 5.0,
        "microns_per_voxel": [1.0, 1.0, 1.0],
    }
    edges = processor.extract_edges(energy_data, vertices, params)
    assert len(edges["traces"]) == 2
    for trace in edges["traces"]:
        trace = np.asarray(trace)
        # y and z should remain constant while x moves monotonically
        assert np.allclose(trace[:, 0], 10.0)
        assert np.allclose(trace[:, 2], 10.0)
        x_diff = np.diff(trace[:, 1])
        assert np.all(x_diff > 0) or np.all(x_diff < 0)
