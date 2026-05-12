import numpy as np
import pytest

from slavv_python.core import SLAVVProcessor
from slavv_python.utils import validate_parameters


def test_validate_parameters_direction_method():
    params = validate_parameters({"direction_method": "uniform"})
    assert params["direction_method"] == "uniform"
    with pytest.raises(ValueError, match="direction_method must be 'hessian' or 'uniform'"):
        validate_parameters({"direction_method": "invalid"})


def test_extract_edges_uniform_direction_method_skips_hessian(monkeypatch):
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
        "direction_method": "uniform",
    }

    monkeypatch.setattr(
        "slavv_python.core.edges.candidates.estimate_vessel_directions",
        lambda *args, **kwargs: iter(()).throw(
            AssertionError("Hessian estimator should not be called")
        ),
    )

    edges = processor.extract_edges(energy_data, vertices, params)
    assert len(edges["traces"]) == 0
    assert edges["diagnostics"]["candidate_traced_edge_count"] == 2
