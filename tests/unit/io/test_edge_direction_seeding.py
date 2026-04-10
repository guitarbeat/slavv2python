from unittest.mock import patch

import numpy as np

from slavv.core import SLAVVProcessor


@patch(
    "slavv.core.tracing.estimate_vessel_directions",
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
    assert edges["diagnostics"]["candidate_traced_edge_count"] == 2
    assert len(edges["traces"]) == 0
    for trace in [
        np.array([[10.0, 10.0, 10.0], [10.0, 14.0, 10.0], [10.0, 18.0, 10.0]], dtype=float),
        np.array([[10.0, 10.0, 10.0], [10.0, 6.0, 10.0], [10.0, 2.0, 10.0]], dtype=float),
    ]:
        trace = np.asarray(trace)
        # y and z should remain constant while x moves monotonically
        assert np.allclose(trace[:, 0], 10.0)
        assert np.allclose(trace[:, 2], 10.0)
        x_diff = np.diff(trace[:, 1])
        assert np.all(x_diff > 0) or np.all(x_diff < 0)


def test_extract_edges_direction_padding_is_repeatable(monkeypatch):
    processor = SLAVVProcessor()

    energy = -np.ones((7, 7, 7), dtype=float)
    energy_data = {
        "energy": energy,
        "lumen_radius_pixels": np.array([1.0], dtype=float),
        "lumen_radius_microns": np.array([1.0], dtype=float),
        "lumen_radius_pixels_axes": np.array([[1.0, 1.0, 1.0]], dtype=float),
        "energy_sign": -1.0,
    }
    vertices = {
        "positions": np.array([[3.0, 1.0, 3.0], [3.0, 5.0, 3.0]], dtype=float),
        "scales": np.array([0, 0], dtype=int),
    }
    params = {
        "number_of_edges_per_vertex": 2,
        "step_size_per_origin_radius": 1.5,
        "max_edge_length_per_origin_radius": 3.0,
        "microns_per_voxel": [1.0, 1.0, 1.0],
    }

    seed_calls: list[int | None] = []
    unseeded_call_count = {"value": 0}

    def fake_estimate(*args, **kwargs):
        return np.array([[0.0, 1.0, 0.0]], dtype=float)

    def fake_generate(n_directions, seed=None):
        seed_calls.append(seed)
        if seed is None:
            unseeded_call_count["value"] += 1
            seed_value = unseeded_call_count["value"]
        else:
            seed_value = int(seed)
        sign = 1.0 if seed_value % 2 == 0 else -1.0
        return np.repeat(np.array([[sign, 0.0, 0.0]], dtype=float), n_directions, axis=0)

    def fake_trace_edge(*args, **kwargs):
        start_pos = np.asarray(args[1], dtype=np.float32)
        direction = np.asarray(args[2], dtype=np.float32)
        trace = [
            start_pos.copy(),
            start_pos + direction.astype(np.float32, copy=False),
        ]
        terminal_vertex = 1 if direction[0] > 0 else 0
        metadata = {
            "terminal_vertex": terminal_vertex,
            "stop_reason": "direct_terminal_hit",
            "terminal_resolution": "direct_hit",
        }
        return trace, metadata

    monkeypatch.setattr("slavv.core.tracing.estimate_vessel_directions", fake_estimate)
    monkeypatch.setattr("slavv.core.tracing.generate_edge_directions", fake_generate)
    monkeypatch.setattr("slavv.core.tracing.trace_edge", fake_trace_edge)

    first = processor.extract_edges(energy_data, vertices, params)
    second = processor.extract_edges(energy_data, vertices, params)

    assert seed_calls == [0, 1, 0, 1]
    assert first["connections"].tolist() == second["connections"].tolist()
    assert len(first["traces"]) == len(second["traces"])
    for trace_a, trace_b in zip(first["traces"], second["traces"]):
        assert np.allclose(trace_a, trace_b)
    assert (
        first["diagnostics"]["candidate_traced_edge_count"]
        == second["diagnostics"]["candidate_traced_edge_count"]
    )
