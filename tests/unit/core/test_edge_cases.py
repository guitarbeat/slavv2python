import numpy as np
import pytest

# Add source path for imports
from slavv.core import SLAVVProcessor, tracing
from slavv.core.tracing import _crop_vertices_matlab_style, extract_vertices


def test_extract_handles_no_vertices():
    processor = SLAVVProcessor()
    energy = np.ones((3, 3, 3), dtype=np.float32)
    energy_data = {
        "energy": energy,
        "scale_indices": np.zeros_like(energy, dtype=np.int16),
        "lumen_radius_pixels": np.array([1.0], dtype=np.float32),
        "lumen_radius_microns": np.array([1.0], dtype=np.float32),
        "energy_sign": -1.0,
    }
    vertices = processor.extract_vertices(energy_data, {})
    assert vertices["positions"].shape == (0, 3)

    edges = processor.extract_edges(energy_data, vertices, {})
    assert edges["connections"].shape == (0, 2)

    network = processor.construct_network(edges, vertices, {})
    assert len(network["adjacency_list"]) == 0
    assert network["orphans"].size == 0


def test_process_image_requires_3d():
    processor = SLAVVProcessor()
    img2d = np.zeros((5, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="non-empty 3D array"):
        processor.process_image(img2d, {})


def test_vertex_extraction_uses_fixed_voxel_suppression_not_radius_overlap():
    energy = np.zeros((40, 40, 40), dtype=np.float32)
    scale_indices = np.zeros_like(energy, dtype=np.int16)

    energy[20, 20, 20] = -5.0
    energy[23, 20, 20] = -4.0
    scale_indices[20, 20, 20] = 10
    scale_indices[23, 20, 20] = 10

    energy_data = {
        "energy": energy,
        "scale_indices": scale_indices,
        "lumen_radius_pixels": np.linspace(1.0, 12.0, 16, dtype=np.float32),
        "lumen_radius_microns": np.linspace(1.0, 12.0, 16, dtype=np.float32),
        "lumen_radius_pixels_axes": np.repeat(
            np.linspace(1.0, 12.0, 16, dtype=np.float32)[:, None], 3, axis=1
        ),
        "energy_sign": -1.0,
    }
    params = {
        "energy_upper_bound": 0.0,
        "space_strel_apothem": 1,
        "microns_per_voxel": [1.0, 1.0, 1.0],
        "length_dilation_ratio": 1.0,
    }

    vertices = extract_vertices(energy_data, params)

    assert len(vertices["positions"]) == 2


def test_crop_vertices_removes_boundary_and_extreme_scales():
    positions = np.array(
        [
            [5, 5, 5],
            [0, 5, 5],
            [5, 5, 5],
            [8, 8, 8],
        ],
        dtype=np.float32,
    )
    scales = np.array([1, 1, 0, 2], dtype=np.int16)
    energies = np.array([-2.0, -1.5, -1.0, -0.5], dtype=np.float32)
    lumen_radius_pixels_axes = np.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
        ],
        dtype=np.float32,
    )

    kept_positions, kept_scales, kept_energies = _crop_vertices_matlab_style(
        positions,
        scales,
        energies,
        lumen_radius_pixels_axes,
        (10, 10, 10),
    )

    assert kept_positions.shape == (1, 3)
    assert kept_scales.tolist() == [1]
    assert kept_energies.tolist() == [-2.0]


def test_extract_edges_uses_matlab_default_max_edge_length(monkeypatch):
    captured = {}

    def fake_trace_edge(
        energy,
        start_pos,
        direction,
        step_size,
        max_edge_energy,
        vertex_positions,
        vertex_scales,
        lumen_radius_pixels,
        lumen_radius_microns,
        max_steps,
        microns_per_voxel,
        energy_sign,
        discrete_steps=False,
        vertex_image=None,
        tree=None,
        max_search_radius=0.0,
    ):
        captured["step_size"] = step_size
        captured["max_steps"] = max_steps
        return [start_pos.copy()]

    monkeypatch.setattr(tracing, "trace_edge", fake_trace_edge)
    monkeypatch.setattr(
        tracing,
        "estimate_vessel_directions",
        lambda energy, pos, radius, mpv: np.array([[0.0, 0.0, 1.0]], dtype=float),
    )

    energy = np.zeros((16, 16, 16), dtype=np.float32)
    energy_data = {
        "energy": energy,
        "lumen_radius_pixels": np.array([2.0, 4.0], dtype=np.float32),
        "lumen_radius_pixels_axes": np.repeat(
            np.array([2.0, 4.0], dtype=np.float32)[:, None], 3, axis=1
        ),
        "lumen_radius_microns": np.array([2.0, 4.0], dtype=np.float32),
        "energy_sign": -1.0,
    }
    vertices = {
        "positions": np.array([[8.0, 8.0, 8.0]], dtype=np.float32),
        "scales": np.array([1], dtype=np.int16),
    }

    tracing.extract_edges(
        energy_data,
        vertices,
        {
            "microns_per_voxel": [1.0, 1.0, 1.0],
            "step_size_per_origin_radius": 1.0,
            "number_of_edges_per_vertex": 1,
            "direction_method": "hessian",
        },
    )

    assert captured["step_size"] == pytest.approx(4.0)
    assert captured["max_steps"] == 60


def test_trace_edge_returns_safely_on_nonfinite_direction():
    energy = np.zeros((8, 8, 8), dtype=np.float32)

    trace = tracing.trace_edge(
        energy=energy,
        start_pos=np.array([4.0, 4.0, 4.0], dtype=np.float32),
        direction=np.array([np.nan, 0.0, 0.0], dtype=np.float32),
        step_size=1.0,
        max_edge_energy=0.0,
        vertex_positions=np.empty((0, 3), dtype=np.float32),
        vertex_scales=np.empty((0,), dtype=np.int16),
        lumen_radius_pixels=np.array([1.0], dtype=np.float32),
        lumen_radius_microns=np.array([1.0], dtype=np.float32),
        max_steps=10,
        microns_per_voxel=np.ones(3, dtype=np.float32),
        energy_sign=-1.0,
    )

    assert len(trace) == 1
    assert np.allclose(trace[0], np.array([4.0, 4.0, 4.0], dtype=np.float32))
