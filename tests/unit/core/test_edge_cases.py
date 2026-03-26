import numpy as np
import pytest

# Add source path for imports
from slavv.core import SLAVVProcessor
from slavv.core.tracing import _choose_edges_matlab_style, extract_vertices


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


def test_vertex_extraction_uses_matlab_paint_selection():
    energy = np.zeros((12, 12, 12), dtype=np.float32)
    scale_indices = np.zeros_like(energy, dtype=np.int16)

    energy[5, 5, 5] = -5.0
    energy[6, 5, 5] = -4.0
    scale_indices[5, 5, 5] = 2
    scale_indices[6, 5, 5] = 2

    energy_data = {
        "energy": energy,
        "scale_indices": scale_indices,
        "lumen_radius_pixels": np.array([0.5, 0.8, 1.0, 1.4, 1.8, 2.2], dtype=np.float32),
        "lumen_radius_pixels_axes": np.tile(
            np.array(
                [
                    [0.5, 0.5, 0.5],
                    [0.8, 0.8, 0.8],
                    [1.0, 1.0, 1.0],
                    [1.4, 1.4, 1.4],
                    [1.8, 1.8, 1.8],
                    [2.2, 2.2, 2.2],
                ],
                dtype=np.float32,
            ),
            (1, 1),
        ),
        "lumen_radius_microns": np.array([0.5, 0.8, 1.0, 1.4, 1.8, 2.2], dtype=np.float32),
        "energy_sign": -1.0,
    }
    params = {
        "energy_upper_bound": 0.0,
        "space_strel_apothem": 1,
        "microns_per_voxel": [1.0, 1.0, 1.0],
        "length_dilation_ratio": 1.0,
    }

    vertices = extract_vertices(energy_data, params)

    assert len(vertices["positions"]) == 1
    assert np.allclose(vertices["positions"][0], [5, 5, 5])


def test_choose_edges_filters_nonterminal_duplicates_and_antiparallel():
    vertex_positions = np.array([[1, 1, 1], [1, 5, 1]], dtype=np.float32)
    vertex_scales = np.array([0, 0], dtype=np.int16)
    candidates = {
        "traces": [
            np.array([[1, 1, 1], [1, 3, 1], [1, 5, 1]], dtype=np.float32),
            np.array([[1, 1, 1], [1, 2, 1], [1, 3, 1]], dtype=np.float32),
            np.array([[1, 1, 1], [1, 3, 1], [1, 5, 1]], dtype=np.float32),
            np.array([[1, 5, 1], [1, 3, 1], [1, 1, 1]], dtype=np.float32),
        ],
        "connections": np.array([[0, 1], [0, -1], [0, 1], [1, 0]], dtype=np.int32),
        "metrics": np.array([-5.0, -6.0, -4.0, -3.0], dtype=np.float32),
        "energy_traces": [
            np.array([-5.0, -5.0, -5.0], dtype=np.float32),
            np.array([-6.0, -6.0, -6.0], dtype=np.float32),
            np.array([-4.0, -4.0, -4.0], dtype=np.float32),
            np.array([-3.0, -3.0, -3.0], dtype=np.float32),
        ],
        "scale_traces": [np.zeros(3, dtype=np.int16) for _ in range(4)],
        "origin_indices": np.array([0, 0, 0, 1], dtype=np.int32),
    }

    chosen = _choose_edges_matlab_style(
        candidates,
        vertex_positions,
        vertex_scales,
        np.array([[0.5, 0.5, 0.5]], dtype=np.float32),
        (8, 8, 8),
        {"number_of_edges_per_vertex": 4},
    )

    assert chosen["connections"].tolist() == [[0, 1]]
    assert chosen["diagnostics"]["dangling_edge_count"] == 1
    assert chosen["diagnostics"]["duplicate_directed_pair_count"] == 1
    assert chosen["diagnostics"]["antiparallel_pair_count"] == 1


def test_choose_edges_prunes_degree_excess_and_cycles():
    vertex_positions = np.array(
        [[1, 1, 1], [1, 5, 1], [5, 5, 1], [5, 1, 1]],
        dtype=np.float32,
    )
    vertex_scales = np.zeros(4, dtype=np.int16)
    candidates = {
        "traces": [
            np.array([[1, 1, 1], [1, 3, 1], [1, 5, 1]], dtype=np.float32),
            np.array([[1, 5, 1], [3, 5, 1], [5, 5, 1]], dtype=np.float32),
            np.array([[5, 5, 1], [3, 3, 1], [1, 1, 1]], dtype=np.float32),
            np.array([[1, 1, 1], [3, 1, 1], [5, 1, 1]], dtype=np.float32),
        ],
        "connections": np.array([[0, 1], [1, 2], [2, 0], [0, 3]], dtype=np.int32),
        "metrics": np.array([-5.0, -4.0, -3.0, -2.0], dtype=np.float32),
        "energy_traces": [np.array([-5.0, -5.0, -5.0], dtype=np.float32) for _ in range(4)],
        "scale_traces": [np.zeros(3, dtype=np.int16) for _ in range(4)],
        "origin_indices": np.array([0, 1, 2, 0], dtype=np.int32),
    }

    chosen = _choose_edges_matlab_style(
        candidates,
        vertex_positions,
        vertex_scales,
        np.array([[0.5, 0.5, 0.5]], dtype=np.float32),
        (8, 8, 8),
        {"number_of_edges_per_vertex": 2},
    )

    assert chosen["connections"].tolist() == [[0, 1], [1, 2]]
    assert chosen["diagnostics"]["degree_pruned_count"] == 1
    assert chosen["diagnostics"]["cycle_pruned_count"] == 1
