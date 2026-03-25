import numpy as np
import pytest

# Add source path for imports
from slavv.core import SLAVVProcessor
from slavv.core.tracing import extract_vertices


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
    energy = np.zeros((8, 8, 8), dtype=np.float32)
    scale_indices = np.zeros_like(energy, dtype=np.int16)

    energy[2, 2, 2] = -5.0
    energy[5, 2, 2] = -4.0
    scale_indices[2, 2, 2] = 10
    scale_indices[5, 2, 2] = 10

    energy_data = {
        "energy": energy,
        "scale_indices": scale_indices,
        "lumen_radius_pixels": np.linspace(1.0, 12.0, 16, dtype=np.float32),
        "lumen_radius_microns": np.linspace(1.0, 12.0, 16, dtype=np.float32),
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
