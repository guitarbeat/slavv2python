import pathlib
import sys

import numpy as np
import pytest

# Add source path for imports
<<<<<<< HEAD
from slavv.core import SLAVVProcessor
=======
from slavv.core.pipeline import SLAVVProcessor
>>>>>>> 02551966425602193b36f418552db1552ddb39ea


def test_extract_handles_no_vertices():
    processor = SLAVVProcessor()
    energy = np.ones((3, 3, 3), dtype=np.float32)
    energy_data = {
        'energy': energy,
        'scale_indices': np.zeros_like(energy, dtype=np.int16),
        'lumen_radius_pixels': np.array([1.0], dtype=np.float32),
        'lumen_radius_microns': np.array([1.0], dtype=np.float32),
        'energy_sign': -1.0,
    }
    vertices = processor.extract_vertices(energy_data, {})
    assert vertices['positions'].shape == (0, 3)

    edges = processor.extract_edges(energy_data, vertices, {})
    assert edges['connections'].shape == (0, 2)

    network = processor.construct_network(edges, vertices, {})
    assert len(network['adjacency_list']) == 0
    assert network['orphans'].size == 0


def test_process_image_requires_3d():
    processor = SLAVVProcessor()
    img2d = np.zeros((5, 5), dtype=np.float32)
    with pytest.raises(ValueError):
        processor.process_image(img2d, {})

