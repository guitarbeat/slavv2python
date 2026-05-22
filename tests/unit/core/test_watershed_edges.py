import numpy as np

from slavv_python.engine import SlavvPipeline
from slavv_python.schema.results import EnergyResult, VertexSet


def test_extract_edges_watershed_two_vertices():
    processor = SlavvPipeline()
    energy = np.ones((5, 5, 5), dtype=np.float32)
    energy_data = EnergyResult.from_dict({
        "energy": energy,
        "scale_indices": np.zeros_like(energy, dtype=np.int16),
        "lumen_radius_pixels": np.array([1.0], dtype=np.float32),
        "lumen_radius_microns": np.array([1.0], dtype=np.float32),
        "energy_sign": -1.0,
    })
    vertices = VertexSet.from_dict({
        "positions": np.array([[0, 0, 0], [4, 4, 4]], dtype=float),
    })
    edges = processor.extract_edges_watershed(energy_data, vertices, {})
    assert edges.connections.shape == (1, 2)
    assert edges.connections[0].tolist() == [0, 1]
    assert len(edges.traces) == 1
    assert edges.traces[0].shape[1] == 3
    assert np.isclose(edges.energies[0], 1.0)


def test_extract_edges_watershed_empty_connections_keep_two_columns():
    processor = SlavvPipeline()
    energy = np.ones((5, 5, 5), dtype=np.float32)
    energy_data = EnergyResult.from_dict({
        "energy": energy,
        "scale_indices": np.zeros_like(energy, dtype=np.int16),
        "lumen_radius_pixels": np.array([1.0], dtype=np.float32),
        "lumen_radius_microns": np.array([1.0], dtype=np.float32),
        "energy_sign": -1.0,
    })
    vertices = VertexSet.from_dict({
        "positions": np.array([[2, 2, 2]], dtype=float),
    })

    edges = processor.extract_edges_watershed(energy_data, vertices, {})

    assert edges.connections.shape == (0, 2)
