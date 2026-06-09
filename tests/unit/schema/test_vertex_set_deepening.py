import numpy as np
import pytest

from slavv_python.schema.results import VertexSet


def test_vertex_set_deepening(tmp_path):
    positions = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    scales = np.array([0, 1], dtype=int)
    energies = np.array([0.5, 0.8], dtype=float)
    lumen_radius_pixels = np.array([1.5, 2.5], dtype=float)
    lumen_radius_microns = np.array([15.0, 25.0], dtype=float)

    # Test create (Authoritative Builder)
    vertex_set = VertexSet.create(
        positions, scales, energies, lumen_radius_pixels, lumen_radius_microns, metadata="test"
    )

    assert vertex_set.positions.dtype == np.float32
    assert vertex_set.scales.dtype == np.int16
    assert vertex_set.energies.dtype == np.float32
    assert len(vertex_set.radii_pixels) == 2
    assert vertex_set.extra["metadata"] == "test"

    # Test save/load (Persistence)
    save_path = tmp_path / "vertices.pkl"
    vertex_set.save(save_path)
    assert save_path.exists()

    loaded_set = VertexSet.load(save_path)
    assert np.array_equal(loaded_set.positions, vertex_set.positions)
    assert np.array_equal(loaded_set.scales, vertex_set.scales)
    assert loaded_set.extra["metadata"] == "test"


def test_vertex_set_validation():
    positions = np.array([[1, 2, 3]], dtype=float)
    scales = np.array([0, 1], dtype=int)  # Mismatch
    energies = np.array([0.5], dtype=float)
    lumen_radius_pixels = np.array([1.5, 2.5], dtype=float)
    lumen_radius_microns = np.array([15.0, 25.0], dtype=float)

    with pytest.raises(ValueError, match="Vertex attribute mismatch"):
        VertexSet.create(positions, scales, energies, lumen_radius_pixels, lumen_radius_microns)
